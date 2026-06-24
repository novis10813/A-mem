"""Embedding-field ablation runner for A-MEM LoCoMo evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from llm_text_parsers import parse_plain_text_answer
from load_dataset import load_locomo_dataset
from memory_layer import SimpleEmbeddingRetriever
from test_advanced_robust import RobustAdvancedMemAgent
from utils import aggregate_metrics, calculate_metrics


FieldTuple = Tuple[str, ...]

VARIANT_FIELDS: "OrderedDict[str, FieldTuple]" = OrderedDict(
    [
        ("content", ("content",)),
        ("content_context", ("content", "context")),
        ("content_keywords", ("content", "keywords")),
        ("content_tags", ("content", "tags")),
        ("content_keywords_tags", ("content", "keywords", "tags")),
        ("content_context_keywords", ("content", "context", "keywords")),
        ("full", ("content", "context", "keywords", "tags")),
    ]
)
SUMMARY_METRICS = ("f1", "bleu1")

logger = logging.getLogger("amem_ablation")


def build_embedding_text(memory, fields: Sequence[str]) -> str:
    """Build the labeled text passed to SentenceTransformer for one memory."""
    chunks: List[str] = []
    for field in fields:
        if field == "content":
            value = str(getattr(memory, "content", "")).strip()
            if value:
                chunks.append(f"content: {value}")
        elif field == "context":
            value = str(getattr(memory, "context", "")).strip()
            if value and value != "General":
                chunks.append(f"context: {value}")
        elif field == "keywords":
            value = _format_list_field(getattr(memory, "keywords", []))
            if value:
                chunks.append(f"keywords: {value}")
        elif field == "tags":
            value = _format_list_field(getattr(memory, "tags", []))
            if value:
                chunks.append(f"tags: {value}")
        else:
            raise ValueError(f"Unknown embedding field: {field}")
    return " ".join(chunks)


def _format_list_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return ", ".join(str(item).strip() for item in value if str(item).strip())


def expand_variants(spec: str) -> "OrderedDict[str, FieldTuple]":
    """Expand a variant preset or comma-separated variant list."""
    if spec == "core7":
        return OrderedDict(VARIANT_FIELDS)

    selected: "OrderedDict[str, FieldTuple]" = OrderedDict()
    for name in [part.strip() for part in spec.split(",") if part.strip()]:
        if name not in VARIANT_FIELDS:
            known = ", ".join(["core7", *VARIANT_FIELDS.keys()])
            raise ValueError(f"Unknown variant '{name}'. Known variants: {known}")
        selected[name] = VARIANT_FIELDS[name]

    if not selected:
        raise ValueError("At least one ablation variant is required")
    return selected


def load_cached_memories(memory_cache_dir: Path, sample_idx: int) -> Dict:
    cache_file = memory_cache_dir / f"memory_cache_sample_{sample_idx}.pkl"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Missing {cache_file}. Run the original benchmark first to create memory caches."
        )
    with cache_file.open("rb") as f:
        return pickle.load(f)


def build_retriever_from_memories(
    memories: Mapping,
    fields: Sequence[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> SimpleEmbeddingRetriever:
    retriever = SimpleEmbeddingRetriever(model_name)
    documents = [build_embedding_text(memory, fields) for memory in memories.values()]
    retriever.add_documents(documents)
    return retriever


def build_answer_prompt(
    context: str,
    question: str,
    category: int,
    answer: str,
    temperature_c5: float,
    answer_options: Optional[Sequence[str]] = None,
):
    """Build the answer prompt used by the robust benchmark."""
    if category == 5:
        if answer_options is None:
            answer_options = ["Not mentioned in the conversation", answer]
            random.shuffle(answer_options)
        prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {answer_options[0]} or {answer_options[1]}  Short answer:"""
        return prompt, temperature_c5

    if category == 2:
        prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
        return prompt, 0.7

    prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
    return prompt, 0.7


def answer_question_with_keywords(
    agent,
    keywords: str,
    question: str,
    category: int,
    answer: str,
    answer_options: Optional[Sequence[str]] = None,
):
    raw_context = agent.retrieve_memory(keywords, k=agent.retrieve_k)
    user_prompt, temperature = build_answer_prompt(
        raw_context, question, category, answer, agent.temperature_c5, answer_options
    )
    try:
        response = agent.memory_system.llm_controller.llm.get_completion(
            user_prompt, temperature=temperature
        )
    except Exception as e:
        logger.warning("answer_question failed: %s - returning empty", e)
        response = ""
    return response, user_prompt, raw_context


def evaluate_ablation(
    dataset_path: Path,
    memory_cache_dir: Path,
    variants: Mapping[str, FieldTuple],
    output_dir: Path,
    model: str,
    backend: str,
    ratio: float = 1.0,
    retrieve_k: int = 10,
    temperature_c5: float = 0.5,
    sglang_host: str = "http://localhost",
    sglang_port: int = 30000,
    dry_run: bool = False,
) -> List[Dict]:
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_locomo_dataset(dataset_path)
    if ratio < 1.0:
        samples = samples[: max(1, int(len(samples) * ratio))]

    results_by_variant = {
        variant_name: _empty_result(
            variant_name, fields, model, backend, dataset_path, memory_cache_dir, dry_run
        )
        for variant_name, fields in variants.items()
    }

    for sample_idx, sample in enumerate(samples):
        logger.info("Loading cached memories for sample %s", sample_idx)
        memories = load_cached_memories(memory_cache_dir, sample_idx)
        variant_retrievers = {
            variant_name: build_retriever_from_memories(memories, fields)
            for variant_name, fields in variants.items()
        }

        if dry_run:
            for variant_name, result in results_by_variant.items():
                result["samples"].append(
                    {
                        "sample_id": sample_idx,
                        "memory_count": len(memories),
                        "retriever_document_count": len(variant_retrievers[variant_name].corpus),
                    }
                )
            continue

        agent = RobustAdvancedMemAgent(
            model,
            backend,
            retrieve_k,
            temperature_c5,
            sglang_host,
            sglang_port,
        )
        agent.memory_system.memories = memories

        eligible_qas = [qa for qa in sample.qa if int(qa.category) in [1, 2, 3, 4, 5]]
        query_keywords = {}
        category5_options = {}
        for qa_idx, qa in enumerate(eligible_qas):
            query_keywords[qa_idx] = agent.generate_query_llm(qa.question)
            if int(qa.category) == 5:
                options = ["Not mentioned in the conversation", qa.final_answer]
                random.shuffle(options)
                category5_options[qa_idx] = tuple(options)

        for variant_name, retriever in variant_retrievers.items():
            logger.info("Evaluating sample %s variant %s", sample_idx, variant_name)
            agent.memory_system.retriever = retriever
            result = results_by_variant[variant_name]

            for qa_idx, qa in enumerate(eligible_qas):
                prediction, user_prompt, raw_context = answer_question_with_keywords(
                    agent,
                    query_keywords[qa_idx],
                    qa.question,
                    qa.category,
                    qa.final_answer,
                    category5_options.get(qa_idx),
                )
                prediction = parse_plain_text_answer(prediction)
                metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else _zero_metrics()

                result["total_questions"] += 1
                result["category_distribution"][str(qa.category)] += 1
                result["_all_metrics"].append(metrics)
                result["_all_categories"].append(qa.category)
                result["individual_results"].append(
                    {
                        "sample_id": sample_idx,
                        "question": qa.question,
                        "query_keywords": query_keywords[qa_idx],
                        "prediction": prediction,
                        "reference": qa.final_answer,
                        "category": qa.category,
                        "metrics": metrics,
                        "user_prompt": user_prompt,
                        "raw_context": raw_context,
                    }
                )

    final_results = []
    for variant_name, result in results_by_variant.items():
        result["category_distribution"] = dict(result["category_distribution"])
        if not dry_run:
            result["aggregate_metrics"] = aggregate_metrics(
                result.pop("_all_metrics"), result.pop("_all_categories")
            )
        else:
            result.pop("_all_metrics")
            result.pop("_all_categories")

        output_file = output_dir / f"{variant_name}.json"
        with output_file.open("w") as f:
            json.dump(result, f, indent=2)
        final_results.append(result)

    write_summary_files(final_results, output_dir)
    return final_results


def _empty_result(
    variant_name: str,
    fields: FieldTuple,
    model: str,
    backend: str,
    dataset_path: Path,
    memory_cache_dir: Path,
    dry_run: bool,
) -> Dict:
    return {
        "variant": variant_name,
        "embedding_fields": list(fields),
        "model": model,
        "backend": backend,
        "dataset": str(dataset_path),
        "memory_cache_dir": str(memory_cache_dir),
        "dry_run": dry_run,
        "total_questions": 0,
        "category_distribution": defaultdict(int),
        "individual_results": [],
        "samples": [],
        "_all_metrics": [],
        "_all_categories": [],
    }


def _zero_metrics() -> Dict[str, float]:
    return {
        "exact_match": 0,
        "f1": 0.0,
        "rouge1_f": 0.0,
        "rouge2_f": 0.0,
        "rougeL_f": 0.0,
        "bleu1": 0.0,
        "bleu2": 0.0,
        "bleu3": 0.0,
        "bleu4": 0.0,
        "bert_f1": 0.0,
        "meteor": 0.0,
        "sbert_similarity": 0.0,
    }


def extract_summary_rows(result: Mapping) -> List[Dict]:
    rows: List[Dict] = []
    aggregate_metrics = result.get("aggregate_metrics", {})
    for split, metrics in aggregate_metrics.items():
        row = {
            "variant": result["variant"],
            "split": split,
            "total_questions": result["total_questions"],
        }
        for metric_name in SUMMARY_METRICS:
            row[metric_name] = metrics.get(metric_name, {}).get("mean", "")
        rows.append(row)
    return rows


def write_summary_files(results: Sequence[Mapping], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"

    summary = {
        "variants": [
            {
                "variant": result["variant"],
                "embedding_fields": result.get("embedding_fields", []),
                "total_questions": result["total_questions"],
                "category_distribution": result.get("category_distribution", {}),
                "aggregate_metrics": result.get("aggregate_metrics", {}),
                "dry_run": result.get("dry_run", False),
                "samples": result.get("samples", []),
            }
            for result in results
        ]
    }
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    rows = [row for result in results for row in extract_summary_rows(result)]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "split", "total_questions", *SUMMARY_METRICS]
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A-MEM embedding field ablations")
    parser.add_argument("--dataset", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument("--backend", type=str, default="ollama")
    parser.add_argument("--model", type=str, default="llama3.2:latest")
    parser.add_argument("--memory-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--sglang_host", type=str, default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--variants", type=str, default="core7")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    variants = expand_variants(args.variants)
    evaluate_ablation(
        dataset_path=args.dataset,
        memory_cache_dir=args.memory_cache_dir,
        variants=variants,
        output_dir=args.output_dir,
        model=args.model,
        backend=args.backend,
        ratio=args.ratio,
        retrieve_k=args.retrieve_k,
        temperature_c5=args.temperature_c5,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
