#!/usr/bin/env python3
"""Run paired content+keyword pruning experiments on fixed A-MEM caches."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from amem.llm_text_parsers import (
    parse_keywords_response,
    parse_plain_text_answer,
    sanitize_keywords,
    sanitize_keywords_none,
    set_keyword_pruning_mode,
)
from amem.load_dataset import load_locomo_dataset
from amem.memory_layer_robust import RobustLLMController


CONDITIONS = ("none", "nltk")
METRICS = ("f1", "bleu1")


@dataclass
class SampleState:
    sample_idx: int
    memory_items: list[tuple[str, Any]]
    keyword_maps: dict[str, dict[str, list[str]]]
    retrievers: dict[str, "ContentKeywordRetriever"]


class ContentKeywordRetriever:
    """Embedding retriever for pre-rendered content+keyword documents."""

    def __init__(self, model: SentenceTransformer, documents: Sequence[str]):
        self.documents = list(documents)
        self.embeddings = (
            model.encode(self.documents, show_progress_bar=False)
            if self.documents
            else None
        )

    def search(self, query: str, k: int) -> list[int]:
        if not self.documents or self.embeddings is None:
            return []
        query_embedding = model_encode_one(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        return [int(i) for i in np.argsort(similarities)[-k:][::-1]]


_EMBEDDING_MODEL: SentenceTransformer | None = None
_EMBEDDING_LOCK = Lock()


def model_encode_one(query: str) -> np.ndarray:
    if _EMBEDDING_MODEL is None:
        raise RuntimeError("Embedding model has not been initialized")
    with _EMBEDDING_LOCK:
        return _EMBEDDING_MODEL.encode([query], show_progress_bar=False)[0]


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def simple_tokenize(text: str) -> list[str]:
    return (
        str(text)
        .lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def calculate_f1_bleu1(prediction: str, reference: str) -> dict[str, float]:
    if not prediction or not reference:
        return {"f1": 0.0, "bleu1": 0.0}

    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens or not common_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)

    pred_seq = simple_tokenize(prediction)
    ref_seq = [simple_tokenize(reference)]
    try:
        bleu1 = sentence_bleu(
            ref_seq,
            pred_seq,
            weights=(1, 0, 0, 0),
            smoothing_function=SmoothingFunction().method1,
        )
    except Exception:
        bleu1 = 0.0

    return {"f1": f1, "bleu1": bleu1}


def aggregate_metrics(
    all_metrics: Sequence[Mapping[str, float]],
    all_categories: Sequence[int],
) -> dict[str, dict[str, dict[str, float]]]:
    aggregates: dict[str, list[float]] = defaultdict(list)
    category_aggregates: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(float(value))
            category_aggregates[int(category)][metric_name].append(float(value))

    result: dict[str, dict[str, dict[str, float]]] = {"overall": {}}
    for metric_name, values in aggregates.items():
        result["overall"][metric_name] = summarize_values(values)

    for category in sorted(category_aggregates):
        split = f"category_{category}"
        result[split] = {}
        for metric_name, values in category_aggregates[category].items():
            result[split][metric_name] = summarize_values(values)
    return result


def summarize_values(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
        }
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def build_answer_prompt(
    context: str,
    question: str,
    category: int,
    answer: str,
    temperature_c5: float,
    answer_options: Sequence[str] | None = None,
) -> tuple[str, float]:
    if category == 5:
        options = list(answer_options or ["Not mentioned in the conversation", answer])
        prompt = f"""Based on the context: {context}, answer the following question. {question}

Select the correct answer: {options[0]} or {options[1]}  Short answer:"""
        return prompt, temperature_c5

    if category == 2:
        prompt = f"""Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:"""
        return prompt, 0.7

    prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:"""
    return prompt, 0.7


def generate_query_keywords(llm, question: str) -> str:
    prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""
    response = llm.get_completion(prompt)
    return parse_keywords_response(response)


def answer_question(
    llm,
    context: str,
    question: str,
    category: int,
    answer: str,
    temperature_c5: float,
    answer_options: Sequence[str] | None,
) -> tuple[str, str]:
    prompt, temperature = build_answer_prompt(
        context, question, category, answer, temperature_c5, answer_options
    )
    try:
        response = llm.get_completion(prompt, temperature=temperature)
    except Exception as exc:
        logging.warning("answer_question failed: %s; returning empty", exc)
        response = ""
    return parse_plain_text_answer(response), prompt


def load_memories(memory_cache_dir: Path, sample_idx: int) -> dict[str, Any]:
    cache_file = memory_cache_dir / f"memory_cache_sample_{sample_idx}.pkl"
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing memory cache file: {cache_file}")
    with cache_file.open("rb") as handle:
        return pickle.load(handle)


def transform_keywords(note: Any, condition: str, max_keywords: int) -> list[str]:
    raw_keywords = getattr(note, "keywords", []) or []
    content = getattr(note, "content", "") or ""
    if condition == "none":
        return sanitize_keywords_none(content, raw_keywords, max_keywords=max_keywords)
    if condition == "nltk":
        set_keyword_pruning_mode("nltk")
        return sanitize_keywords(content, raw_keywords, max_keywords=max_keywords)
    raise ValueError(f"Unknown condition: {condition}")


def build_document(note: Any, keywords: Sequence[str]) -> str:
    content = str(getattr(note, "content", "") or "").strip()
    keyword_text = ", ".join(str(keyword).strip() for keyword in keywords if str(keyword).strip())
    return f"content: {content} keywords: {keyword_text}".strip()


def render_context(
    memory_items: Sequence[tuple[str, Any]],
    keyword_map: Mapping[str, Sequence[str]],
    indices: Sequence[int],
) -> str:
    lines = []
    for rank, index in enumerate(indices, start=1):
        note_id, note = memory_items[int(index)]
        timestamp = str(getattr(note, "timestamp", "") or "")
        content = str(getattr(note, "content", "") or "")
        keywords = ", ".join(keyword_map.get(note_id, []))
        lines.append(
            f"memory rank: {rank}\n"
            f"talk start time: {timestamp}\n"
            f"memory content: {content}\n"
            f"memory keywords: {keywords}"
        )
    return "\n\n".join(lines)


def prepare_sample_states(
    samples: Sequence[Any],
    memory_cache_dir: Path,
    embedding_model: SentenceTransformer,
    max_keywords: int,
) -> list[SampleState]:
    states = []
    for sample_idx, _sample in enumerate(samples):
        logging.info("Preparing sample %s", sample_idx)
        memories = load_memories(memory_cache_dir, sample_idx)
        memory_items = list(memories.items())
        keyword_maps: dict[str, dict[str, list[str]]] = {}
        retrievers: dict[str, ContentKeywordRetriever] = {}

        for condition in CONDITIONS:
            keyword_map = {
                note_id: transform_keywords(note, condition, max_keywords)
                for note_id, note in memory_items
            }
            documents = [
                build_document(note, keyword_map[note_id])
                for note_id, note in memory_items
            ]
            keyword_maps[condition] = keyword_map
            retrievers[condition] = ContentKeywordRetriever(embedding_model, documents)

        states.append(
            SampleState(
                sample_idx=sample_idx,
                memory_items=memory_items,
                keyword_maps=keyword_maps,
                retrievers=retrievers,
            )
        )
    return states


def init_condition_results(run_idx: int, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        condition: {
            "run": run_idx,
            "condition": condition,
            "embedding_fields": ["content", "keywords"],
            "context_fields": ["timestamp", "content", "keywords"],
            "keyword_source": "stored_cache_keywords",
            "source_memory_cache_dir": str(args.memory_cache_dir),
            "model": args.model,
            "backend": args.backend,
            "dataset": str(args.dataset),
            "retrieve_k": args.retrieve_k,
            "temperature_c5": args.temperature_c5,
            "max_keywords": args.max_keywords,
            "total_questions": 0,
            "category_distribution": Counter(),
            "individual_results": [],
            "_all_metrics": [],
            "_all_categories": [],
        }
        for condition in CONDITIONS
    }


def evaluate_sample_run(
    run_idx: int,
    sample: Any,
    state: SampleState,
    llm: Any | None,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    if llm is None:
        llm = RobustLLMController(
            backend=args.backend,
            model=args.model,
            sglang_host=args.sglang_host,
            sglang_port=args.sglang_port,
        ).llm

    rng = random.Random(args.seed + run_idx * 1000 + state.sample_idx)
    result_by_condition = init_condition_results(run_idx, args)

    eligible_qas = [qa for qa in sample.qa if int(qa.category) in [1, 2, 3, 4, 5]]
    if args.qa_limit is not None:
        eligible_qas = eligible_qas[: args.qa_limit]
    logging.info("Run %s sample %s: %s QAs", run_idx, state.sample_idx, len(eligible_qas))

    for qa_idx, qa in enumerate(eligible_qas):
        query_keywords = generate_query_keywords(llm, qa.question)
        answer_options = None
        if int(qa.category) == 5:
            options = ["Not mentioned in the conversation", qa.final_answer]
            rng.shuffle(options)
            answer_options = tuple(options)

        for condition in CONDITIONS:
            retriever = state.retrievers[condition]
            indices = retriever.search(query_keywords, args.retrieve_k)
            raw_context = render_context(
                state.memory_items, state.keyword_maps[condition], indices
            )
            prediction, user_prompt = answer_question(
                llm,
                raw_context,
                qa.question,
                int(qa.category),
                qa.final_answer,
                args.temperature_c5,
                answer_options,
            )
            metrics = calculate_f1_bleu1(prediction, qa.final_answer)
            result = result_by_condition[condition]
            result["total_questions"] += 1
            result["category_distribution"][str(qa.category)] += 1
            result["_all_metrics"].append(metrics)
            result["_all_categories"].append(int(qa.category))
            result["individual_results"].append(
                {
                    "sample_id": state.sample_idx,
                    "qa_idx": qa_idx,
                    "question": qa.question,
                    "query_keywords": query_keywords,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": int(qa.category),
                    "metrics": metrics,
                    "retrieved_indices": indices,
                    "raw_context": raw_context,
                    "user_prompt": user_prompt,
                }
            )

    return result_by_condition


def merge_sample_results(
    run_idx: int,
    args: argparse.Namespace,
    sample_results: Sequence[Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    merged = init_condition_results(run_idx, args)
    for sample_result in sample_results:
        for condition in CONDITIONS:
            target = merged[condition]
            source = sample_result[condition]
            target["total_questions"] += source["total_questions"]
            target["category_distribution"].update(source["category_distribution"])
            target["individual_results"].extend(source["individual_results"])
            target["_all_metrics"].extend(source["_all_metrics"])
            target["_all_categories"].extend(source["_all_categories"])

    for condition, result in merged.items():
        result["individual_results"].sort(
            key=lambda row: (row["sample_id"], row["qa_idx"])
        )
        result["category_distribution"] = dict(result["category_distribution"])
        result["aggregate_metrics"] = aggregate_metrics(
            result.pop("_all_metrics"), result.pop("_all_categories")
        )
    return merged


def evaluate_run(
    run_idx: int,
    samples: Sequence[Any],
    sample_states: Sequence[SampleState],
    llm,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    if args.max_workers == 1:
        sample_results = [
            evaluate_sample_run(run_idx, sample, state, llm, args)
            for sample, state in zip(samples, sample_states)
        ]
    else:
        worker_count = min(args.max_workers, len(samples))
        logging.info("Run %s sample workers: %s", run_idx, worker_count)
        sample_results_by_idx = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(evaluate_sample_run, run_idx, sample, state, None, args): state.sample_idx
                for sample, state in zip(samples, sample_states)
            }
            for future in as_completed(futures):
                sample_idx = futures[future]
                sample_results_by_idx[sample_idx] = future.result()
                logging.info(
                    "Run %s sample %s completed (%s/%s)",
                    run_idx,
                    sample_idx,
                    len(sample_results_by_idx),
                    len(futures),
                )
        sample_results = [
            sample_results_by_idx[state.sample_idx] for state in sample_states
        ]

    return merge_sample_results(run_idx, args, sample_results)


def write_run_results(
    output_dir: Path,
    run_idx: int,
    result_by_condition: Mapping[str, Mapping[str, Any]],
) -> None:
    run_dir = output_dir / f"run_{run_idx:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for condition, result in result_by_condition.items():
        with (run_dir / f"{condition}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


def run_complete(output_dir: Path, run_idx: int) -> bool:
    run_dir = output_dir / f"run_{run_idx:02d}"
    return all((run_dir / f"{condition}.json").exists() for condition in CONDITIONS)


def load_existing_run_results(output_dir: Path) -> dict[int, dict[str, dict[str, Any]]]:
    runs: dict[int, dict[str, dict[str, Any]]] = {}
    for run_dir in sorted(output_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        try:
            run_idx = int(run_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        condition_results = {}
        for condition in CONDITIONS:
            path = run_dir / f"{condition}.json"
            if not path.exists():
                break
            with path.open(encoding="utf-8") as handle:
                condition_results[condition] = json.load(handle)
        if len(condition_results) == len(CONDITIONS):
            runs[run_idx] = condition_results
    return runs


def write_cross_run_summary(output_dir: Path) -> None:
    runs = load_existing_run_results(output_dir)
    summary_rows = []
    delta_rows = []

    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for run_idx, condition_results in runs.items():
        for condition, result in condition_results.items():
            for split, metrics in result.get("aggregate_metrics", {}).items():
                for metric in METRICS:
                    values[(condition, split, metric)].append(
                        metrics.get(metric, {}).get("mean", 0.0)
                    )

        none_result = condition_results.get("none", {})
        nltk_result = condition_results.get("nltk", {})
        for split, none_metrics in none_result.get("aggregate_metrics", {}).items():
            nltk_metrics = nltk_result.get("aggregate_metrics", {}).get(split, {})
            for metric in METRICS:
                none_value = none_metrics.get(metric, {}).get("mean", 0.0)
                nltk_value = nltk_metrics.get(metric, {}).get("mean", 0.0)
                delta_rows.append(
                    {
                        "run": run_idx,
                        "split": split,
                        "metric": metric,
                        "delta_nltk_minus_none": nltk_value - none_value,
                    }
                )

    for (condition, split, metric), metric_values in sorted(values.items()):
        stats = summarize_values(metric_values)
        summary_rows.append(
            {
                "condition": condition,
                "split": split,
                "metric": metric,
                "runs": len(metric_values),
                "mean_across_runs": stats["mean"],
                "std_across_runs": stats["std"],
                "median_across_runs": stats["median"],
                "min_across_runs": stats["min"],
                "max_across_runs": stats["max"],
            }
        )

    grouped_deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in delta_rows:
        grouped_deltas[(row["split"], row["metric"])].append(row["delta_nltk_minus_none"])

    delta_summary_rows = []
    for (split, metric), metric_values in sorted(grouped_deltas.items()):
        stats = summarize_values(metric_values)
        delta_summary_rows.append(
            {
                "split": split,
                "metric": metric,
                "runs": len(metric_values),
                "mean_delta_nltk_minus_none": stats["mean"],
                "std_delta": stats["std"],
                "median_delta": stats["median"],
                "min_delta": stats["min"],
                "max_delta": stats["max"],
                "positive_runs": sum(1 for value in metric_values if value > 0),
            }
        )

    write_csv(output_dir / "summary_across_runs.csv", summary_rows)
    write_csv(output_dir / "delta_nltk_minus_none_by_run.csv", delta_rows)
    write_csv(output_dir / "delta_nltk_minus_none_summary.csv", delta_summary_rows)

    with (output_dir / "summary_across_runs.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "completed_runs": sorted(runs),
                "summary": summary_rows,
                "delta_summary": delta_summary_rows,
            },
            handle,
            indent=2,
        )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare content+keyword retrieval with raw vs NLTK-pruned keywords"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument(
        "--memory-cache-dir",
        type=Path,
        default=Path("artifacts/caches/cached_memories_robust_ollama_llama3.2:1b"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/content_keyword_pruning/ollama_llama3.2-1b_k10"),
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--max-keywords", type=int, default=5)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--qa-limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--sglang_host", default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    global _EMBEDDING_MODEL

    args = parse_args()
    args.dataset = repo_path(args.dataset)
    args.memory_cache_dir = repo_path(args.memory_cache_dir)
    args.output_dir = repo_path(args.output_dir)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.runs < 0:
        raise ValueError("--runs must be >= 0")
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("--ratio must be between 0.0 and 1.0")
    if args.retrieve_k < 1:
        raise ValueError("--retrieve_k must be >= 1")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Loading dataset: %s", args.dataset)
    samples = load_locomo_dataset(args.dataset)
    if args.ratio < 1.0:
        samples = samples[: max(1, int(len(samples) * args.ratio))]
    if args.sample_limit is not None:
        samples = samples[: args.sample_limit]

    logging.info("Loading embedding model: %s", args.embedding_model)
    _EMBEDDING_MODEL = SentenceTransformer(args.embedding_model)
    sample_states = prepare_sample_states(
        samples, args.memory_cache_dir, _EMBEDDING_MODEL, args.max_keywords
    )

    llm_controller = None
    if args.runs > 0 and args.max_workers == 1:
        llm_controller = RobustLLMController(
            backend=args.backend,
            model=args.model,
            sglang_host=args.sglang_host,
            sglang_port=args.sglang_port,
        )

    for run_idx in range(args.runs):
        if args.resume and run_complete(args.output_dir, run_idx):
            logging.info("Skipping completed run %s", run_idx)
            continue

        random.seed(args.seed + run_idx)
        logging.info("Starting run %s/%s", run_idx + 1, args.runs)
        shared_llm = llm_controller.llm if llm_controller is not None else None
        result_by_condition = evaluate_run(
            run_idx, samples, sample_states, shared_llm, args
        )
        write_run_results(args.output_dir, run_idx, result_by_condition)
        write_cross_run_summary(args.output_dir)
        logging.info("Finished run %s", run_idx)

    write_cross_run_summary(args.output_dir)
    logging.info("Done. Results: %s", args.output_dir)


if __name__ == "__main__":
    main()
