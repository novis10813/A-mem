#!/usr/bin/env python3
"""Rebuild A-MEM memories for each pruning condition and evaluate content+keywords."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_content_keyword_pruning_experiment as ck
from amem.llm_text_parsers import set_keyword_pruning_mode
from amem.load_dataset import load_locomo_dataset
from amem.memory_layer_robust import RobustAgenticMemorySystem, RobustLLMController


CONDITIONS = ("none", "nltk")


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def eligible_qas(sample: Any, qa_limit: int | None) -> list[Any]:
    qas = [qa for qa in sample.qa if int(qa.category) in [1, 2, 3, 4, 5]]
    return qas[:qa_limit] if qa_limit is not None else qas


def build_query_plan_for_sample(
    run_idx: int,
    sample_idx: int,
    sample: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    llm = RobustLLMController(
        backend=args.backend,
        model=args.model,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
    ).llm
    rng = random.Random(args.seed + run_idx * 1000 + sample_idx)
    plan = []
    for qa_idx, qa in enumerate(eligible_qas(sample, args.qa_limit)):
        answer_options = None
        if int(qa.category) == 5:
            options = ["Not mentioned in the conversation", qa.final_answer]
            rng.shuffle(options)
            answer_options = tuple(options)
        plan.append(
            {
                "qa_idx": qa_idx,
                "query_keywords": ck.generate_query_keywords(llm, qa.question),
                "answer_options": answer_options,
            }
        )
    return plan


def build_query_plan(
    run_idx: int,
    samples: Sequence[Any],
    args: argparse.Namespace,
) -> dict[int, list[dict[str, Any]]]:
    worker_count = min(args.max_workers, len(samples))
    logging.info("Run %s query workers: %s", run_idx, worker_count)
    plans: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(build_query_plan_for_sample, run_idx, sample_idx, sample, args): sample_idx
            for sample_idx, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            sample_idx = futures[future]
            plans[sample_idx] = future.result()
            logging.info(
                "Run %s query sample %s completed (%s/%s)",
                run_idx,
                sample_idx,
                len(plans),
                len(futures),
            )
    return plans


def conversation_turns(sample: Any, turn_limit: int | None = None) -> list[tuple[str, Any]]:
    turns = [
        (session.date_time, turn)
        for _, session in sample.conversation.sessions.items()
        for turn in session.turns
    ]
    return turns[:turn_limit] if turn_limit is not None else turns


def rebuild_memories_for_sample(
    sample: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    agent = RobustAgenticMemorySystem(
        model_name=args.embedding_model,
        llm_backend=args.backend,
        llm_model=args.model,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
    )
    for turn_datetime, turn in conversation_turns(sample, args.turn_limit):
        content = "Speaker " + turn.speaker + "says : " + turn.text
        agent.add_note(content, time=turn_datetime)
    return agent.memories


def save_rebuilt_memories(
    output_dir: Path,
    run_idx: int,
    condition: str,
    sample_idx: int,
    memories: Mapping[str, Any],
) -> None:
    cache_dir = output_dir / "rebuilt_caches" / f"run_{run_idx:02d}" / condition
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / f"memory_cache_sample_{sample_idx}.pkl").open("wb") as handle:
        pickle.dump(dict(memories), handle)


def evaluate_rebuilt_sample(
    run_idx: int,
    condition: str,
    sample_idx: int,
    sample: Any,
    query_plan: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_keyword_pruning_mode(condition)
    llm = RobustLLMController(
        backend=args.backend,
        model=args.model,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
    ).llm

    memories = rebuild_memories_for_sample(sample, args)
    save_rebuilt_memories(args.output_dir, run_idx, condition, sample_idx, memories)

    memory_items = list(memories.items())
    keyword_map = {
        note_id: list(getattr(note, "keywords", []) or [])
        for note_id, note in memory_items
    }
    documents = [
        ck.build_document(note, keyword_map[note_id])
        for note_id, note in memory_items
    ]
    retriever = ck.ContentKeywordRetriever(ck._EMBEDDING_MODEL, documents)

    sample_qas = eligible_qas(sample, args.qa_limit)
    result = {
        "total_questions": 0,
        "category_distribution": Counter(),
        "individual_results": [],
        "_all_metrics": [],
        "_all_categories": [],
    }

    for plan_item in query_plan:
        qa_idx = int(plan_item["qa_idx"])
        qa = sample_qas[qa_idx]
        query_keywords = str(plan_item["query_keywords"])
        indices = retriever.search(query_keywords, args.retrieve_k)
        raw_context = ck.render_context(memory_items, keyword_map, indices)
        prediction, user_prompt = ck.answer_question(
            llm,
            raw_context,
            qa.question,
            int(qa.category),
            qa.final_answer,
            args.temperature_c5,
            plan_item.get("answer_options"),
        )
        metrics = ck.calculate_f1_bleu1(prediction, qa.final_answer)
        result["total_questions"] += 1
        result["category_distribution"][str(qa.category)] += 1
        result["_all_metrics"].append(metrics)
        result["_all_categories"].append(int(qa.category))
        result["individual_results"].append(
            {
                "sample_id": sample_idx,
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
    return result


def init_condition_result(
    run_idx: int,
    condition: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "run": run_idx,
        "condition": condition,
        "experiment": "rebuild_memory_cache",
        "embedding_fields": ["content", "keywords"],
        "context_fields": ["timestamp", "content", "keywords"],
        "model": args.model,
        "backend": args.backend,
        "dataset": str(args.dataset),
        "retrieve_k": args.retrieve_k,
        "temperature_c5": args.temperature_c5,
        "total_questions": 0,
        "category_distribution": Counter(),
        "individual_results": [],
        "_all_metrics": [],
        "_all_categories": [],
    }


def merge_condition_samples(
    run_idx: int,
    condition: str,
    args: argparse.Namespace,
    sample_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    merged = init_condition_result(run_idx, condition, args)
    for sample_result in sample_results:
        merged["total_questions"] += sample_result["total_questions"]
        merged["category_distribution"].update(sample_result["category_distribution"])
        merged["individual_results"].extend(sample_result["individual_results"])
        merged["_all_metrics"].extend(sample_result["_all_metrics"])
        merged["_all_categories"].extend(sample_result["_all_categories"])

    merged["individual_results"].sort(key=lambda row: (row["sample_id"], row["qa_idx"]))
    merged["category_distribution"] = dict(merged["category_distribution"])
    merged["aggregate_metrics"] = ck.aggregate_metrics(
        merged.pop("_all_metrics"), merged.pop("_all_categories")
    )
    return merged


def evaluate_condition_run(
    run_idx: int,
    condition: str,
    samples: Sequence[Any],
    query_plan: Mapping[int, Sequence[Mapping[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_keyword_pruning_mode(condition)
    worker_count = min(args.max_workers, len(samples))
    logging.info("Run %s condition %s sample workers: %s", run_idx, condition, worker_count)
    sample_results_by_idx = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                evaluate_rebuilt_sample,
                run_idx,
                condition,
                sample_idx,
                sample,
                query_plan[sample_idx],
                args,
            ): sample_idx
            for sample_idx, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            sample_idx = futures[future]
            sample_results_by_idx[sample_idx] = future.result()
            logging.info(
                "Run %s condition %s sample %s completed (%s/%s)",
                run_idx,
                condition,
                sample_idx,
                len(sample_results_by_idx),
                len(futures),
            )

    sample_results = [
        sample_results_by_idx[sample_idx] for sample_idx in range(len(samples))
    ]
    return merge_condition_samples(run_idx, condition, args, sample_results)


def write_run_results(
    output_dir: Path,
    run_idx: int,
    results_by_condition: Mapping[str, Mapping[str, Any]],
) -> None:
    run_dir = output_dir / f"run_{run_idx:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for condition, result in results_by_condition.items():
        with (run_dir / f"{condition}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


def run_complete(output_dir: Path, run_idx: int) -> bool:
    run_dir = output_dir / f"run_{run_idx:02d}"
    return all((run_dir / f"{condition}.json").exists() for condition in CONDITIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild memories per pruning mode and evaluate content+keyword QA"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/content_keyword_rebuild/ollama_llama3.2-1b_k10_30runs"),
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--qa-limit", type=int, default=None)
    parser.add_argument("--turn-limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--sglang_host", default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset = repo_path(args.dataset)
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
    ck._EMBEDDING_MODEL = SentenceTransformer(args.embedding_model)

    for run_idx in range(args.runs):
        if args.resume and run_complete(args.output_dir, run_idx):
            logging.info("Skipping completed run %s", run_idx)
            continue

        random.seed(args.seed + run_idx)
        logging.info("Starting rebuild run %s/%s", run_idx + 1, args.runs)
        query_plan = build_query_plan(run_idx, samples, args)
        results_by_condition = {}
        for condition in CONDITIONS:
            results_by_condition[condition] = evaluate_condition_run(
                run_idx, condition, samples, query_plan, args
            )

        write_run_results(args.output_dir, run_idx, results_by_condition)
        ck.write_cross_run_summary(args.output_dir)
        logging.info("Finished rebuild run %s", run_idx)

    ck.write_cross_run_summary(args.output_dir)
    logging.info("Done. Results: %s", args.output_dir)


if __name__ == "__main__":
    main()
