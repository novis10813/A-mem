#!/usr/bin/env python3
"""Evaluate QA against reusable A-MEM memory caches."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_common import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_RESULTS_ROOT,
    build_manifest_payload,
    construction_cache_dir,
    content_keywords_complete,
    experiment_results_dir,
    experiment_cache_dir,
    load_manifest,
    qa_mode_dir,
    qa_run_dir,
    repo_path,
    robust_complete,
    summarize_values,
    validate_experiment_id,
    write_manifest,
)
from experiment_config import evaluate_args_from_config, load_experiment_config  # noqa: E402
from amem.benchmark.results import write_run_results  # noqa: E402
from amem.load_dataset import load_locomo_dataset  # noqa: E402
from amem.memory_layer_robust import (  # noqa: E402
    RobustLLMController,
    build_bm25_retriever_from_memories,
    robust_retrieval_document,
)
from amem.reranking import DEFAULT_CROSS_ENCODER_MODEL, build_reranker  # noqa: E402
from amem.retrieval_pipeline import (  # noqa: E402
    BM25CandidateGenerator,
    BM25Reranker,
    CrossEncoderRerankerStage,
    EmbeddingCandidateGenerator,
    EmbeddingRerankerStage,
    LimitStage,
    RetrievalPipeline,
)
from amem.methods.amem.qa import robust_dict_to_qa_results  # noqa: E402


METRICS = ("f1", "bleu1")


def load_content_keyword_module():
    import run_content_keyword_pruning_experiment as ck

    return ck


def load_robust_agent_class():
    from test_advanced_robust import RobustAdvancedMemAgent

    return RobustAdvancedMemAgent


def merge_robust_sample_outputs(sample_outputs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from test_advanced_robust import merge_sample_outputs

    return merge_sample_outputs(sample_outputs)


def select_samples(samples: Sequence[Any], ratio: float, sample_limit: int | None) -> list[Any]:
    selected = list(samples)
    if ratio < 1.0:
        selected = selected[: max(1, int(len(selected) * ratio))]
    if sample_limit is not None:
        selected = selected[:sample_limit]
    return selected


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(part.strip() for part in value.split(",") if part.strip())
    if not conditions:
        raise ValueError("--keyword-conditions must contain at least one condition")
    unsupported = sorted(set(conditions) - {"none", "nltk"})
    if unsupported:
        raise ValueError(f"Unsupported keyword conditions: {unsupported}")
    return conditions


def discover_construction_runs(cache_root: Path, experiment_id: str) -> list[int]:
    experiment_dir = cache_root / experiment_id
    runs = []
    for path in sorted(experiment_dir.glob("construction_run_*")):
        if not path.is_dir():
            continue
        try:
            runs.append(int(path.name.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return runs


def evaluate_content_keywords_run(
    construction_run: int,
    qa_run: int,
    samples: Sequence[Any],
    sample_states: Sequence[Any],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    ck = load_content_keyword_module()
    random.seed(args.seed + construction_run * 100_000 + qa_run)
    llm = None
    if args.max_workers == 1:
        llm = RobustLLMController(
            backend=args.backend,
            model=args.model,
            sglang_host=args.sglang_host,
            sglang_port=args.sglang_port,
        ).llm
    results = ck.evaluate_run(qa_run, samples, sample_states, llm, args)
    for condition, result in results.items():
        result["construction_run"] = construction_run
        result["qa_run"] = qa_run
        result["source_construction_cache_dir"] = str(
            construction_cache_dir(args.cache_root, args.cache_experiment_id, construction_run)
        )
    return results


def write_content_keywords_run(
    construction_run: int,
    qa_run: int,
    results_by_condition: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    run_dir = qa_run_dir(
        args.results_root,
        args.experiment_id,
        construction_run,
        "content_keywords",
        qa_run,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for condition, result in results_by_condition.items():
        with (run_dir / f"{condition}.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


def load_robust_agent_from_cache(
    cache_dir: Path,
    sample_idx: int,
    args: argparse.Namespace,
) -> Any:
    RobustAdvancedMemAgent = load_robust_agent_class()
    agent = RobustAdvancedMemAgent(
        args.model,
        args.backend,
        args.retrieve_k,
        args.temperature_c5,
        args.sglang_host,
        args.sglang_port,
    )
    memory_cache_file = cache_dir / f"memory_cache_sample_{sample_idx}.pkl"
    retriever_cache_file = cache_dir / f"retriever_cache_sample_{sample_idx}.pkl"
    retriever_embeddings_file = cache_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy"
    if not memory_cache_file.exists():
        raise FileNotFoundError(f"Missing memory cache file: {memory_cache_file}")
    with memory_cache_file.open("rb") as handle:
        agent.memory_system.memories = pickle.load(handle)
    if getattr(args, "retrieval_mode", "embedding") == "bm25":
        agent.memory_system.retriever = build_bm25_retriever_from_memories(
            agent.memory_system.memories
        )
    elif retriever_cache_file.exists() and retriever_embeddings_file.exists():
        agent.memory_system.retriever = agent.memory_system.retriever.load(
            str(retriever_cache_file), str(retriever_embeddings_file)
        )
    else:
        agent.memory_system.retriever = agent.memory_system.retriever.load_from_local_memory(
            agent.memory_system.memories, args.embedding_model
        )
    if getattr(args, "retrieval_pipeline", None) is not None:
        agent.memory_system.retrieval_pipeline = build_retrieval_pipeline(
            args.retrieval_pipeline,
            agent.memory_system,
        )
    return agent


def build_retrieval_pipeline(config: Any, memory_system: Any) -> RetrievalPipeline:
    memories = list(memory_system.memories.values())
    stages = []
    for stage in config.stages:
        if stage.type == "embedding":
            stages.append(
                EmbeddingCandidateGenerator(
                    name=stage.name,
                    top_k=stage.top_k,
                    query=stage.query,
                    retriever=memory_system.retriever,
                    memories=memories,
                    memory_text=memory_system._memory_rerank_text,
                )
            )
        elif stage.type == "bm25":
            stages.append(
                BM25CandidateGenerator(
                    name=stage.name,
                    top_k=stage.top_k,
                    query=stage.query,
                    memories=memories,
                    memory_text=memory_system._memory_rerank_text,
                    document_text=robust_retrieval_document,
                )
            )
        elif stage.type == "embedding_rerank":
            stages.append(
                EmbeddingRerankerStage(
                    name=stage.name,
                    top_k=stage.top_k,
                    query=stage.query,
                    retriever=memory_system.retriever,
                )
            )
        elif stage.type == "bm25_rerank":
            stages.append(BM25Reranker(name=stage.name, top_k=stage.top_k, query=stage.query))
        elif stage.type == "cross_encoder":
            reranker = build_reranker(
                "cross_encoder",
                stage.model or DEFAULT_CROSS_ENCODER_MODEL,
                stage.batch_size,
            )
            stages.append(
                CrossEncoderRerankerStage(
                    name=stage.name,
                    top_k=stage.top_k,
                    query=stage.query,
                    reranker=reranker,
                )
            )
        elif stage.type == "limit":
            stages.append(LimitStage(name=stage.name, top_k=stage.top_k))
        else:
            raise ValueError(f"Unsupported retrieval pipeline stage: {stage.type}")
    return RetrievalPipeline(stages=stages, final_k=config.final_k)


def evaluate_robust_sample(
    construction_run: int,
    qa_run: int,
    sample_idx: int,
    sample: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from amem.utils import calculate_metrics
    from amem.llm_text_parsers import parse_plain_text_answer

    random.seed(args.seed + construction_run * 100_000 + qa_run * 1000 + sample_idx)
    cache_dir = construction_cache_dir(args.cache_root, args.cache_experiment_id, construction_run)
    agent = load_robust_agent_from_cache(cache_dir, sample_idx, args)
    eligible_qas = [qa for qa in sample.qa if int(qa.category) in [1, 2, 3, 4, 5]]
    if args.qa_limit is not None:
        eligible_qas = eligible_qas[: args.qa_limit]

    results = []
    all_metrics = []
    all_categories = []
    category_counts = Counter()
    for qa_idx, qa in enumerate(eligible_qas):
        prediction, user_prompt, raw_context = agent.answer_question(
            qa.question, int(qa.category), qa.final_answer
        )
        prediction = parse_plain_text_answer(prediction)
        metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {}
        all_metrics.append(metrics)
        all_categories.append(int(qa.category))
        category_counts[str(qa.category)] += 1
        results.append(
            {
                "sample_id": sample_idx,
                "qa_idx": qa_idx,
                "question": qa.question,
                "query_keywords": agent.last_retrieval_info.get("query_keywords", ""),
                "prediction": prediction,
                "reference": qa.final_answer,
                "category": int(qa.category),
                "metrics": metrics,
                "retrieval_info": agent.last_retrieval_info,
                "raw_context": raw_context,
                "user_prompt": user_prompt,
            }
        )

    return {
        "sample_idx": sample_idx,
        "results": results,
        "metrics": all_metrics,
        "categories": all_categories,
        "category_counts": dict(category_counts),
        "error_num": 0,
        "stage_timing": {},
    }


def evaluate_robust_run(
    construction_run: int,
    qa_run: int,
    samples: Sequence[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.max_workers == 1:
        sample_outputs = [
            evaluate_robust_sample(construction_run, qa_run, sample_idx, sample, args)
            for sample_idx, sample in enumerate(samples)
        ]
    else:
        sample_outputs = []
        worker_count = min(args.max_workers, len(samples))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    evaluate_robust_sample,
                    construction_run,
                    qa_run,
                    sample_idx,
                    sample,
                    args,
                ): sample_idx
                for sample_idx, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                sample_outputs.append(future.result())

    from amem.utils import aggregate_metrics

    merged = merge_robust_sample_outputs(sample_outputs)
    return {
        "construction_run": construction_run,
        "qa_run": qa_run,
        "model": args.model,
        "dataset": str(args.dataset),
        "memory_layer": "robust",
        "source_construction_cache_dir": str(
            construction_cache_dir(args.cache_root, args.cache_experiment_id, construction_run)
        ),
        "backend": args.backend,
        "retrieval_pipeline": retrieval_pipeline_to_json(getattr(args, "retrieval_pipeline", None)),
        "temperature_c5": args.temperature_c5,
        "total_questions": merged["total_questions"],
        "category_distribution": dict(merged["category_counts"]),
        "aggregate_metrics": aggregate_metrics(merged["metrics"], merged["categories"]),
        "individual_results": merged["results"],
    }


def write_robust_run(
    construction_run: int,
    qa_run: int,
    result: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    run_dir = qa_run_dir(args.results_root, args.experiment_id, construction_run, "robust", qa_run)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    normalized_results = robust_dict_to_qa_results(result, args.experiment_id)
    normalized_dir = run_dir / "normalized"
    write_run_results(normalized_dir, normalized_results)


def retrieval_pipeline_to_json(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "final_k": config.final_k,
        "stages": [
            {
                "type": stage.type,
                "name": stage.name,
                "top_k": stage.top_k,
                "query": stage.query,
                **({"model": stage.model} if stage.model else {}),
                **(
                    {"batch_size": stage.batch_size}
                    if stage.type == "cross_encoder"
                    else {}
                ),
            }
            for stage in config.stages
        ],
    }


def flatten_metric_rows(
    construction_run: int,
    qa_run: int,
    condition: str,
    aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for split, metrics in aggregate.items():
        for metric, stats in metrics.items():
            if not isinstance(stats, Mapping):
                continue
            row = {
                "construction_run": construction_run,
                "qa_run": qa_run,
                "condition": condition,
                "split": split,
                "metric": metric,
            }
            row.update(stats)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_mode_summary(mode_dir: Path, mode: str, conditions: Sequence[str]) -> None:
    per_run_rows = []
    for run_dir in sorted(mode_dir.glob("qa_run_*")):
        if not run_dir.is_dir():
            continue
        try:
            qa_run = int(run_dir.name.rsplit("_", 1)[1])
            construction_run = int(mode_dir.parent.name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue

        if mode == "content_keywords":
            for condition in conditions:
                path = run_dir / f"{condition}.json"
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    per_run_rows.extend(
                        flatten_metric_rows(
                            construction_run,
                            qa_run,
                            condition,
                            payload.get("aggregate_metrics", {}),
                        )
                    )
        else:
            path = run_dir / "results.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                per_run_rows.extend(
                    flatten_metric_rows(
                        construction_run,
                        qa_run,
                        "robust",
                        payload.get("aggregate_metrics", {}),
                    )
                )

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in per_run_rows:
        if "mean" in row:
            grouped[(row["condition"], row["split"], row["metric"])].append(float(row["mean"]))

    summary_rows = []
    for (condition, split, metric), values in sorted(grouped.items()):
        stats = summarize_values(values)
        summary_rows.append(
            {
                "condition": condition,
                "split": split,
                "metric": metric,
                "runs": len(values),
                "mean_across_runs": stats["mean"],
                "std_across_runs": stats["std"],
                "median_across_runs": stats["median"],
                "min_across_runs": stats["min"],
                "max_across_runs": stats["max"],
            }
        )

    write_csv(mode_dir / "per_run_metrics.csv", per_run_rows)
    write_csv(mode_dir / "summary_across_runs.csv", summary_rows)
    with (mode_dir / "summary_across_runs.json").open("w", encoding="utf-8") as handle:
        json.dump({"per_run": per_run_rows, "summary": summary_rows}, handle, indent=2)


def write_experiment_summary(results_root: Path, experiment_id: str) -> None:
    experiment_dir = experiment_results_dir(results_root, experiment_id)
    rows = []
    for summary_path in sorted(experiment_dir.glob("construction_run_*/*/per_run_metrics.csv")):
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    write_csv(experiment_dir / "per_run_metrics.csv", rows)


def evaluate_content_keywords(
    construction_run: int,
    samples: Sequence[Any],
    args: argparse.Namespace,
) -> None:
    ck = load_content_keyword_module()
    from sentence_transformers import SentenceTransformer

    mode_dir = qa_mode_dir(
        args.results_root, args.experiment_id, construction_run, "content_keywords"
    )
    cache_dir = construction_cache_dir(args.cache_root, args.cache_experiment_id, construction_run)
    args.memory_cache_dir = cache_dir
    ck.CONDITIONS = args.keyword_conditions
    ck._EMBEDDING_MODEL = SentenceTransformer(args.embedding_model)
    sample_states = ck.prepare_sample_states(
        samples, cache_dir, ck._EMBEDDING_MODEL, args.max_keywords
    )
    for qa_run in range(args.qa_runs):
        run_dir = qa_run_dir(
            args.results_root,
            args.experiment_id,
            construction_run,
            "content_keywords",
            qa_run,
        )
        if args.resume and content_keywords_complete(run_dir, args.keyword_conditions):
            logging.info(
                "Skipping construction_run_%02d content_keywords qa_run_%02d",
                construction_run,
                qa_run,
            )
            continue
        results = evaluate_content_keywords_run(
            construction_run, qa_run, samples, sample_states, args
        )
        write_content_keywords_run(construction_run, qa_run, results, args)
        write_mode_summary(mode_dir, "content_keywords", args.keyword_conditions)
    write_mode_summary(mode_dir, "content_keywords", args.keyword_conditions)


def evaluate_robust(
    construction_run: int,
    samples: Sequence[Any],
    args: argparse.Namespace,
) -> None:
    mode_dir = qa_mode_dir(args.results_root, args.experiment_id, construction_run, "robust")
    for qa_run in range(args.qa_runs):
        run_dir = qa_run_dir(
            args.results_root, args.experiment_id, construction_run, "robust", qa_run
        )
        if args.resume and robust_complete(run_dir):
            logging.info(
                "Skipping construction_run_%02d robust qa_run_%02d",
                construction_run,
                qa_run,
            )
            continue
        result = evaluate_robust_run(construction_run, qa_run, samples, args)
        write_robust_run(construction_run, qa_run, result, args)
        write_mode_summary(mode_dir, "robust", ("robust",))
    write_mode_summary(mode_dir, "robust", ("robust",))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QA from saved A-MEM caches")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--experiment-id", required=False)
    parser.add_argument("--cache-experiment-id", default=None)
    parser.add_argument("--dataset", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument(
        "--qa-mode", choices=["content_keywords", "robust", "both"], default="content_keywords"
    )
    parser.add_argument("--keyword-conditions", default="none,nltk")
    parser.add_argument("--qa-runs", type=int, default=1)
    parser.add_argument("--construction-runs", type=int, default=None)
    parser.add_argument("--retrieve-k", "--retrieve_k", dest="retrieve_k", type=int, default=10)
    parser.add_argument("--temperature-c5", "--temperature_c5", dest="temperature_c5", type=float, default=0.5)
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
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()
    if args.config:
        config_args = evaluate_args_from_config(load_experiment_config(args.config))
        if args.resume:
            config_args.resume = True
        if args.log_level:
            config_args.log_level = args.log_level
        return config_args
    args.log_level = args.log_level or "INFO"
    args.retrieval_pipeline = None
    args.retrieval_mode = "embedding"
    return args


def main() -> None:
    args = parse_args()
    if not args.experiment_id:
        raise ValueError("--experiment-id is required unless --config is provided")
    args.experiment_id = validate_experiment_id(args.experiment_id)
    args.cache_experiment_id = validate_experiment_id(args.cache_experiment_id or args.experiment_id)
    args.dataset = repo_path(args.dataset)
    args.cache_root = repo_path(args.cache_root)
    args.results_root = repo_path(args.results_root)
    if isinstance(args.keyword_conditions, str):
        args.keyword_conditions = parse_conditions(args.keyword_conditions)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if args.qa_runs < 0:
        raise ValueError("--qa-runs must be >= 0")
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("--ratio must be between 0.0 and 1.0")
    if args.retrieve_k < 1:
        raise ValueError("--retrieve-k must be >= 1")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    results_dir = experiment_results_dir(args.results_root, args.experiment_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    source_cache_manifest = load_manifest(
        experiment_cache_dir(args.cache_root, args.cache_experiment_id)
    )
    write_manifest(
        results_dir,
        build_manifest_payload(
            experiment_id=args.experiment_id,
            cache_experiment_id=args.cache_experiment_id,
            stage="qa_evaluation",
            dataset=args.dataset,
            created_at=datetime.now().isoformat(timespec="seconds"),
            config_source=getattr(args, "config_source", None),
            construction={
                "runs": args.construction_runs,
                "embedding_model": args.embedding_model,
            },
            evaluation={
                "qa_mode": args.qa_mode,
                "qa_runs": args.qa_runs,
                "keyword_conditions": list(args.keyword_conditions),
                "retrieval_pipeline": retrieval_pipeline_to_json(
                    getattr(args, "retrieval_pipeline", None)
                ),
                "temperature_c5": args.temperature_c5,
                "max_keywords": args.max_keywords,
                "seed": args.seed,
            },
            runtime={
                "backend": {
                    "name": args.backend,
                    "model": args.model,
                    "sglang_host": args.sglang_host,
                    "sglang_port": args.sglang_port,
                },
                "limits": {
                    "ratio": args.ratio,
                    "sample_limit": args.sample_limit,
                    "qa_limit": args.qa_limit,
                },
                "run": {
                    "resume": args.resume,
                    "log_level": args.log_level,
                    "max_workers": args.max_workers,
                },
            },
            source_cache_manifest=source_cache_manifest,
        ),
    )

    construction_runs = (
        list(range(args.construction_runs))
        if args.construction_runs is not None
        else discover_construction_runs(args.cache_root, args.cache_experiment_id)
    )
    if not construction_runs:
        raise FileNotFoundError(
            f"No construction_run_* directories found under "
            f"{args.cache_root / args.cache_experiment_id}"
        )

    logging.info("Loading dataset: %s", args.dataset)
    samples = select_samples(load_locomo_dataset(args.dataset), args.ratio, args.sample_limit)
    modes = ["content_keywords", "robust"] if args.qa_mode == "both" else [args.qa_mode]
    for construction_run in construction_runs:
        for mode in modes:
            logging.info("Evaluating construction_run_%02d mode=%s", construction_run, mode)
            if mode == "content_keywords":
                evaluate_content_keywords(construction_run, samples, args)
            else:
                evaluate_robust(construction_run, samples, args)
    write_experiment_summary(args.results_root, args.experiment_id)
    logging.info("Done. Results: %s", results_dir)


if __name__ == "__main__":
    main()
