#!/usr/bin/env python3
"""Build reusable A-MEM memory caches for two-stage experiments."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_common import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    build_manifest_payload,
    construction_cache_dir,
    construction_complete,
    expected_cache_files,
    experiment_cache_dir,
    repo_path,
    validate_experiment_id,
    write_json,
    write_manifest,
)
from experiment_config import build_args_from_config, load_experiment_config  # noqa: E402
from amem.benchmark.artifacts import write_jsonl, write_memory_store  # noqa: E402
from amem.benchmark.schemas import to_jsonable  # noqa: E402
from amem.llm_text_parsers import set_keyword_pruning_mode  # noqa: E402
from amem.load_dataset import load_locomo_dataset  # noqa: E402
from amem.memory_layer_robust import RobustAgenticMemorySystem  # noqa: E402
from amem.memory_pipeline import MemoryProcessingPipeline, PipelineTimingHook  # noqa: E402
from amem.methods.amem.serialization import memories_to_store  # noqa: E402


def select_samples(samples: Sequence[Any], ratio: float, sample_limit: int | None) -> list[Any]:
    selected = list(samples)
    if ratio < 1.0:
        selected = selected[: max(1, int(len(selected) * ratio))]
    if sample_limit is not None:
        selected = selected[:sample_limit]
    return selected


def conversation_turns(sample: Any, turn_limit: int | None) -> list[tuple[str, Any]]:
    turns = [
        (session.date_time, turn)
        for _, session in sample.conversation.sessions.items()
        for turn in session.turns
    ]
    return turns[:turn_limit] if turn_limit is not None else turns


def build_sample_cache(
    construction_run: int,
    sample_idx: int,
    sample: Any,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    timing_hook = PipelineTimingHook()
    agent = RobustAgenticMemorySystem(
        model_name=args.embedding_model,
        llm_backend=args.backend,
        llm_model=args.model,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        pipeline=MemoryProcessingPipeline(hooks=[timing_hook]),
    )

    turns = conversation_turns(sample, args.turn_limit)
    for turn_datetime, turn in turns:
        content = "Speaker " + turn.speaker + "says : " + turn.text
        agent.add_note(content, time=turn_datetime)

    memory_cache_file = output_dir / f"memory_cache_sample_{sample_idx}.pkl"
    retriever_cache_file = output_dir / f"retriever_cache_sample_{sample_idx}.pkl"
    retriever_embeddings_file = output_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy"

    with memory_cache_file.open("wb") as handle:
        pickle.dump(agent.memories, handle)
    agent.retriever.save(str(retriever_cache_file), str(retriever_embeddings_file))
    if not retriever_embeddings_file.exists():
        np.save(retriever_embeddings_file, np.empty((0, 0)))

    normalized_dir = output_dir / "normalized"
    private_refs = {
        "memory_cache": str(memory_cache_file.relative_to(output_dir)),
        "retriever_cache": str(retriever_cache_file.relative_to(output_dir)),
        "retriever_embeddings": str(retriever_embeddings_file.relative_to(output_dir)),
    }
    store = memories_to_store(agent.memories, sample_idx, private_refs=private_refs)
    write_memory_store(normalized_dir / f"memory_store_sample_{sample_idx}.json", store)
    write_jsonl(
        normalized_dir / f"memory_records_sample_{sample_idx}.jsonl",
        [to_jsonable(record) for record in store.records],
    )
    write_jsonl(
        normalized_dir / f"memory_nodes_sample_{sample_idx}.jsonl",
        [to_jsonable(node) for node in store.nodes],
    )
    write_jsonl(
        normalized_dir / f"memory_edges_sample_{sample_idx}.jsonl",
        [to_jsonable(edge) for edge in store.edges],
    )

    return {
        "construction_run": construction_run,
        "sample_idx": sample_idx,
        "turns": len(turns),
        "memories": len(agent.memories),
        "stage_timing": timing_hook.summary(),
    }


def build_construction_run(
    construction_run: int,
    samples: Sequence[Any],
    args: argparse.Namespace,
) -> None:
    output_dir = construction_cache_dir(args.cache_root, args.experiment_id, construction_run)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = list(range(len(samples)))

    if args.resume and construction_complete(output_dir, sample_indices):
        logging.info("Skipping completed construction_run_%02d", construction_run)
        return

    logging.info(
        "Starting construction_run_%02d with %s samples", construction_run, len(samples)
    )
    if args.max_workers == 1:
        sample_metadata = [
            build_sample_cache(construction_run, sample_idx, sample, output_dir, args)
            for sample_idx, sample in enumerate(samples)
        ]
    else:
        sample_metadata_by_idx: dict[int, dict[str, Any]] = {}
        worker_count = min(args.max_workers, len(samples))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    build_sample_cache,
                    construction_run,
                    sample_idx,
                    sample,
                    output_dir,
                    args,
                ): sample_idx
                for sample_idx, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                sample_idx = futures[future]
                sample_metadata_by_idx[sample_idx] = future.result()
                logging.info(
                    "construction_run_%02d sample %s completed (%s/%s)",
                    construction_run,
                    sample_idx,
                    len(sample_metadata_by_idx),
                    len(futures),
                )
        sample_metadata = [
            sample_metadata_by_idx[sample_idx] for sample_idx in range(len(samples))
        ]

    metadata = {
        "experiment_id": args.experiment_id,
        "construction_run": construction_run,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset),
        "backend": args.backend,
        "model": args.model,
        "keyword_pruning_mode": args.keyword_pruning_mode,
        "embedding_model": args.embedding_model,
        "ratio": args.ratio,
        "sample_limit": args.sample_limit,
        "turn_limit": args.turn_limit,
        "samples": len(samples),
        "sample_metadata": sample_metadata,
    }
    write_json(output_dir / "metadata.json", metadata)

    missing = [path for path in expected_cache_files(output_dir, sample_indices) if not path.exists()]
    if missing:
        raise RuntimeError(f"Construction run missing expected cache files: {missing}")
    normalized_missing = [
        output_dir / "normalized" / f"memory_store_sample_{sample_idx}.json"
        for sample_idx in sample_indices
        if not (output_dir / "normalized" / f"memory_store_sample_{sample_idx}.json").exists()
    ]
    if normalized_missing:
        raise RuntimeError(f"Construction run missing normalized memory stores: {normalized_missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable A-MEM memory caches")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--experiment-id", required=False)
    parser.add_argument("--dataset", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--construction-runs", type=int, default=1)
    parser.add_argument(
        "--keyword-pruning-mode", choices=["none", "simple", "nltk"], default="nltk"
    )
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--turn-limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--sglang_host", default="http://localhost")
    parser.add_argument("--sglang_port", type=int, default=30000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()
    if args.config:
        config_args = build_args_from_config(load_experiment_config(args.config))
        if args.resume:
            config_args.resume = True
        if args.log_level:
            config_args.log_level = args.log_level
        return config_args
    args.log_level = args.log_level or "INFO"
    return args


def main() -> None:
    args = parse_args()
    if not args.experiment_id:
        raise ValueError("--experiment-id is required unless --config is provided")
    args.experiment_id = validate_experiment_id(args.experiment_id)
    args.dataset = repo_path(args.dataset)
    args.cache_root = repo_path(args.cache_root)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if args.construction_runs < 0:
        raise ValueError("--construction-runs must be >= 0")
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("--ratio must be between 0.0 and 1.0")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")

    set_keyword_pruning_mode(args.keyword_pruning_mode)
    cache_dir = experiment_cache_dir(args.cache_root, args.experiment_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        cache_dir,
        build_manifest_payload(
            experiment_id=args.experiment_id,
            stage="memory_construction",
            dataset=args.dataset,
            created_at=datetime.now().isoformat(timespec="seconds"),
            config_source=getattr(args, "config_source", None),
            construction={
                "runs": args.construction_runs,
                "keyword_pruning_mode": args.keyword_pruning_mode,
                "embedding_model": args.embedding_model,
                "max_workers": args.max_workers,
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
                    "turn_limit": args.turn_limit,
                },
                "run": {
                    "resume": args.resume,
                    "log_level": args.log_level,
                },
            },
        ),
    )

    logging.info("Loading dataset: %s", args.dataset)
    samples = select_samples(load_locomo_dataset(args.dataset), args.ratio, args.sample_limit)
    for construction_run in range(args.construction_runs):
        build_construction_run(construction_run, samples, args)

    logging.info("Done. Caches: %s", cache_dir)


if __name__ == "__main__":
    main()
