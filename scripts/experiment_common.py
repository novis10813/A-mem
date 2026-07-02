"""Shared helpers for two-stage A-MEM experiments."""

from __future__ import annotations

import json
import re
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = Path("artifacts")
DEFAULT_CACHE_ROOT = ARTIFACTS_ROOT / "caches"
DEFAULT_RESULTS_ROOT = ARTIFACTS_ROOT / "results"
DEFAULT_LOG_ROOT = ARTIFACTS_ROOT / "logs"

_EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_experiment_id(experiment_id: str) -> str:
    if not experiment_id or not experiment_id.strip():
        raise ValueError("--experiment-id must not be empty")
    if experiment_id in {".", ".."} or "/" in experiment_id or "\\" in experiment_id:
        raise ValueError("--experiment-id must be a single safe path component")
    if not _EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError(
            "--experiment-id may contain only letters, digits, '.', '_', ':', and '-'"
        )
    return experiment_id


def construction_run_name(run_idx: int) -> str:
    return f"construction_run_{run_idx:02d}"


def qa_run_name(run_idx: int) -> str:
    return f"qa_run_{run_idx:02d}"


def experiment_cache_dir(cache_root: Path | str, experiment_id: str) -> Path:
    return repo_path(cache_root) / validate_experiment_id(experiment_id)


def construction_cache_dir(
    cache_root: Path | str, experiment_id: str, construction_run: int
) -> Path:
    return experiment_cache_dir(cache_root, experiment_id) / construction_run_name(
        construction_run
    )


def experiment_results_dir(results_root: Path | str, experiment_id: str) -> Path:
    return repo_path(results_root) / validate_experiment_id(experiment_id)


def qa_mode_dir(
    results_root: Path | str,
    experiment_id: str,
    construction_run: int,
    qa_mode: str,
) -> Path:
    return (
        experiment_results_dir(results_root, experiment_id)
        / construction_run_name(construction_run)
        / qa_mode
    )


def qa_run_dir(
    results_root: Path | str,
    experiment_id: str,
    construction_run: int,
    qa_mode: str,
    qa_run: int,
) -> Path:
    return qa_mode_dir(results_root, experiment_id, construction_run, qa_mode) / qa_run_name(
        qa_run
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_manifest(root: Path, payload: Mapping[str, Any]) -> None:
    write_json(root / "manifest.json", dict(payload))


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    return read_json(path) if path.exists() else {}


def expected_cache_files(cache_dir: Path, sample_indices: Sequence[int]) -> list[Path]:
    files: list[Path] = [cache_dir / "metadata.json"]
    for sample_idx in sample_indices:
        files.extend(
            [
                cache_dir / f"memory_cache_sample_{sample_idx}.pkl",
                cache_dir / f"retriever_cache_sample_{sample_idx}.pkl",
                cache_dir / f"retriever_cache_embeddings_sample_{sample_idx}.npy",
            ]
        )
    return files


def construction_complete(cache_dir: Path, sample_indices: Sequence[int]) -> bool:
    return all(path.exists() for path in expected_cache_files(cache_dir, sample_indices))


def content_keywords_complete(run_dir: Path, conditions: Sequence[str]) -> bool:
    return all((run_dir / f"{condition}.json").exists() for condition in conditions)


def robust_complete(run_dir: Path) -> bool:
    return (run_dir / "results.json").exists()


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
                    payload = read_json(path)
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
                payload = read_json(path)
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
    write_json(
        mode_dir / "summary_across_runs.json",
        {"per_run": per_run_rows, "summary": summary_rows},
    )
