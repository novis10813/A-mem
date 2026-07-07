"""Normalized result writing helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from .artifacts import write_json, write_qa_results_jsonl, write_usage_summary
from .schemas import QAResult, to_jsonable


def write_run_results(run_dir: Path, results: Sequence[QAResult]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_qa_results_jsonl(run_dir / "results.jsonl", results)
    write_json(
        run_dir / "results.json",
        {
            "total_questions": len(results),
            "individual_results": [to_jsonable(result) for result in results],
        },
    )
    usage_records = [record for result in results for record in result.usage]
    write_usage_summary(run_dir / "usage_summary.json", usage_records)


def flatten_usage_rows(results: Sequence[QAResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for record in result.usage:
            rows.append(
                {
                    "experiment_id": result.experiment_id,
                    "construction_run": result.construction_run,
                    "qa_run": result.qa_run,
                    "sample_id": result.sample_id,
                    "qa_idx": result.qa_idx,
                    "phase": record.phase,
                    "call_id": record.call_id,
                    "source": record.source,
                    "provider": record.provider,
                    "model": record.model,
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                    "total_tokens": record.total_tokens,
                    "estimated_tokens": record.estimated_tokens,
                    "latency_seconds": record.latency_seconds,
                    "cost_usd": record.cost_usd,
                }
            )
    return rows


def write_usage_csv(path: Path, results: Sequence[QAResult]) -> None:
    rows = flatten_usage_rows(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
