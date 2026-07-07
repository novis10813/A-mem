"""Read and write normalized benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hooks import summarize_usage
from .schemas import MemoryStore, QAResult, UsageRecord, from_jsonable, to_jsonable


def write_memory_store(path: Path, store: MemoryStore) -> None:
    write_json(path, to_jsonable(store))


def read_memory_store(path: Path) -> MemoryStore:
    return from_jsonable(MemoryStore, read_json(path))


def write_qa_results_jsonl(path: Path, results: Sequence[QAResult]) -> None:
    write_jsonl(path, [to_jsonable(result) for result in results])


def write_usage_summary(path: Path, records: Sequence[UsageRecord]) -> None:
    write_json(path, summarize_usage(records))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
