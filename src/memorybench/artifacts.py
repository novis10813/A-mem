from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Any

from .schemas import MemoryStore


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def artifact_key(value: str) -> str:
    return value.replace(":", "__").replace("/", "_")


def write_memory_store(directory: Path, store: MemoryStore) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "private").mkdir(exist_ok=True)
    atomic_jsonl(directory / "records.jsonl", (item.model_dump(mode="json") for item in store.records))
    atomic_jsonl(directory / "nodes.jsonl", (item.model_dump(mode="json") for item in store.nodes))
    atomic_jsonl(directory / "edges.jsonl", (item.model_dump(mode="json") for item in store.edges))
    atomic_jsonl(directory / "layers.jsonl", (item.model_dump(mode="json") for item in store.layers))
    atomic_json(directory / "store.json", {
        "schema_version": "memorybench/memory-store/v1",
        "sample_id": store.sample_id,
        "records": "records.jsonl",
        "nodes": "nodes.jsonl",
        "edges": "edges.jsonl",
        "layers": "layers.jsonl",
        "private_refs": store.private_refs,
        "metadata": store.metadata,
    })


def read_memory_store(directory: Path) -> MemoryStore:
    header_path = directory / "store.json"
    if not header_path.exists():
        raise FileNotFoundError(f"Memory store header not found: {header_path}")
    header = json.loads(header_path.read_text(encoding="utf-8"))
    return MemoryStore.model_validate({
        "sample_id": header["sample_id"],
        "records": read_jsonl(directory / header["records"]),
        "nodes": read_jsonl(directory / header["nodes"]),
        "edges": read_jsonl(directory / header["edges"]),
        "layers": read_jsonl(directory / header["layers"]),
        "private_refs": header.get("private_refs", {}),
        "metadata": header.get("metadata", {}),
    })
