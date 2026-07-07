"""Convert A-Mem memory objects into normalized benchmark stores."""

from __future__ import annotations

from typing import Any, Mapping

from amem.benchmark.schemas import MemoryEdge, MemoryNode, MemoryRecord, MemoryStore


def memory_note_to_record(note: Any, sample_id: int) -> MemoryRecord:
    memory_id = str(getattr(note, "id"))
    content = str(getattr(note, "content", ""))
    context = str(getattr(note, "context", ""))
    keywords = tuple(str(item) for item in getattr(note, "keywords", []) or [])
    tags = tuple(str(item) for item in getattr(note, "tags", []) or [])
    links = tuple(str(item) for item in getattr(note, "links", []) or [])
    timestamp = getattr(note, "timestamp", None)
    text = (
        f"talk start time: {timestamp}\n"
        f"memory content: {content}\n"
        f"memory context: {context}\n"
        f"memory keywords: {', '.join(keywords)}\n"
        f"memory tags: {', '.join(tags)}"
    )
    return MemoryRecord(
        memory_id=memory_id,
        sample_id=sample_id,
        text=text,
        timestamp=None if timestamp is None else str(timestamp),
        content=content,
        summary=context,
        keywords=keywords,
        tags=tags,
        links=links,
        metadata={"source_method": "amem"},
    )


def memories_to_store(
    memories: Mapping[str, Any],
    sample_id: int,
    private_refs: Mapping[str, str] | None = None,
) -> MemoryStore:
    records = tuple(memory_note_to_record(note, sample_id) for note in memories.values())
    nodes = tuple(
        MemoryNode(
            node_id=record.memory_id,
            node_type="amem_note",
            text=record.text,
            label=record.content,
            timestamp=record.timestamp,
            properties={"keywords": list(record.keywords), "tags": list(record.tags)},
        )
        for record in records
    )
    edges = []
    for record in records:
        for target_id in record.links:
            edges.append(
                MemoryEdge(
                    edge_id=f"{record.memory_id}->{target_id}",
                    source_id=record.memory_id,
                    target_id=target_id,
                    edge_type="amem_link",
                )
            )
    return MemoryStore(
        sample_id=sample_id,
        records=records,
        nodes=nodes,
        edges=tuple(edges),
        private_refs=dict(private_refs or {}),
        metadata={"source_method": "amem"},
    )
