from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from memorybench.schemas import MemoryEdge, MemoryLayer, MemoryNode, MemoryRecord, MemoryStore


def notes_to_store(sample_id: str, notes: Iterable[Any]) -> MemoryStore:
    notes = list(notes)
    records = tuple(MemoryRecord(
        record_id=str(note.id), text=str(note.content), content=str(note.content),
        timestamp=getattr(note, "timestamp", None), keywords=tuple(getattr(note, "keywords", ())),
        metadata={"context": getattr(note, "context", ""), "tags": list(getattr(note, "tags", ()))},
    ) for note in notes)
    nodes = tuple(MemoryNode(
        node_id=str(note.id), type="amem_note", text=str(note.content), layer="amem_notes",
        properties={"context": getattr(note, "context", ""), "keywords": list(getattr(note, "keywords", ()))},
    ) for note in notes)
    edges = tuple(MemoryEdge(
        edge_id=f"{note.id}->{target}", source_id=str(note.id), target_id=str(target), type="amem_link",
    ) for note in notes for target in getattr(note, "links", ()))
    return MemoryStore(
        sample_id=sample_id, records=records, nodes=nodes, edges=edges,
        layers=(MemoryLayer(name="amem_notes", node_ids=tuple(node.node_id for node in nodes), edge_ids=tuple(edge.edge_id for edge in edges)),),
    )
