from __future__ import annotations

from typing import Any

from memorybench.components import staged_retrieve
from memorybench.config import RetrievalConfig
from memorybench.schemas import MemoryStore, Question


def retrieve_amem(
    question: Question,
    store: MemoryStore,
    config: RetrievalConfig,
) -> dict[str, Any]:
    result = staged_retrieve(question, store, config)
    records = {record.record_id: record for record in store.records}
    item_ids = [item["record_id"] for item in result["items"]]
    expanded = list(result["items"])
    expansion_trace = []
    for source_id in tuple(item_ids):
        source = records[source_id]
        for neighbor_id in source.metadata.get("links", ()):
            if neighbor_id not in records or neighbor_id in item_ids:
                continue
            expanded.append(records[neighbor_id].model_dump(mode="json"))
            item_ids.append(neighbor_id)
            expansion_trace.append({"source_id": source_id, "neighbor_id": neighbor_id})
    return {**result, "items": expanded, "neighbor_expansion": expansion_trace}
