"""Generic context builders over normalized memory stores."""

from __future__ import annotations

from typing import Any, Sequence

from .schemas import MemoryRecord, MemoryStore


def build_memory_fields_context(
    store: MemoryStore,
    item_ids: Sequence[str],
    fields: Sequence[str],
) -> dict[str, Any]:
    records_by_id = {record.memory_id: record for record in store.records}
    chunks = []
    used_records = []
    for item_id in item_ids:
        record = records_by_id[item_id]
        used_records.append(record.memory_id)
        chunks.append(_format_record(record, fields))
    return {"text": "\n".join(chunks), "record_ids": used_records, "fields": list(fields)}


def _format_record(record: MemoryRecord, fields: Sequence[str]) -> str:
    lines = []
    for field in fields:
        value = getattr(record, field)
        if isinstance(value, tuple):
            value = ", ".join(value)
        lines.append(f"{field}: {value}")
    return "\n".join(lines)
