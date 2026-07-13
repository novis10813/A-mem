from __future__ import annotations

from typing import Any

from memorybench.config import ContextConfig


def build_amem_context(
    retrieval: dict[str, Any],
    config: ContextConfig,
) -> dict[str, Any]:
    chunks = []
    item_ids = []
    for item in retrieval.get("items", ()):
        metadata = item.get("metadata", {})
        chunks.append(
            "talk start time:" + str(item.get("timestamp") or "") + "\n"
            "memory content: " + str(item.get("content") or item.get("text") or "") + "\n"
            "memory context: " + str(metadata.get("context", "")) + "\n"
            "memory keywords: " + ", ".join(item.get("keywords", ())) + "\n"
            "memory tags: " + ", ".join(metadata.get("tags", ()))
        )
        item_ids.append(item["record_id"])
    return {"text": "\n".join(chunks), "item_ids": item_ids, "adapter": config.adapter}
