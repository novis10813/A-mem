from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

from .config import ContextConfig, RetrievalConfig
from .schemas import DatasetSample, MemoryRecord, MemoryStore, Question


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w']+", text.lower()))


class TurnChunker:
    def chunk(self, sample: DatasetSample) -> Sequence[MemoryRecord]:
        return tuple(MemoryRecord(
            record_id=turn.turn_id, text=turn.text, content=turn.text,
            timestamp=turn.timestamp, speaker=turn.speaker, session_id=turn.session_id,
            evidence_refs=(turn.evidence_id,),
        ) for turn in sample.turns)


class TurnRAGConstruction:
    def __init__(self, chunker: TurnChunker) -> None:
        self.chunker = chunker

    def build_sample(self, sample: DatasetSample) -> MemoryStore:
        return MemoryStore(sample_id=sample.sample_id, records=tuple(self.chunker.chunk(sample)))


def staged_retrieve(question: Question, store: MemoryStore, config: RetrievalConfig) -> dict[str, Any]:
    candidates = list(store.records)
    traces = []
    query_text = question.text
    for stage in config.stages:
        started = time.perf_counter()
        if stage.adapter == "bm25":
            query = _tokens(query_text)
            ranked = sorted(
                candidates,
                key=lambda record: (-len(query & _tokens(record.text)), record.record_id),
            )[:stage.top_k]
            scores = [float(len(query & _tokens(record.text))) for record in ranked]
        elif stage.adapter == "limit":
            ranked, scores = candidates[:stage.top_k], [None] * min(stage.top_k, len(candidates))
        elif stage.adapter == "query_transform":
            mode = stage.params.get("mode", "lowercase")
            query_text = query_text.lower() if mode == "lowercase" else f"{stage.params.get('prefix', '')}{query_text}"
            ranked, scores = candidates, [None] * len(candidates)
        elif stage.adapter in {"embedding", "embedding_rerank"}:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer(stage.params.get("model", "all-MiniLM-L6-v2"))
            query_vector = model.encode([query_text], normalize_embeddings=True)[0]
            vectors = model.encode([record.text for record in candidates], normalize_embeddings=True)
            scored = sorted(zip(candidates, np.asarray(vectors) @ query_vector), key=lambda pair: (-float(pair[1]), pair[0].record_id))[:stage.top_k]
            ranked, scores = [pair[0] for pair in scored], [float(pair[1]) for pair in scored]
        elif stage.adapter == "cross_encoder":
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(stage.params.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
            values = model.predict([(query_text, record.text) for record in candidates])
            scored = sorted(zip(candidates, values), key=lambda pair: (-float(pair[1]), pair[0].record_id))[:stage.top_k]
            ranked, scores = [pair[0] for pair in scored], [float(pair[1]) for pair in scored]
        else:
            raise ValueError(f"Unsupported retrieval stage '{stage.adapter}'")
        traces.append({
            "adapter": stage.adapter, "query": query_text,
            "input_ranking": [r.record_id for r in candidates],
            "output_ranking": [r.record_id for r in ranked], "scores": scores,
            "timing_ms": (time.perf_counter() - started) * 1000,
            "config": stage.model_dump(mode="json"),
        })
        candidates = ranked
    return {"items": [record.model_dump(mode="json") for record in candidates], "stages": traces}


def build_context(retrieval: dict[str, Any], config: ContextConfig) -> dict[str, Any]:
    lines = []
    for item in retrieval["items"]:
        values = [str(item[field]) for field in config.fields if item.get(field) not in (None, "", [])]
        lines.append(" | ".join(values))
    return {"text": "\n".join(lines), "fields": list(config.fields), "item_ids": [i["record_id"] for i in retrieval["items"]]}


def answer(adapter: str, question: Question, context: dict[str, Any]) -> str:
    if adapter == "failing":
        raise RuntimeError("intentional QA failure")
    if adapter == "extractive":
        return context["text"].splitlines()[0] if context["text"] else ""
    raise ValueError(f"Unsupported QA adapter '{adapter}'")


def metrics(adapters: Sequence[Any], prediction: str, reference: str) -> dict[str, float]:
    result = {}
    for config in adapters:
        if config.adapter == "exact_match":
            result["exact_match"] = float(prediction.strip().casefold() == reference.strip().casefold())
        else:
            raise ValueError(f"Unsupported metric adapter '{config.adapter}'")
    return result
