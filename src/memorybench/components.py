from __future__ import annotations

import re
import time
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .config import ContextConfig, RetrievalConfig
from .schemas import DatasetSample, MemoryRecord, MemoryStore, Question


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w']+", text.lower()))


def _token_list(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.casefold())


def _bm25_scores(query_text: str, documents: Sequence[str]) -> list[float]:
    query = _token_list(query_text)
    tokenized = [_token_list(document) for document in documents]
    if not tokenized:
        return []
    document_count = len(tokenized)
    average_length = sum(len(document) for document in tokenized) / document_count or 1.0
    document_frequency = {
        term: sum(term in document for document in tokenized)
        for term in set(query)
    }
    k1, b = 1.5, 0.75
    scores = []
    for document in tokenized:
        frequencies = Counter(document)
        score = 0.0
        for term in query:
            frequency = frequencies[term]
            if not frequency:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * len(document) / average_length
            )
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return scores


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
    usage = []
    query_text = question.text
    for stage_index, stage in enumerate(config.stages):
        started = time.perf_counter()
        usage_start = len(usage)
        effective_query = question.text if stage.query == "original_question" else query_text
        if stage.adapter == "bm25":
            raw_scores = _bm25_scores(effective_query, [record.text for record in candidates])
            scored = sorted(
                zip(candidates, raw_scores),
                key=lambda pair: (-pair[1], pair[0].record_id),
            )[:stage.top_k]
            ranked = [record for record, _ in scored]
            scores = [score for _, score in scored]
        elif stage.adapter == "limit":
            ranked, scores = candidates[:stage.top_k], [None] * min(stage.top_k, len(candidates))
        elif stage.adapter == "query_transform":
            if stage.llm is not None:
                from .providers import complete

                template = stage.params.get(
                    "prompt",
                    "Generate concise retrieval keywords for this question:\n{question}",
                )
                response = complete(stage.llm, template.format(question=effective_query))
                query_text = response.text.strip()
                usage.append({
                    "phase": "retrieve_qa",
                    "component": f"retrieval:{stage.adapter}",
                    "model": stage.llm.model,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                    "source": response.usage_source,
                    "latency_ms": response.latency_ms,
                })
            else:
                mode = stage.params.get("mode", "lowercase")
                query_text = (
                    effective_query.lower()
                    if mode == "lowercase"
                    else f"{stage.params.get('prefix', '')}{effective_query}"
                )
            effective_query = query_text
            ranked, scores = candidates, [None] * len(candidates)
        elif stage.adapter in {"embedding", "embedding_rerank"}:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer(stage.params.get("model", "all-MiniLM-L6-v2"))
            query_vector = model.encode([effective_query], normalize_embeddings=True)[0]
            vectors = model.encode([record.text for record in candidates], normalize_embeddings=True)
            scored = sorted(zip(candidates, np.asarray(vectors) @ query_vector), key=lambda pair: (-float(pair[1]), pair[0].record_id))[:stage.top_k]
            ranked, scores = [pair[0] for pair in scored], [float(pair[1]) for pair in scored]
        elif stage.adapter == "cross_encoder":
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(stage.params.get("model", DEFAULT_CROSS_ENCODER_MODEL))
            values = model.predict([(effective_query, record.text) for record in candidates])
            scored = sorted(zip(candidates, values), key=lambda pair: (-float(pair[1]), pair[0].record_id))[:stage.top_k]
            ranked, scores = [pair[0] for pair in scored], [float(pair[1]) for pair in scored]
        else:
            raise ValueError(f"Unsupported retrieval stage '{stage.adapter}'")
        latency_ms = (time.perf_counter() - started) * 1000
        traces.append({
            "stage_index": stage_index,
            "adapter": stage.adapter, "query": effective_query,
            "input_ranking": [r.record_id for r in candidates],
            "output_ranking": [r.record_id for r in ranked], "scores": scores,
            "latency_ms": latency_ms,
            "timing_ms": latency_ms,
            "config": stage.model_dump(mode="json"),
            "usage": usage[usage_start:],
        })
        candidates = ranked
    return {
        "items": [record.model_dump(mode="json") for record in candidates],
        "stages": traces,
        "usage": usage,
    }


def build_context(retrieval: dict[str, Any], config: ContextConfig) -> dict[str, Any]:
    lines = []
    for item in retrieval["items"]:
        values = [str(item[field]) for field in config.fields if item.get(field) not in (None, "", [])]
        lines.append(" | ".join(values))
    return {"text": "\n".join(lines), "fields": list(config.fields), "item_ids": [i["record_id"] for i in retrieval["items"]]}


def answer(adapter: str, question: Question, context: dict[str, Any], params: dict[str, Any] | None = None) -> str:
    params = params or {}
    if adapter == "failing":
        selected = params.get("question_ids")
        if not selected or question.question_id in selected:
            raise RuntimeError("intentional QA failure")
        return context["text"].splitlines()[0] if context["text"] else ""
    if adapter == "extractive":
        return context["text"].splitlines()[0] if context["text"] else ""
    raise ValueError(f"Unsupported QA adapter '{adapter}'")


def metric_scores(names: Sequence[str], prediction: str, reference: str) -> dict[str, float]:
    prediction_tokens = _token_list(prediction)
    reference_tokens = _token_list(reference)
    result: dict[str, float] = {}
    for name in names:
        if name == "exact_match":
            result[name] = float(prediction.strip().casefold() == reference.strip().casefold())
        elif name == "f1":
            predicted, expected = set(prediction_tokens), set(reference_tokens)
            common = len(predicted & expected)
            precision = common / len(predicted) if predicted else 0.0
            recall = common / len(expected) if expected else 0.0
            result[name] = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
        elif name == "bleu1":
            if not prediction_tokens or not reference_tokens:
                result[name] = 0.0
                continue
            predicted_counts = Counter(prediction_tokens)
            reference_counts = Counter(reference_tokens)
            clipped = sum(
                min(count, reference_counts[token])
                for token, count in predicted_counts.items()
            )
            precision = clipped / len(prediction_tokens)
            brevity_penalty = (
                1.0 if len(prediction_tokens) >= len(reference_tokens)
                else math.exp(1.0 - len(reference_tokens) / len(prediction_tokens))
            )
            result[name] = brevity_penalty * precision
        else:
            raise ValueError(f"Unsupported metric adapter '{name}'")
    return result


def metrics(adapters: Sequence[Any], prediction: str, reference: str) -> dict[str, float]:
    return metric_scores(tuple(config.adapter for config in adapters), prediction, reference)
