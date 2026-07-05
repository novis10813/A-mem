"""Reranking helpers for two-stage memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@dataclass(frozen=True)
class RerankedCandidate:
    index: int
    score: float


class BaseReranker(Protocol):
    mode: str

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[int, str]],
        top_k: int,
    ) -> list[RerankedCandidate]:
        """Rank candidate memory texts for the query."""


class CrossEncoderReranker:
    mode = "cross_encoder"

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[int, str]],
        top_k: int,
    ) -> list[RerankedCandidate]:
        if top_k < 1 or not candidates:
            return []

        pairs = [(query, text) for _, text in candidates]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scores = [float(score) for score in raw_scores]
        ranked = sorted(
            enumerate(zip(candidates, scores)),
            key=lambda item: (-item[1][1], item[0]),
        )
        return [
            RerankedCandidate(index=int(candidate[0]), score=score)
            for _, (candidate, score) in ranked[:top_k]
        ]


def build_reranker(
    mode: str,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    batch_size: int = 32,
) -> BaseReranker | None:
    if mode == "off":
        return None
    if mode == "cross_encoder":
        return CrossEncoderReranker(model_name=model_name, batch_size=batch_size)
    raise ValueError(f"Unsupported rerank mode: {mode}")
