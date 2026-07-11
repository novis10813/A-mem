"""Composable retrieval stages for robust QA."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from .reranking import BaseReranker, RetrievalCandidate


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


@dataclass(frozen=True)
class RetrievalRequest:
    similarity_query: str
    original_question: str | None
    final_k: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def query_for(self, selector: str) -> str:
        if selector == "similarity_query":
            return self.similarity_query
        if selector == "original_question":
            return self.original_question or self.similarity_query
        raise ValueError(f"Unsupported retrieval query selector: {selector}")


@dataclass(frozen=True)
class MemoryCandidate:
    memory_index: int
    memory_id: str
    text: str
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    source_stage: str = ""
    stage_trace: tuple[dict[str, Any], ...] = ()

    def __getitem__(self, index: int) -> int | str:
        if index == 0:
            return self.memory_index
        if index == 1:
            return self.text
        raise IndexError(index)

    def with_stage(
        self,
        *,
        stage_name: str,
        rank: int,
        score: float | None = None,
    ) -> "MemoryCandidate":
        scores = dict(self.scores)
        if score is not None:
            scores[stage_name] = float(score)
        ranks = dict(self.ranks)
        ranks[stage_name] = int(rank)
        trace = (*self.stage_trace, {"stage": stage_name, "rank": int(rank), "score": score})
        return replace(
            self,
            scores=scores,
            ranks=ranks,
            source_stage=stage_name,
            stage_trace=trace,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "memory_index": self.memory_index,
            "memory_id": self.memory_id,
            "text": self.text,
            "scores": self.scores,
            "ranks": self.ranks,
            "source_stage": self.source_stage,
            "stage_trace": list(self.stage_trace),
        }


class RetrievalStage(Protocol):
    name: str
    stage_type: str
    top_k: int
    query: str

    def run(
        self,
        request: RetrievalRequest,
        candidates: Sequence[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        """Return the next candidate list."""


MemoryTextFn = Callable[[Any], str]


@dataclass
class BaseStage:
    name: str
    top_k: int
    query: str
    stage_type: str

    def _query_text(self, request: RetrievalRequest) -> str:
        return request.query_for(self.query)


class EmbeddingCandidateGenerator(BaseStage):
    def __init__(
        self,
        *,
        name: str = "embedding_candidates",
        top_k: int,
        retriever: Any,
        memories: Sequence[Any],
        memory_text: MemoryTextFn,
        query: str = "similarity_query",
    ) -> None:
        super().__init__(name=name, top_k=top_k, query=query, stage_type="embedding")
        self.retriever = retriever
        self.memories = memories
        self.memory_text = memory_text

    def run(
        self,
        request: RetrievalRequest,
        candidates: Sequence[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        if candidates:
            raise ValueError(f"{self.name} must be the first retrieval stage")
        indices = [int(index) for index in self.retriever.search(self._query_text(request), self.top_k)]
        generated = []
        for rank, index in enumerate(indices, start=1):
            memory = self.memories[index]
            generated.append(
                MemoryCandidate(
                    memory_index=index,
                    memory_id=str(memory.id),
                    text=self.memory_text(memory),
                ).with_stage(stage_name=self.name, rank=rank)
            )
        return generated


class EmbeddingRerankerStage(BaseStage):
    def __init__(
        self,
        *,
        name: str = "embedding_rerank",
        top_k: int,
        retriever: Any,
        query: str = "similarity_query",
    ) -> None:
        super().__init__(name=name, top_k=top_k, query=query, stage_type="embedding_rerank")
        self.retriever = retriever

    def run(
        self,
        request: RetrievalRequest,
        candidates: Sequence[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        if not candidates or self.top_k < 1:
            return []
        if not hasattr(self.retriever, "model") or getattr(self.retriever, "embeddings", None) is None:
            ranked_indices = [int(index) for index in self.retriever.search(self._query_text(request), len(candidates))]
            rank_lookup = {memory_index: rank for rank, memory_index in enumerate(ranked_indices)}
            ranked = sorted(
                enumerate(candidates),
                key=lambda item: (rank_lookup.get(item[1].memory_index, len(ranked_indices)), item[0]),
            )[: self.top_k]
            return [
                candidate.with_stage(stage_name=self.name, rank=rank)
                for rank, (_, candidate) in enumerate(ranked, start=1)
            ]

        import numpy as np

        query_embedding = np.asarray(self.retriever.model.encode([self._query_text(request)])[0])
        embeddings = np.asarray(self.retriever.embeddings)
        scored: list[tuple[int, MemoryCandidate, float]] = []
        query_norm = float(np.linalg.norm(query_embedding)) or 1.0
        for original_rank, candidate in enumerate(candidates):
            memory_embedding = embeddings[candidate.memory_index]
            memory_norm = float(np.linalg.norm(memory_embedding)) or 1.0
            score = float(np.dot(query_embedding, memory_embedding) / (query_norm * memory_norm))
            scored.append((original_rank, candidate, score))
        scored.sort(key=lambda item: (-item[2], item[0]))
        return [
            candidate.with_stage(stage_name=self.name, rank=rank, score=score)
            for rank, (_, candidate, score) in enumerate(scored[: self.top_k], start=1)
        ]


class BM25CandidateGenerator(BaseStage):
    def __init__(
        self,
        *,
        name: str = "bm25_candidates",
        top_k: int,
        memories: Sequence[Any],
        memory_text: MemoryTextFn,
        document_text: MemoryTextFn,
        query: str = "similarity_query",
    ) -> None:
        super().__init__(name=name, top_k=top_k, query=query, stage_type="bm25")
        self.memories = memories
        self.memory_text = memory_text
        self.document_text = document_text

    def run(
        self,
        request: RetrievalRequest,
        candidates: Sequence[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        if candidates:
            raise ValueError(f"{self.name} must be the first retrieval stage")
        if not self.memories or self.top_k < 1:
            return []
        from rank_bm25 import BM25Okapi

        documents = [self.document_text(memory) for memory in self.memories]
        bm25 = BM25Okapi([_tokenize(document) for document in documents])
        scores = [float(score) for score in bm25.get_scores(_tokenize(self._query_text(request)))]
        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))[: self.top_k]
        return [
            MemoryCandidate(
                memory_index=index,
                memory_id=str(self.memories[index].id),
                text=self.memory_text(self.memories[index]),
            ).with_stage(stage_name=self.name, rank=rank, score=score)
            for rank, (index, score) in enumerate(ranked, start=1)
        ]


class BM25Reranker(BaseStage):
    def __init__(
        self,
        *,
        name: str = "bm25_rerank",
        top_k: int,
        query: str = "original_question",
    ) -> None:
        super().__init__(name=name, top_k=top_k, query=query, stage_type="bm25_rerank")

    def run(
        self,
        request: RetrievalRequest,
        candidates: Sequence[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        if not candidates or self.top_k < 1:
            return []
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi([_tokenize(candidate.text) for candidate in candidates])
        scores = [float(score) for score in bm25.get_scores(_tokenize(self._query_text(request)))]
        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))[: self.top_k]
        return [
            candidates[index].with_stage(stage_name=self.name, rank=rank, score=score)
            for rank, (index, score) in enumerate(ranked, start=1)
        ]


class CrossEncoderRerankerStage(BaseStage):
    def __init__(
        self,
        *,
        reranker: BaseReranker,
        name: str = "cross_encoder_rerank",
        top_k: int,
        query: str = "original_question",
    ) -> None:
        super().__init__(name=name, top_k=top_k, query=query, stage_type="cross_encoder")
        self.reranker = reranker

    def run(
        self,
        request: RetrievalRequest,
        candidates: Sequence[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        if not candidates or self.top_k < 1:
            return []
        rerank_inputs = [
            RetrievalCandidate(
                memory_index=candidate.memory_index,
                memory_id=candidate.memory_id,
                text=candidate.text,
                retrieval_score=None,
                rerank_score=None,
                source=candidate.source_stage,
                rank=candidate.ranks.get(candidate.source_stage, rank),
            )
            for rank, candidate in enumerate(candidates, start=1)
        ]
        reranked = self.reranker.rerank(self._query_text(request), rerank_inputs, self.top_k)
        by_index = {candidate.memory_index: candidate for candidate in candidates}
        return [
            by_index[item.index].with_stage(stage_name=self.name, rank=rank, score=item.score)
            for rank, item in enumerate(reranked, start=1)
            if item.index in by_index
        ]


class LimitStage(BaseStage):
    def __init__(self, *, name: str = "limit", top_k: int) -> None:
        super().__init__(name=name, top_k=top_k, query="similarity_query", stage_type="limit")

    def run(
        self,
        request: RetrievalRequest,
        candidates: Sequence[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        return [
            candidate.with_stage(stage_name=self.name, rank=rank)
            for rank, candidate in enumerate(list(candidates)[: self.top_k], start=1)
        ]


class RetrievalPipeline:
    def __init__(self, stages: Sequence[RetrievalStage], final_k: int) -> None:
        if final_k < 1:
            raise ValueError("final_k must be >= 1")
        if not stages:
            raise ValueError("retrieval pipeline requires at least one stage")
        self.stages = list(stages)
        self.final_k = final_k
        self.last_stage_info: list[dict[str, Any]] = []
        self.last_candidates: list[MemoryCandidate] = []

    def run(self, request: RetrievalRequest) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        self.last_stage_info = []
        self.last_candidates = []
        for stage in self.stages:
            input_count = len(candidates)
            output = stage.run(request, candidates)
            candidates = list(output)
            self.last_stage_info.append(
                {
                    "name": stage.name,
                    "type": stage.stage_type,
                    "query": stage.query,
                    "top_k": stage.top_k,
                    "input_count": input_count,
                    "output_count": len(candidates),
                }
            )
        self.last_candidates = list(candidates)
        return candidates[: request.final_k]
