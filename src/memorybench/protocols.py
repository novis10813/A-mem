from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .schemas import DatasetBundle, DatasetSample, MemoryRecord, MemoryStore, QAResult, Question


class DatasetAdapter(Protocol):
    def load(self, path: str) -> DatasetBundle: ...


class ConstructionAdapter(Protocol):
    def build_sample(self, sample: DatasetSample) -> MemoryStore: ...


class Chunker(Protocol):
    def chunk(self, sample: DatasetSample) -> Sequence[MemoryRecord]: ...


class OneShotRetrievalAdapter(Protocol):
    def retrieve(self, question: Question, store: MemoryStore) -> Mapping[str, Any]: ...


class InteractiveRetrievalAdapter(Protocol):
    def bind(self, store: MemoryStore) -> Any: ...


class ContextAdapter(Protocol):
    def build(self, retrieval: Mapping[str, Any]) -> Mapping[str, Any]: ...


class QAAdapter(Protocol):
    def answer(self, question: Question, context: Mapping[str, Any]) -> str: ...


class MetricAdapter(Protocol):
    def compute(self, prediction: str, reference: str) -> Mapping[str, float]: ...


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> Any: ...
