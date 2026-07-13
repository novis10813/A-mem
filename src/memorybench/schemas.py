from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaxonomyDimension(Schema):
    name: str
    values: tuple[str, ...]
    canonical_mapping: dict[str, str] = Field(default_factory=dict)
    source: str | None = None


class DatasetTaxonomy(Schema):
    dimensions: tuple[TaxonomyDimension, ...] = ()


class Turn(Schema):
    turn_id: str
    evidence_id: str
    speaker: str
    text: str
    session_id: str
    timestamp: str | None = None


class Question(Schema):
    question_id: str
    text: str
    reference: str
    evidence_ids: tuple[str, ...] = ()
    labels: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class DatasetSample(Schema):
    sample_id: str
    turns: tuple[Turn, ...]
    questions: tuple[Question, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetBundle(Schema):
    dataset_id: str
    taxonomy: DatasetTaxonomy
    samples: tuple[DatasetSample, ...]


class MemoryRecord(Schema):
    record_id: str
    text: str
    content: str | None = None
    timestamp: str | None = None
    speaker: str | None = None
    session_id: str | None = None
    evidence_refs: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryNode(Schema):
    node_id: str
    type: str
    text: str | None = None
    layer: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class MemoryEdge(Schema):
    edge_id: str
    source_id: str
    target_id: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class MemoryLayer(Schema):
    name: str
    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()


class MemoryStore(Schema):
    sample_id: str
    records: tuple[MemoryRecord, ...] = ()
    nodes: tuple[MemoryNode, ...] = ()
    edges: tuple[MemoryEdge, ...] = ()
    layers: tuple[MemoryLayer, ...] = ()
    private_refs: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsageRecord(Schema):
    phase: str
    component: str
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    source: str = "reported"
    latency_ms: float | None = None


class QAResult(Schema):
    question_id: str
    sample_id: str
    status: str
    question: str
    reference: str
    prediction: str = ""
    labels: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    tool_traces: tuple[dict[str, Any], ...] = ()
    context: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    usage: tuple[UsageRecord, ...] = ()
    errors: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)
