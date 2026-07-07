"""Stable schemas for component benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T")


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    sample_id: int
    text: str
    timestamp: str | None = None
    content: str | None = None
    summary: str | None = None
    keywords: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryNode:
    node_id: str
    node_type: str
    text: str | None = None
    label: str | None = None
    timestamp: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    text: str | None = None
    valid_at: str | None = None
    invalid_at: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryLayer:
    name: str
    node_ids: tuple[str, ...] = ()
    edge_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryStore:
    sample_id: int
    records: tuple[MemoryRecord, ...] = ()
    nodes: tuple[MemoryNode, ...] = ()
    edges: tuple[MemoryEdge, ...] = ()
    layers: tuple[MemoryLayer, ...] = ()
    indices: Mapping[str, Any] = field(default_factory=dict)
    private_refs: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedItem:
    item_id: str
    rank: int
    text: str
    item_type: str = "record"
    score: float | None = None
    source_stage: str = ""
    trace: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalToolCall:
    tool_name: str
    arguments: Mapping[str, Any]
    output_items: tuple[RetrievedItem, ...] = ()
    output_text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UsageRecord:
    phase: str
    call_id: str
    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_tokens: int | None = None
    cost_usd: float | None = None
    latency_seconds: float | None = None
    source: str = "reported"
    tokenizer: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QAResult:
    experiment_id: str
    construction_run: int
    qa_run: int
    sample_id: int
    qa_idx: int
    question: str
    reference: str
    prediction: str
    category: int | None
    metrics: Mapping[str, Any]
    retrieval: Mapping[str, Any]
    context: Mapping[str, Any]
    prompt: str | None
    usage: tuple[UsageRecord, ...] = ()
    errors: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def from_jsonable(cls: type[T], payload: Mapping[str, Any]) -> T:
    kwargs = {}
    type_hints = get_type_hints(cls)
    for item in fields(cls):
        if item.name not in payload:
            continue
        kwargs[item.name] = _coerce_value(type_hints[item.name], payload[item.name])
    return cls(**kwargs)


def _coerce_value(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple and args:
        inner = args[0]
        return tuple(_coerce_value(inner, item) for item in value)
    if is_dataclass(annotation) and isinstance(value, Mapping):
        return from_jsonable(annotation, value)
    return value
