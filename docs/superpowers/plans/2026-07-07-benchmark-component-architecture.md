# Benchmark Component Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working slice of the component benchmark architecture: normalized memory stores, component registries, A-Mem export compatibility, one-shot retrieval/context/QA plumbing, token usage accounting, and dashboard-readable normalized results.

**Architecture:** Keep the current two-stage experiment flow, but add a new `amem.benchmark` layer that owns schemas, artifact IO, hooks, config normalization, and result writing. Existing A-Mem internals remain in place and are wrapped by adapters, so old CLI paths continue to work while new normalized artifacts become available for mixed construction/retrieval/QA experiments.

**Tech Stack:** Python `>=3.13,<3.14`, `uv`, `pytest`, dataclasses, JSON/JSONL artifacts, existing `rank-bm25`, `sentence-transformers`, Gradio dashboard optional dependency.

## Global Constraints

- Use `uv run python -m pytest ...` for tests.
- Generated experiment outputs must stay under `artifacts/` and must not be committed.
- Keep root-level `test_advanced*.py` files as compatibility evaluation entrypoints.
- Do not rewrite `src/amem/memory_layer_robust.py` internals before adapters exist.
- Preserve current two-stage commands during the transition.
- Token usage accounting is a standard hook, not a construction/retrieval/QA adapter.
- Graph methods must not be forced to flatten into RAG except as an explicit baseline.
- This first plan creates graph-capable schemas and toolset interfaces; it does not implement full MRAgent or Zep/Graphiti methods.

---

## Scope Check

The full design covers several future subsystems. This plan implements the minimum useful foundation and A-Mem compatibility path:

- normalized schemas and IO;
- hook and token usage infrastructure;
- config/registry boundaries;
- A-Mem normalized export;
- record-based retrieval/context adapters;
- QA result writing;
- dashboard loader support.

Full external paper adapters, persistent graph databases, and production tool-calling agents should be separate follow-up plans once this foundation is merged.

## File Structure

- Create `src/amem/benchmark/__init__.py`: public exports for benchmark primitives.
- Create `src/amem/benchmark/schemas.py`: dataclasses and JSON conversion helpers.
- Create `src/amem/benchmark/hooks.py`: hook protocol, timing hook, token usage hook, usage summaries.
- Create `src/amem/benchmark/artifacts.py`: JSON/JSONL read/write helpers for `MemoryStore` and `QAResult`.
- Create `src/amem/benchmark/registry.py`: small adapter registry with clear errors.
- Create `src/amem/benchmark/config.py`: new component config dataclasses plus translation from current config.
- Create `src/amem/benchmark/context.py`: generic context builders over `MemoryStore`.
- Create `src/amem/benchmark/results.py`: normalized result writer and summary aggregation.
- Create `src/amem/methods/__init__.py`: method package marker.
- Create `src/amem/methods/amem/__init__.py`: A-Mem adapter exports.
- Create `src/amem/methods/amem/serialization.py`: convert robust memory notes to `MemoryStore`.
- Create `src/amem/methods/amem/construction.py`: A-Mem construction adapter wrapper.
- Create `src/amem/methods/amem/qa.py`: robust QA output normalization helpers.
- Modify `src/amem/retrieval_pipeline.py`: add constructors/helpers that can consume `MemoryStore.records` without breaking existing object-based callers.
- Modify `scripts/build_memories.py`: write normalized memory stores alongside existing pickle caches.
- Modify `scripts/evaluate_memories.py`: write normalized QA results and usage summaries alongside existing results.
- Modify `scripts/experiment_data_loader.py`: discover normalized results when present, with fallback to current robust path.
- Create tests:
  - `tests/test_benchmark_schemas.py`
  - `tests/test_benchmark_hooks.py`
  - `tests/test_benchmark_artifacts.py`
  - `tests/test_benchmark_config.py`
  - `tests/test_amem_serialization.py`
  - `tests/test_benchmark_results.py`

---

### Task 1: Benchmark Schemas and JSON Round Trips

**Files:**
- Create: `src/amem/benchmark/__init__.py`
- Create: `src/amem/benchmark/schemas.py`
- Test: `tests/test_benchmark_schemas.py`

**Interfaces:**
- Produces: `MemoryRecord`, `MemoryNode`, `MemoryEdge`, `MemoryLayer`, `MemoryStore`, `RetrievedItem`, `RetrievalToolCall`, `UsageRecord`, `QAResult`.
- Produces: `to_jsonable(value: Any) -> Any`, `from_jsonable(cls: type[T], payload: Mapping[str, Any]) -> T`.
- Consumed by: all later tasks.

- [ ] **Step 1: Write failing schema round-trip tests**

Add `tests/test_benchmark_schemas.py`:

```python
from amem.benchmark.schemas import (
    MemoryEdge,
    MemoryLayer,
    MemoryNode,
    MemoryRecord,
    MemoryStore,
    QAResult,
    RetrievedItem,
    RetrievalToolCall,
    UsageRecord,
    from_jsonable,
    to_jsonable,
)


def test_memory_store_round_trip_preserves_graph_and_records():
    store = MemoryStore(
        sample_id=7,
        records=(
            MemoryRecord(
                memory_id="m1",
                sample_id=7,
                text="Alice visited Taipei.",
                timestamp="2026-01-01T10:00:00",
                content="Alice visited Taipei.",
                keywords=("alice", "taipei"),
                links=("m2",),
            ),
        ),
        nodes=(MemoryNode(node_id="n1", node_type="entity", label="Alice"),),
        edges=(
            MemoryEdge(
                edge_id="e1",
                source_id="n1",
                target_id="n2",
                edge_type="visited",
                valid_at="2026-01-01T10:00:00",
            ),
        ),
        layers=(MemoryLayer(name="semantic_entity", node_ids=("n1", "n2"), edge_ids=("e1",)),),
        private_refs={"pickle": "private/memory_cache_sample_7.pkl"},
    )

    payload = to_jsonable(store)
    restored = from_jsonable(MemoryStore, payload)

    assert restored == store
    assert payload["records"][0]["keywords"] == ["alice", "taipei"]
    assert payload["layers"][0]["name"] == "semantic_entity"


def test_qa_result_round_trip_preserves_usage_and_tool_calls():
    result = QAResult(
        experiment_id="exp",
        construction_run=0,
        qa_run=1,
        sample_id=2,
        qa_idx=3,
        question="Where did Alice go?",
        reference="Taipei",
        prediction="Alice went to Taipei.",
        category=4,
        metrics={"f1": 0.8},
        retrieval={
            "items": [
                to_jsonable(RetrievedItem(item_id="m1", rank=1, text="Alice visited Taipei."))
            ],
            "tool_calls": [
                to_jsonable(
                    RetrievalToolCall(
                        tool_name="query_event_keywords",
                        arguments={"keywords": ["taipei"]},
                        output_text="m1",
                    )
                )
            ],
        },
        context={"text": "Alice visited Taipei."},
        prompt="Answer the question.",
        usage=(UsageRecord(phase="qa", call_id="answer", total_tokens=42),),
    )

    restored = from_jsonable(QAResult, to_jsonable(result))

    assert restored == result
    assert restored.usage[0].source == "reported"
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `uv run python -m pytest tests/test_benchmark_schemas.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'amem.benchmark'`.

- [ ] **Step 3: Add schema implementation**

Create `src/amem/benchmark/__init__.py`:

```python
"""Benchmark component primitives for memory experiment comparisons."""

from .schemas import (
    MemoryEdge,
    MemoryLayer,
    MemoryNode,
    MemoryRecord,
    MemoryStore,
    QAResult,
    RetrievedItem,
    RetrievalToolCall,
    UsageRecord,
    from_jsonable,
    to_jsonable,
)

__all__ = [
    "MemoryEdge",
    "MemoryLayer",
    "MemoryNode",
    "MemoryRecord",
    "MemoryStore",
    "QAResult",
    "RetrievedItem",
    "RetrievalToolCall",
    "UsageRecord",
    "from_jsonable",
    "to_jsonable",
]
```

Create `src/amem/benchmark/schemas.py`:

```python
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
```

- [ ] **Step 4: Run focused schema tests**

Run: `uv run python -m pytest tests/test_benchmark_schemas.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/amem/benchmark/__init__.py src/amem/benchmark/schemas.py tests/test_benchmark_schemas.py
git commit -m "feat: add benchmark artifact schemas"
```

---

### Task 2: Hook Protocol and Token Usage Accounting

**Files:**
- Create: `src/amem/benchmark/hooks.py`
- Test: `tests/test_benchmark_hooks.py`

**Interfaces:**
- Consumes: `UsageRecord`.
- Produces: `Hook`, `HookContext`, `NoOpHook`, `TokenUsageHook`, `summarize_usage(records: Sequence[UsageRecord]) -> dict[str, Any]`.
- Consumed by: construction/evaluation runners and result writer.

- [ ] **Step 1: Write failing hook tests**

Add `tests/test_benchmark_hooks.py`:

```python
from amem.benchmark.hooks import HookContext, TokenUsageHook, summarize_usage
from amem.benchmark.schemas import UsageRecord


def test_token_usage_hook_records_reported_usage():
    hook = TokenUsageHook()
    hook.after_llm_call(
        HookContext(phase="qa", sample_id=1, qa_idx=2),
        call_id="answer",
        provider="openai",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        latency_seconds=0.25,
    )

    assert hook.records == (
        UsageRecord(
            phase="qa",
            call_id="answer",
            provider="openai",
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_seconds=0.25,
            source="reported",
            metadata={"sample_id": 1, "qa_idx": 2},
        ),
    )


def test_token_usage_hook_estimates_when_usage_missing():
    hook = TokenUsageHook(estimate_when_missing=True, tokenizer="words")
    hook.after_llm_call(
        HookContext(phase="qa", sample_id=1, qa_idx=2),
        call_id="answer",
        provider="ollama",
        model="llama3.2:1b",
        prompt="one two three",
        completion="four five",
    )

    record = hook.records[0]
    assert record.estimated_tokens == 5
    assert record.source == "estimated"
    assert record.tokenizer == "words"


def test_summarize_usage_keeps_reported_and_estimated_separate():
    summary = summarize_usage(
        [
            UsageRecord(phase="qa", call_id="a", total_tokens=10, source="reported"),
            UsageRecord(phase="qa", call_id="b", estimated_tokens=7, source="estimated"),
        ]
    )

    assert summary["by_source"]["reported"]["total_tokens"] == 10
    assert summary["by_source"]["estimated"]["estimated_tokens"] == 7
    assert summary["calls"] == 2
```

- [ ] **Step 2: Run focused hook tests and verify failure**

Run: `uv run python -m pytest tests/test_benchmark_hooks.py -v`

Expected: FAIL with `ModuleNotFoundError` or missing `amem.benchmark.hooks`.

- [ ] **Step 3: Add hook implementation**

Create `src/amem/benchmark/hooks.py`:

```python
"""Hook interfaces and standard observability hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schemas import UsageRecord


@dataclass(frozen=True)
class HookContext:
    phase: str
    sample_id: int | None = None
    qa_idx: int | None = None
    construction_run: int | None = None
    qa_run: int | None = None
    metadata: Mapping[str, Any] | None = None

    def as_metadata(self) -> dict[str, Any]:
        data = dict(self.metadata or {})
        for key in ("sample_id", "qa_idx", "construction_run", "qa_run"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


class Hook:
    def before_llm_call(self, context: HookContext, **kwargs: Any) -> None:
        pass

    def after_llm_call(self, context: HookContext, **kwargs: Any) -> None:
        pass


class NoOpHook(Hook):
    pass


class TokenUsageHook(Hook):
    def __init__(self, *, estimate_when_missing: bool = False, tokenizer: str = "words") -> None:
        self.estimate_when_missing = estimate_when_missing
        self.tokenizer = tokenizer
        self._records: list[UsageRecord] = []

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        return tuple(self._records)

    def after_llm_call(
        self,
        context: HookContext,
        *,
        call_id: str,
        provider: str | None = None,
        model: str | None = None,
        usage: Mapping[str, Any] | None = None,
        prompt: str | None = None,
        completion: str | None = None,
        latency_seconds: float | None = None,
        cost_usd: float | None = None,
    ) -> None:
        if usage:
            record = UsageRecord(
                phase=context.phase,
                call_id=call_id,
                provider=provider,
                model=model,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                total_tokens=_int_or_none(usage.get("total_tokens")),
                cost_usd=cost_usd,
                latency_seconds=latency_seconds,
                source="reported",
                metadata=context.as_metadata(),
            )
        elif self.estimate_when_missing:
            record = UsageRecord(
                phase=context.phase,
                call_id=call_id,
                provider=provider,
                model=model,
                estimated_tokens=_estimate_tokens(prompt, completion),
                cost_usd=cost_usd,
                latency_seconds=latency_seconds,
                source="estimated",
                tokenizer=self.tokenizer,
                metadata=context.as_metadata(),
            )
        else:
            return
        self._records.append(record)


def summarize_usage(records: Sequence[UsageRecord]) -> dict[str, Any]:
    summary: dict[str, Any] = {"calls": len(records), "by_source": {}}
    for record in records:
        bucket = summary["by_source"].setdefault(
            record.source,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_tokens": 0,
                "latency_seconds": 0.0,
                "cost_usd": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += record.prompt_tokens or 0
        bucket["completion_tokens"] += record.completion_tokens or 0
        bucket["total_tokens"] += record.total_tokens or 0
        bucket["estimated_tokens"] += record.estimated_tokens or 0
        bucket["latency_seconds"] += record.latency_seconds or 0.0
        bucket["cost_usd"] += record.cost_usd or 0.0
    return summary


def _estimate_tokens(prompt: str | None, completion: str | None) -> int:
    return len(((prompt or "") + " " + (completion or "")).split())


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)
```

- [ ] **Step 4: Run focused hook tests**

Run: `uv run python -m pytest tests/test_benchmark_hooks.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/amem/benchmark/hooks.py tests/test_benchmark_hooks.py
git commit -m "feat: add benchmark token usage hook"
```

---

### Task 3: Normalized Artifact IO

**Files:**
- Create: `src/amem/benchmark/artifacts.py`
- Test: `tests/test_benchmark_artifacts.py`

**Interfaces:**
- Consumes: `MemoryStore`, `QAResult`, `UsageRecord`, `to_jsonable`, `from_jsonable`.
- Produces:
  - `write_memory_store(path: Path, store: MemoryStore) -> None`
  - `read_memory_store(path: Path) -> MemoryStore`
  - `write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None`
  - `read_jsonl(path: Path) -> list[dict[str, Any]]`
  - `write_qa_results_jsonl(path: Path, results: Sequence[QAResult]) -> None`
  - `write_usage_summary(path: Path, records: Sequence[UsageRecord]) -> None`

- [ ] **Step 1: Write failing artifact tests**

Add `tests/test_benchmark_artifacts.py`:

```python
import json
from pathlib import Path

from amem.benchmark.artifacts import (
    read_jsonl,
    read_memory_store,
    write_jsonl,
    write_memory_store,
    write_qa_results_jsonl,
    write_usage_summary,
)
from amem.benchmark.schemas import MemoryRecord, MemoryStore, QAResult, UsageRecord


def test_memory_store_json_round_trip(tmp_path: Path):
    path = tmp_path / "memory_store_sample_0.json"
    store = MemoryStore(
        sample_id=0,
        records=(MemoryRecord(memory_id="m0", sample_id=0, text="hello"),),
    )

    write_memory_store(path, store)

    assert read_memory_store(path) == store
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["records"][0]["memory_id"] == "m0"


def test_jsonl_helpers_create_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(path, [{"a": 1}, {"b": 2}])

    assert read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_qa_results_jsonl_and_usage_summary(tmp_path: Path):
    result = QAResult(
        experiment_id="exp",
        construction_run=0,
        qa_run=0,
        sample_id=0,
        qa_idx=0,
        question="q",
        reference="r",
        prediction="p",
        category=1,
        metrics={"f1": 0.5},
        retrieval={"items": []},
        context={"text": ""},
        prompt=None,
        usage=(UsageRecord(phase="qa", call_id="answer", total_tokens=9),),
    )

    write_qa_results_jsonl(tmp_path / "results.jsonl", [result])
    write_usage_summary(tmp_path / "usage_summary.json", result.usage)

    assert read_jsonl(tmp_path / "results.jsonl")[0]["usage"][0]["total_tokens"] == 9
    summary = json.loads((tmp_path / "usage_summary.json").read_text(encoding="utf-8"))
    assert summary["by_source"]["reported"]["total_tokens"] == 9
```

- [ ] **Step 2: Run focused artifact tests and verify failure**

Run: `uv run python -m pytest tests/test_benchmark_artifacts.py -v`

Expected: FAIL with missing `amem.benchmark.artifacts`.

- [ ] **Step 3: Add artifact implementation**

Create `src/amem/benchmark/artifacts.py`:

```python
"""Read and write normalized benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hooks import summarize_usage
from .schemas import MemoryStore, QAResult, UsageRecord, from_jsonable, to_jsonable


def write_memory_store(path: Path, store: MemoryStore) -> None:
    write_json(path, to_jsonable(store))


def read_memory_store(path: Path) -> MemoryStore:
    return from_jsonable(MemoryStore, read_json(path))


def write_qa_results_jsonl(path: Path, results: Sequence[QAResult]) -> None:
    write_jsonl(path, [to_jsonable(result) for result in results])


def write_usage_summary(path: Path, records: Sequence[UsageRecord]) -> None:
    write_json(path, summarize_usage(records))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
```

- [ ] **Step 4: Run artifact tests**

Run: `uv run python -m pytest tests/test_benchmark_artifacts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/amem/benchmark/artifacts.py tests/test_benchmark_artifacts.py
git commit -m "feat: add normalized benchmark artifact IO"
```

---

### Task 4: Component Config and Registry

**Files:**
- Create: `src/amem/benchmark/registry.py`
- Create: `src/amem/benchmark/config.py`
- Test: `tests/test_benchmark_config.py`

**Interfaces:**
- Produces: `ComponentConfig`, `BenchmarkConfig`, `RunConfig`, `translate_legacy_config(config: Any) -> BenchmarkConfig`.
- Produces: `AdapterRegistry.register(kind: str, name: str, adapter: Any) -> None`, `AdapterRegistry.get(kind: str, name: str) -> Any`.
- Consumed by: runner integration tasks.

- [ ] **Step 1: Write failing config and registry tests**

Add `tests/test_benchmark_config.py`:

```python
from types import SimpleNamespace

import pytest

from amem.benchmark.config import BenchmarkConfig, translate_legacy_config
from amem.benchmark.registry import AdapterRegistry


def test_registry_returns_registered_adapter_and_rejects_unknown():
    registry = AdapterRegistry()
    adapter = object()
    registry.register("construction", "amem", adapter)

    assert registry.get("construction", "amem") is adapter
    with pytest.raises(KeyError, match="Unknown retrieval adapter: missing"):
        registry.get("retrieval", "missing")


def test_translate_legacy_config_preserves_existing_experiment_shape():
    legacy = SimpleNamespace(
        experiment_id="exp",
        dataset="data/locomo10.json",
        construction=SimpleNamespace(
            runs=1,
            keyword_pruning_mode="nltk",
            embedding_model="all-MiniLM-L6-v2",
        ),
        evaluation=SimpleNamespace(
            qa_mode="robust",
            qa_runs=2,
            retrieval_pipeline=SimpleNamespace(
                final_k=10,
                stages=(SimpleNamespace(type="embedding", name="embedding_candidates", top_k=10),),
            ),
        ),
        backend=SimpleNamespace(name="ollama", model="llama3.2:1b"),
        run=SimpleNamespace(resume=True),
    )

    config = translate_legacy_config(legacy)

    assert isinstance(config, BenchmarkConfig)
    assert config.construction.adapter == "amem"
    assert config.retrieval.adapter == "pipeline"
    assert config.qa.adapter == "robust_plain_text"
    assert config.run.hooks[0]["type"] == "token_usage"
```

- [ ] **Step 2: Run focused config tests and verify failure**

Run: `uv run python -m pytest tests/test_benchmark_config.py -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: Add registry implementation**

Create `src/amem/benchmark/registry.py`:

```python
"""Adapter registry for benchmark components."""

from __future__ import annotations

from typing import Any


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, dict[str, Any]] = {}

    def register(self, kind: str, name: str, adapter: Any) -> None:
        self._adapters.setdefault(kind, {})[name] = adapter

    def get(self, kind: str, name: str) -> Any:
        try:
            return self._adapters[kind][name]
        except KeyError as exc:
            raise KeyError(f"Unknown {kind} adapter: {name}") from exc
```

- [ ] **Step 4: Add config implementation**

Create `src/amem/benchmark/config.py`:

```python
"""Component benchmark config objects and legacy translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ComponentConfig:
    adapter: str
    params: Mapping[str, Any] = field(default_factory=dict)
    hooks: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class RetrievalConfig(ComponentConfig):
    stages: tuple[Mapping[str, Any], ...] = ()
    tools: tuple[str, ...] = ()
    view: str | None = None


@dataclass(frozen=True)
class QAConfig(ComponentConfig):
    runs: int = 1
    backend: str = "ollama"
    model: str = "llama3.2:1b"


@dataclass(frozen=True)
class RunConfig:
    resume: bool = False
    hooks: tuple[Mapping[str, Any], ...] = field(
        default_factory=lambda: (
            {"type": "token_usage", "mode": "reported_or_estimated", "estimate_when_missing": True},
        )
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    experiment_id: str
    dataset: str
    construction: ComponentConfig
    retrieval: RetrievalConfig
    context: ComponentConfig
    qa: QAConfig
    metrics: tuple[str, ...] = ("f1", "bleu1")
    run: RunConfig = field(default_factory=RunConfig)


def translate_legacy_config(config: Any) -> BenchmarkConfig:
    stages = tuple(
        {
            "type": stage.type,
            "name": getattr(stage, "name", stage.type),
            "top_k": stage.top_k,
            **({"query": stage.query} if hasattr(stage, "query") else {}),
            **({"model": stage.model} if getattr(stage, "model", None) else {}),
            **({"batch_size": stage.batch_size} if hasattr(stage, "batch_size") else {}),
        }
        for stage in config.evaluation.retrieval_pipeline.stages
    )
    return BenchmarkConfig(
        experiment_id=config.experiment_id,
        dataset=str(config.dataset),
        construction=ComponentConfig(
            adapter="amem",
            params={
                "runs": config.construction.runs,
                "keyword_pruning_mode": config.construction.keyword_pruning_mode,
                "embedding_model": config.construction.embedding_model,
            },
        ),
        retrieval=RetrievalConfig(adapter="pipeline", stages=stages),
        context=ComponentConfig(adapter="amem_full"),
        qa=QAConfig(
            adapter="robust_plain_text",
            runs=config.evaluation.qa_runs,
            backend=config.backend.name,
            model=config.backend.model,
        ),
        run=RunConfig(resume=bool(config.run.resume)),
    )
```

- [ ] **Step 5: Run focused config tests**

Run: `uv run python -m pytest tests/test_benchmark_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/amem/benchmark/registry.py src/amem/benchmark/config.py tests/test_benchmark_config.py
git commit -m "feat: add benchmark component config and registry"
```

---

### Task 5: A-Mem MemoryStore Serialization

**Files:**
- Create: `src/amem/methods/__init__.py`
- Create: `src/amem/methods/amem/__init__.py`
- Create: `src/amem/methods/amem/serialization.py`
- Test: `tests/test_amem_serialization.py`

**Interfaces:**
- Consumes: robust memory note-like objects with `id`, `content`, `context`, `keywords`, `tags`, `links`, `timestamp`.
- Produces: `memory_note_to_record(note: Any, sample_id: int) -> MemoryRecord`, `memories_to_store(memories: Mapping[str, Any], sample_id: int, private_refs: Mapping[str, str] | None = None) -> MemoryStore`.
- Consumed by: build script normalized export.

- [ ] **Step 1: Write failing A-Mem serialization tests**

Add `tests/test_amem_serialization.py`:

```python
from types import SimpleNamespace

from amem.methods.amem.serialization import memories_to_store, memory_note_to_record


def test_memory_note_to_record_preserves_core_fields():
    note = SimpleNamespace(
        id="note-1",
        content="Speaker Alice says: I moved to Taipei.",
        context="Alice talked about moving.",
        keywords=["Alice", "Taipei"],
        tags=["move"],
        links=["note-2"],
        timestamp="2026-01-01T10:00:00",
    )

    record = memory_note_to_record(note, sample_id=3)

    assert record.memory_id == "note-1"
    assert record.sample_id == 3
    assert record.content == note.content
    assert record.summary == note.context
    assert record.keywords == ("Alice", "Taipei")
    assert record.links == ("note-2",)
    assert "memory content:" in record.text


def test_memories_to_store_adds_note_nodes_and_link_edges():
    memories = {
        "note-1": SimpleNamespace(
            id="note-1",
            content="A",
            context="ctx",
            keywords=[],
            tags=[],
            links=["note-2"],
            timestamp=None,
        ),
        "note-2": SimpleNamespace(
            id="note-2",
            content="B",
            context="ctx",
            keywords=[],
            tags=[],
            links=[],
            timestamp=None,
        ),
    }

    store = memories_to_store(memories, sample_id=4, private_refs={"pickle": "private/cache.pkl"})

    assert [record.memory_id for record in store.records] == ["note-1", "note-2"]
    assert {node.node_id for node in store.nodes} == {"note-1", "note-2"}
    assert store.edges[0].source_id == "note-1"
    assert store.edges[0].target_id == "note-2"
    assert store.private_refs["pickle"] == "private/cache.pkl"
```

- [ ] **Step 2: Run focused serialization tests and verify failure**

Run: `uv run python -m pytest tests/test_amem_serialization.py -v`

Expected: FAIL with missing `amem.methods`.

- [ ] **Step 3: Add method package markers**

Create `src/amem/methods/__init__.py`:

```python
"""Method adapters for benchmark components."""
```

Create `src/amem/methods/amem/__init__.py`:

```python
"""A-Mem benchmark adapters."""

from .serialization import memories_to_store, memory_note_to_record

__all__ = ["memories_to_store", "memory_note_to_record"]
```

- [ ] **Step 4: Add serialization implementation**

Create `src/amem/methods/amem/serialization.py`:

```python
"""Convert A-Mem memory objects into normalized benchmark stores."""

from __future__ import annotations

from typing import Any, Mapping

from amem.benchmark.schemas import MemoryEdge, MemoryNode, MemoryRecord, MemoryStore


def memory_note_to_record(note: Any, sample_id: int) -> MemoryRecord:
    memory_id = str(getattr(note, "id"))
    content = str(getattr(note, "content", ""))
    context = str(getattr(note, "context", ""))
    keywords = tuple(str(item) for item in getattr(note, "keywords", []) or [])
    tags = tuple(str(item) for item in getattr(note, "tags", []) or [])
    links = tuple(str(item) for item in getattr(note, "links", []) or [])
    timestamp = getattr(note, "timestamp", None)
    text = (
        f"talk start time: {timestamp}\n"
        f"memory content: {content}\n"
        f"memory context: {context}\n"
        f"memory keywords: {', '.join(keywords)}\n"
        f"memory tags: {', '.join(tags)}"
    )
    return MemoryRecord(
        memory_id=memory_id,
        sample_id=sample_id,
        text=text,
        timestamp=None if timestamp is None else str(timestamp),
        content=content,
        summary=context,
        keywords=keywords,
        tags=tags,
        links=links,
        metadata={"source_method": "amem"},
    )


def memories_to_store(
    memories: Mapping[str, Any],
    sample_id: int,
    private_refs: Mapping[str, str] | None = None,
) -> MemoryStore:
    records = tuple(memory_note_to_record(note, sample_id) for note in memories.values())
    nodes = tuple(
        MemoryNode(
            node_id=record.memory_id,
            node_type="amem_note",
            text=record.text,
            label=record.content,
            timestamp=record.timestamp,
            properties={"keywords": list(record.keywords), "tags": list(record.tags)},
        )
        for record in records
    )
    edges = []
    for record in records:
        for target_id in record.links:
            edges.append(
                MemoryEdge(
                    edge_id=f"{record.memory_id}->{target_id}",
                    source_id=record.memory_id,
                    target_id=target_id,
                    edge_type="amem_link",
                )
            )
    return MemoryStore(
        sample_id=sample_id,
        records=records,
        nodes=nodes,
        edges=tuple(edges),
        private_refs=dict(private_refs or {}),
        metadata={"source_method": "amem"},
    )
```

- [ ] **Step 5: Run serialization tests**

Run: `uv run python -m pytest tests/test_amem_serialization.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/amem/methods/__init__.py src/amem/methods/amem/__init__.py src/amem/methods/amem/serialization.py tests/test_amem_serialization.py
git commit -m "feat: normalize amem memories to benchmark stores"
```

---

### Task 6: Build Script Writes Normalized Stores

**Files:**
- Modify: `scripts/build_memories.py`
- Test: `tests/test_experiment_entrypoints.py`

**Interfaces:**
- Consumes: `memories_to_store`, `write_memory_store`, `write_jsonl`, `to_jsonable`.
- Produces normalized files under `construction_run_XX/normalized/`:
  - `memory_store_sample_<idx>.json`
  - `memory_records_sample_<idx>.jsonl`
  - `memory_nodes_sample_<idx>.jsonl`
  - `memory_edges_sample_<idx>.jsonl`

- [ ] **Step 1: Add failing entrypoint test for normalized output path helpers**

Append to `tests/test_experiment_entrypoints.py`:

```python
from pathlib import Path

from amem.benchmark.artifacts import read_memory_store
from amem.benchmark.schemas import MemoryRecord, MemoryStore
from amem.methods.amem.serialization import memories_to_store


def test_amem_store_files_use_normalized_construction_directory(tmp_path: Path):
    store = MemoryStore(
        sample_id=0,
        records=(MemoryRecord(memory_id="m0", sample_id=0, text="hello"),),
    )
    normalized_dir = tmp_path / "construction_run_00" / "normalized"
    from amem.benchmark.artifacts import write_memory_store

    write_memory_store(normalized_dir / "memory_store_sample_0.json", store)

    assert read_memory_store(normalized_dir / "memory_store_sample_0.json") == store
```

Run: `uv run python -m pytest tests/test_experiment_entrypoints.py::test_amem_store_files_use_normalized_construction_directory -v`

Expected: PASS if Tasks 1 and 3 are complete. This guards the artifact location before editing the script.

- [ ] **Step 2: Modify `build_sample_cache` to write normalized files**

In `scripts/build_memories.py`, add imports:

```python
from amem.benchmark.artifacts import write_jsonl, write_memory_store  # noqa: E402
from amem.benchmark.schemas import to_jsonable  # noqa: E402
from amem.methods.amem.serialization import memories_to_store  # noqa: E402
```

Inside `build_sample_cache`, after existing pickle and retriever saves, add:

```python
    normalized_dir = output_dir / "normalized"
    private_refs = {
        "memory_cache": str(memory_cache_file.relative_to(output_dir)),
        "retriever_cache": str(retriever_cache_file.relative_to(output_dir)),
        "retriever_embeddings": str(retriever_embeddings_file.relative_to(output_dir)),
    }
    store = memories_to_store(agent.memories, sample_idx, private_refs=private_refs)
    write_memory_store(normalized_dir / f"memory_store_sample_{sample_idx}.json", store)
    write_jsonl(
        normalized_dir / f"memory_records_sample_{sample_idx}.jsonl",
        [to_jsonable(record) for record in store.records],
    )
    write_jsonl(
        normalized_dir / f"memory_nodes_sample_{sample_idx}.jsonl",
        [to_jsonable(node) for node in store.nodes],
    )
    write_jsonl(
        normalized_dir / f"memory_edges_sample_{sample_idx}.jsonl",
        [to_jsonable(edge) for edge in store.edges],
    )
```

- [ ] **Step 3: Add normalized file existence to metadata check**

In `build_construction_run`, after the existing `missing` check, add:

```python
    normalized_missing = [
        output_dir / "normalized" / f"memory_store_sample_{sample_idx}.json"
        for sample_idx in sample_indices
        if not (output_dir / "normalized" / f"memory_store_sample_{sample_idx}.json").exists()
    ]
    if normalized_missing:
        raise RuntimeError(f"Construction run missing normalized memory stores: {normalized_missing}")
```

- [ ] **Step 4: Run compile and focused entrypoint tests**

Run:

```bash
uv run python -m py_compile scripts/build_memories.py
uv run python -m pytest tests/test_experiment_entrypoints.py tests/test_amem_serialization.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_memories.py tests/test_experiment_entrypoints.py
git commit -m "feat: export normalized stores during memory build"
```

---

### Task 7: Context Builders and Normalized Result Writer

**Files:**
- Create: `src/amem/benchmark/context.py`
- Create: `src/amem/benchmark/results.py`
- Test: `tests/test_benchmark_results.py`

**Interfaces:**
- Consumes: `MemoryStore`, `RetrievedItem`, `QAResult`, `UsageRecord`.
- Produces:
  - `build_memory_fields_context(store: MemoryStore, item_ids: Sequence[str], fields: Sequence[str]) -> dict[str, Any]`
  - `write_run_results(run_dir: Path, results: Sequence[QAResult]) -> None`
  - `flatten_usage_rows(results: Sequence[QAResult]) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing context/result tests**

Add `tests/test_benchmark_results.py`:

```python
import json
from pathlib import Path

from amem.benchmark.context import build_memory_fields_context
from amem.benchmark.results import flatten_usage_rows, write_run_results
from amem.benchmark.schemas import MemoryRecord, MemoryStore, QAResult, UsageRecord


def test_build_memory_fields_context_includes_configured_fields_only():
    store = MemoryStore(
        sample_id=0,
        records=(
            MemoryRecord(
                memory_id="m0",
                sample_id=0,
                text="full",
                timestamp="t",
                content="content",
                keywords=("k1", "k2"),
                tags=("hidden",),
            ),
        ),
    )

    context = build_memory_fields_context(store, ["m0"], ["timestamp", "content", "keywords"])

    assert "content" in context["text"]
    assert "k1" in context["text"]
    assert "hidden" not in context["text"]


def test_write_run_results_writes_json_jsonl_and_usage_summary(tmp_path: Path):
    result = QAResult(
        experiment_id="exp",
        construction_run=0,
        qa_run=0,
        sample_id=0,
        qa_idx=1,
        question="q",
        reference="r",
        prediction="p",
        category=1,
        metrics={"f1": 1.0},
        retrieval={"items": []},
        context={"text": ""},
        prompt="prompt",
        usage=(UsageRecord(phase="qa", call_id="answer", total_tokens=12),),
    )

    write_run_results(tmp_path, [result])

    assert (tmp_path / "results.jsonl").exists()
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["individual_results"][0]["qa_idx"] == 1
    assert json.loads((tmp_path / "usage_summary.json").read_text(encoding="utf-8"))["by_source"]["reported"]["total_tokens"] == 12
    assert flatten_usage_rows([result])[0]["total_tokens"] == 12
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run python -m pytest tests/test_benchmark_results.py -v`

Expected: FAIL with missing modules.

- [ ] **Step 3: Add context builder**

Create `src/amem/benchmark/context.py`:

```python
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
```

- [ ] **Step 4: Add result writer**

Create `src/amem/benchmark/results.py`:

```python
"""Normalized result writing helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from .artifacts import write_json, write_qa_results_jsonl, write_usage_summary
from .schemas import QAResult, to_jsonable


def write_run_results(run_dir: Path, results: Sequence[QAResult]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_qa_results_jsonl(run_dir / "results.jsonl", results)
    write_json(
        run_dir / "results.json",
        {
            "total_questions": len(results),
            "individual_results": [to_jsonable(result) for result in results],
        },
    )
    usage_records = [record for result in results for record in result.usage]
    write_usage_summary(run_dir / "usage_summary.json", usage_records)


def flatten_usage_rows(results: Sequence[QAResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for record in result.usage:
            rows.append(
                {
                    "experiment_id": result.experiment_id,
                    "construction_run": result.construction_run,
                    "qa_run": result.qa_run,
                    "sample_id": result.sample_id,
                    "qa_idx": result.qa_idx,
                    "phase": record.phase,
                    "call_id": record.call_id,
                    "source": record.source,
                    "provider": record.provider,
                    "model": record.model,
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                    "total_tokens": record.total_tokens,
                    "estimated_tokens": record.estimated_tokens,
                    "latency_seconds": record.latency_seconds,
                    "cost_usd": record.cost_usd,
                }
            )
    return rows


def write_usage_csv(path: Path, results: Sequence[QAResult]) -> None:
    rows = flatten_usage_rows(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 5: Run focused result tests**

Run: `uv run python -m pytest tests/test_benchmark_results.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/amem/benchmark/context.py src/amem/benchmark/results.py tests/test_benchmark_results.py
git commit -m "feat: add normalized context and result writers"
```

---

### Task 8: Evaluation Writes Normalized QA Results Beside Existing Results

**Files:**
- Create: `src/amem/methods/amem/qa.py`
- Modify: `scripts/evaluate_memories.py`
- Test: `tests/test_benchmark_results.py`
- Test: `tests/test_experiment_entrypoints.py`

**Interfaces:**
- Consumes: existing robust result dictionaries from `evaluate_robust_run`.
- Produces: `robust_dict_to_qa_results(payload: Mapping[str, Any]) -> list[QAResult]`.
- Produces normalized `results.jsonl`, normalized `results.json`, and `usage_summary.json` in each robust `qa_run_XX` directory.

- [ ] **Step 1: Write failing QA normalization test**

Append to `tests/test_benchmark_results.py`:

```python
from amem.methods.amem.qa import robust_dict_to_qa_results


def test_robust_dict_to_qa_results_normalizes_existing_payload():
    payload = {
        "construction_run": 0,
        "qa_run": 1,
        "individual_results": [
            {
                "sample_id": 2,
                "qa_idx": 3,
                "question": "q",
                "reference": "r",
                "prediction": "p",
                "category": 4,
                "metrics": {"f1": 0.5},
                "retrieval_info": {"indices": [0]},
                "raw_context": "ctx",
                "user_prompt": "prompt",
            }
        ],
    }

    results = robust_dict_to_qa_results(payload, experiment_id="exp")

    assert results[0].experiment_id == "exp"
    assert results[0].retrieval["info"]["indices"] == [0]
    assert results[0].context["text"] == "ctx"
```

- [ ] **Step 2: Run focused QA normalization test and verify failure**

Run: `uv run python -m pytest tests/test_benchmark_results.py::test_robust_dict_to_qa_results_normalizes_existing_payload -v`

Expected: FAIL with missing `amem.methods.amem.qa`.

- [ ] **Step 3: Add robust QA normalization helper**

Create `src/amem/methods/amem/qa.py`:

```python
"""Normalize existing A-Mem robust QA outputs."""

from __future__ import annotations

from typing import Any, Mapping

from amem.benchmark.schemas import QAResult


def robust_dict_to_qa_results(payload: Mapping[str, Any], experiment_id: str) -> list[QAResult]:
    results = []
    for row in payload.get("individual_results", []):
        results.append(
            QAResult(
                experiment_id=experiment_id,
                construction_run=int(payload.get("construction_run", 0)),
                qa_run=int(payload.get("qa_run", 0)),
                sample_id=int(row["sample_id"]),
                qa_idx=int(row.get("qa_idx", 0)),
                question=str(row["question"]),
                reference=str(row.get("reference", "")),
                prediction=str(row.get("prediction", "")),
                category=row.get("category"),
                metrics=row.get("metrics", {}),
                retrieval={"info": row.get("retrieval_info", {}), "items": []},
                context={"text": row.get("raw_context", "")},
                prompt=row.get("user_prompt"),
                metadata={"source_format": "amem_robust"},
            )
        )
    return results
```

- [ ] **Step 4: Modify `write_robust_run` to write normalized result artifacts**

In `scripts/evaluate_memories.py`, add imports:

```python
from amem.benchmark.results import write_run_results  # noqa: E402
from amem.methods.amem.qa import robust_dict_to_qa_results  # noqa: E402
```

In `write_robust_run`, after the existing `json.dump`, add normalized output under a subdirectory so the legacy robust `results.json` remains unchanged:

```python
    normalized_results = robust_dict_to_qa_results(result, args.experiment_id)
    normalized_dir = run_dir / "normalized"
    write_run_results(normalized_dir, normalized_results)
```

Use the `normalized/` subdirectory in this first integration step to avoid changing current dashboard assumptions before Task 9.

- [ ] **Step 5: Run compile and focused tests**

Run:

```bash
uv run python -m py_compile scripts/evaluate_memories.py src/amem/methods/amem/qa.py
uv run python -m pytest tests/test_benchmark_results.py tests/test_experiment_entrypoints.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/amem/methods/amem/qa.py scripts/evaluate_memories.py tests/test_benchmark_results.py
git commit -m "feat: write normalized qa results for robust evaluation"
```

---

### Task 9: Dashboard Loader Reads Normalized Results with Fallback

**Files:**
- Modify: `scripts/experiment_data_loader.py`
- Test: `tests/test_experiment_data_loader.py`

**Interfaces:**
- Consumes normalized robust path: `construction_run_XX/robust/qa_run_XX/normalized/results.json`.
- Preserves fallback path: `construction_run_XX/robust/qa_run_XX/results.json`.
- Produces result maps with `question_key`, `usage`, `retrieval`, and `context` fields when normalized data is available.

- [ ] **Step 1: Add failing data-loader test for normalized preference**

Append to `tests/test_experiment_data_loader.py`:

```python
import json
from pathlib import Path

from scripts.experiment_data_loader import load_experiment_results


def test_load_experiment_results_prefers_normalized_results(tmp_path: Path):
    root = tmp_path / "results"
    run_dir = root / "exp" / "construction_run_00" / "robust" / "qa_run_00"
    (run_dir / "normalized").mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps({"individual_results": [{"sample_id": 0, "question": "old"}]}),
        encoding="utf-8",
    )
    (run_dir / "normalized" / "results.json").write_text(
        json.dumps(
            {
                "individual_results": [
                    {
                        "sample_id": 0,
                        "qa_idx": 0,
                        "question": "new",
                        "reference": "r",
                        "prediction": "p",
                        "category": 1,
                        "metrics": {"f1": 1.0},
                        "usage": [{"total_tokens": 10, "source": "reported"}],
                        "retrieval": {"items": []},
                        "context": {"text": "ctx"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    data = load_experiment_results("exp", 0, 0, results_root=root)

    assert data["individual_results"][0]["question"] == "new"
    assert data["individual_results"][0]["usage"][0]["total_tokens"] == 10
    assert "question_key" in data["individual_results"][0]
```

- [ ] **Step 2: Run focused loader test and verify failure**

Run: `uv run python -m pytest tests/test_experiment_data_loader.py::test_load_experiment_results_prefers_normalized_results -v`

Expected: FAIL because loader reads only legacy `results.json`.

- [ ] **Step 3: Modify loader path selection**

In `scripts/experiment_data_loader.py`, update `load_experiment_results` path logic:

```python
    run_dir = (
        results_root
        / experiment_id
        / f"construction_run_{construction_run:02d}"
        / "robust"
        / f"qa_run_{qa_run:02d}"
    )
    normalized_path = run_dir / "normalized" / "results.json"
    legacy_path = run_dir / "results.json"
    path = normalized_path if normalized_path.exists() else legacy_path
```

Keep the existing `if not path.exists()` check and `load_result_file(path)` call.

- [ ] **Step 4: Make `load_result_file` tolerate normalized result keys**

In `load_result_file`, keep existing question key logic and avoid assuming legacy-only fields:

```python
    for r in data.get("individual_results", []):
        r["question_key"] = question_key(r["sample_id"], r["question"])
        r.setdefault("retrieval_info", r.get("retrieval", {}))
        r.setdefault("raw_context", r.get("context", {}).get("text", ""))
```

- [ ] **Step 5: Run dashboard loader tests**

Run: `uv run python -m pytest tests/test_experiment_data_loader.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/experiment_data_loader.py tests/test_experiment_data_loader.py
git commit -m "feat: load normalized qa results in dashboard data layer"
```

---

### Task 10: Final Compatibility Verification and Documentation Link

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Test: existing focused test suite.

**Interfaces:**
- Consumes all previous tasks.
- Produces documented migration path and verified compatibility.

- [ ] **Step 1: Add README note for normalized artifacts**

In `README.md`, under the two-stage pipeline section, add:

```markdown
The component benchmark migration also writes normalized artifacts beside legacy
A-Mem caches/results. Normalized construction stores live under
`construction_run_XX/normalized/`, and normalized QA rows live under
`qa_run_XX/normalized/`. These files are intended for cross-method comparisons
and dashboard loading; legacy pickle caches and robust result files remain
available during the transition.
```

- [ ] **Step 2: Add AGENTS.md pointer to the design and plan**

In `AGENTS.md`, under "Retrieval and Reranking Designs", add:

```markdown
- `docs/superpowers/specs/2026-07-07-benchmark-component-architecture-design.md`:
  component benchmark architecture for mixing construction, retrieval, QA,
  graph memory, non-graph RAG, and token usage accounting.
- `docs/superpowers/plans/2026-07-07-benchmark-component-architecture.md`:
  implementation plan for the first component benchmark migration slice.
```

- [ ] **Step 3: Run compile checks**

Run:

```bash
uv run python -m py_compile scripts/build_memories.py scripts/evaluate_memories.py scripts/experiment_data_loader.py
```

Expected: exit code 0.

- [ ] **Step 4: Run benchmark foundation tests**

Run:

```bash
uv run python -m pytest \
  tests/test_benchmark_schemas.py \
  tests/test_benchmark_hooks.py \
  tests/test_benchmark_artifacts.py \
  tests/test_benchmark_config.py \
  tests/test_amem_serialization.py \
  tests/test_benchmark_results.py \
  tests/test_experiment_data_loader.py \
  -v
```

Expected: all tests pass.

- [ ] **Step 5: Run existing compatibility tests**

Run:

```bash
uv run python -m pytest \
  tests/test_experiment_common.py \
  tests/test_experiment_config.py \
  tests/test_experiment_entrypoints.py \
  tests/test_retrieval_pipeline.py \
  tests/test_reproduction_package.py \
  -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: document component benchmark migration"
```

---

## Self-Review

**Spec coverage:** This plan covers normalized `MemoryStore`, graph-capable schemas, one-shot retrieval compatibility, token usage hook design, artifact layout, dashboard loading, and A-Mem compatibility. It intentionally defers full MRAgent/Zep adapter implementations and persistent graph database exports to follow-up plans.

**Marker scan:** The plan contains no deferred-detail markers or unspecified test-writing steps. Each task includes concrete files, interfaces, test snippets, commands, and commit messages.

**Type consistency:** The shared schema names are consistent across tasks: `MemoryStore`, `MemoryRecord`, `RetrievedItem`, `RetrievalToolCall`, `UsageRecord`, and `QAResult`. Artifact helpers consistently use `to_jsonable` and `from_jsonable`. Normalized QA output is written under `qa_run_XX/normalized/` in the first integration step to avoid breaking legacy `results.json`.
