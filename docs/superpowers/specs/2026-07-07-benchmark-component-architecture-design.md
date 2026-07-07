# Benchmark Component Architecture Design

Date: 2026-07-07

## Goal

Restructure the repository so experiments can compare and mix different memory
construction, retrieval, and QA methods while reusing the existing two-stage
experiment flow and Gradio visualization pipeline.

The current codebase is centered on A-Mem reproduction. That should remain the
first supported method, but future experiments need to compose pieces such as:

- A-Mem construction + BM25 retrieval + robust QA.
- A-Mem construction + embedding retrieval + CrossEncoder rerank + another QA prompt.
- Another paper's construction + existing retrieval pipeline + same QA evaluator.
- Raw-turn or summary-based construction baselines + shared retrieval and QA.
- Heterogeneous or hierarchical graph memory construction + graph access tools +
  tool-calling QA.
- Non-graph RAG construction + one-shot retrieval + the same QA and metrics.

Examples that should fit this architecture include MRAgent-style graph
episodic memory, Zep/Graphiti-style temporal knowledge graphs, A-Mem-style
linked notes, and ordinary chunked RAG baselines.

Reference requirements from target methods:

- MRAgent builds graph-structured episodic memory from rewritten dialogue turns,
  keywords, topic/personal event nodes, and graph links, then answers through a
  tool-calling loop over keyword, topic, personal, temporal, and context tools.
- Zep/Graphiti uses temporal knowledge graph memory with episode, semantic
  entity, and community subgraphs, plus temporal metadata for evolving facts and
  relationships.
- Ordinary RAG baselines may have no graph at all and should still use the same
  construction/retrieval/QA comparison surface.

## Design Principles

Use three formal component boundaries:

- `construction`: converts a conversation sample into a `MemoryStore`.
- `retrieval`: exposes memory access, either as one-shot retrieved items or as
  an interactive toolset used during QA.
- `qa`: converts a question and context into a prediction.

`MemoryStore`, not `MemoryRecord`, is the main interoperability layer. Flat text
records are one view over a memory store. Graph methods can also export nodes,
edges, layers, temporal metadata, and private graph indices.

Keep one-shot retrieval internally composable with explicit stages because
retrieval variants are often the experimental condition. Graph and agentic
methods may instead expose a retrieval toolset. Use hooks for lower-level
behavior, tracing, and small transformations, but do not hide major experimental
conditions inside hooks.

Any hook or stage that changes experiment behavior must be written into the
manifest and per-result trace.

## Proposed Package Layout

```text
src/amem/
  benchmark/
    __init__.py
    schemas.py
    registry.py
    runner.py
    artifacts.py
    hooks.py
    metrics.py
    datasets.py
    config.py

  methods/
    amem/
      __init__.py
      construction.py
      retrieval.py
      context.py
      qa.py
      serialization.py

    graph/
      __init__.py
      memory_store.py
      temporal_graph.py
      toolsets.py

    baselines/
      raw_turns.py
      summary_memory.py
      closed_book_qa.py

  retrieval_pipeline.py
  reranking.py
  memory_layer_robust.py
```

Existing A-Mem implementation files can stay in place initially. The new
`methods/amem/` package should wrap them instead of rewriting internals first.

## Core Schemas

`src/amem/benchmark/schemas.py` should define stable dataclasses used by all
methods. `MemoryStore` is the root artifact. `MemoryRecord` is the flat text
view used by normal RAG, BM25, embedding retrieval, and dashboard summaries.
Graph methods can populate `MemoryNode`, `MemoryEdge`, and `MemoryLayer` without
losing their native structure.

```python
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
    errors: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Each construction method may keep private caches, but it must export a
normalized `MemoryStore`. Non-graph RAG may only populate `records`. A-Mem can
populate records plus note-link edges. MRAgent-style or Zep/Graphiti-style
methods can populate episode, topic, entity, event, and community nodes plus
heterogeneous or temporal edges.

The store should support two views:

- `records`: stable text records for one-shot retrieval and simple dashboards.
- `graph`: nodes, edges, and layers for graph traversal, temporal reasoning, and
  tool-calling QA.

Do not force graph methods to flatten their native memory into records only.
Flattening is useful as a derived baseline, not as the canonical artifact.

## Component Interfaces

Use protocols rather than inheritance-heavy base classes.

```python
class ConstructionAdapter(Protocol):
    name: str

    def build_sample(
        self,
        sample: Any,
        sample_id: int,
        output_dir: Path,
        config: Mapping[str, Any],
        hooks: Sequence[Hook],
    ) -> MemoryStore:
        ...


class RetrievalAdapter(Protocol):
    name: str

    def retrieve(
        self,
        question: str,
        store: MemoryStore,
        config: Mapping[str, Any],
        hooks: Sequence[Hook],
    ) -> RetrievalOutput:
        ...


class RetrievalToolset(Protocol):
    name: str

    def tools(
        self,
        store: MemoryStore,
        config: Mapping[str, Any],
        hooks: Sequence[Hook],
    ) -> Mapping[str, Callable[..., RetrievalToolCall]]:
        ...


class ContextBuilder(Protocol):
    name: str

    def build_context(
        self,
        question: str,
        retrieved: Sequence[RetrievedItem],
        store: MemoryStore,
        config: Mapping[str, Any],
    ) -> ContextOutput:
        ...


class QAAdapter(Protocol):
    name: str

    def answer(
        self,
        question: str,
        context: ContextOutput,
        reference: str | None,
        config: Mapping[str, Any],
        hooks: Sequence[Hook],
        retrieval_tools: Mapping[str, Callable[..., RetrievalToolCall]] | None = None,
    ) -> QAOutput:
        ...
```

The runner owns dataset iteration, resume behavior, artifact paths, metrics, and
result writing. Adapters own method-specific behavior.

One-shot QA adapters usually receive a concrete context. Tool-calling QA
adapters may receive a retrieval toolset and build context incrementally through
tool calls. Both paths must write comparable `QAResult` objects.

## Hooks

Hooks are for behavior that cuts across components or makes small transformations.

Recommended hook events:

- `before_sample_construction`
- `after_memory_item_created`
- `after_sample_construction`
- `before_retrieval`
- `after_retrieval_stage`
- `after_retrieval`
- `before_tool_call`
- `after_tool_call`
- `before_prompt`
- `after_llm`
- `after_answer_parse`
- `on_error`

Appropriate hook uses:

- timing and trace collection;
- keyword pruning;
- answer parsing;
- prompt template substitution;
- debug dumps;
- field ablations;
- context text redaction or formatting experiments.
- graph-to-record flattening baselines;
- temporal filtering or timestamp normalization when configured as an explicit
  experiment condition.

Do not use hooks to silently replace a major component such as the retrieval
algorithm, construction semantics, or QA task definition. Those should be
explicit adapters, retrieval stages, or named toolsets.

## Retrieval and Memory Access

Retrieval is better described as memory access because some methods are not
single-pass top-k retrieval systems. Support two retrieval families.

### One-Shot Retrieval

Keep the existing `src/amem/retrieval_pipeline.py` direction, but make it consume
the `records` view of `MemoryStore` instead of A-Mem memory objects where
possible.

Retrieval config should remain explicit:

```yaml
retrieval:
  adapter: pipeline
  stages:
    - type: embedding
      name: embedding_candidates
      top_k: 50
      query: generated_query
    - type: cross_encoder
      name: cross_encoder_rerank
      top_k: 10
      query: original_question
```

This keeps experimental factors visible and makes result traces explainable.

### Interactive Retrieval Toolsets

Graph methods such as MRAgent-style graph memory may answer questions through an
LLM tool-calling loop. In this case the retrieval adapter exposes tools rather
than returning one final top-k list before QA.

```yaml
retrieval:
  adapter: graph_toolset
  tools:
    - query_event_keywords
    - query_topic_events
    - query_personal_information
    - query_conversation_time
    - query_event_context
  params:
    max_tool_calls: 8
```

Every tool call must be logged as `RetrievalToolCall` entries in the QA result.
This makes tool-calling graph QA comparable to one-shot RAG in the dashboard:
both expose retrieved support, access traces, and final predictions.

### Graph and Temporal Retrieval

Graph retrieval adapters may use nodes, edges, layers, and temporal fields:

- heterogeneous node types such as episodes, topics, people, facts, entities,
  events, or communities;
- edge types such as mentions, source, semantic relation, temporal predecessor,
  topic membership, and community membership;
- layer names such as `episode`, `semantic_entity`, `personal_event`, `topic`,
  or `community`;
- temporal fields such as `timestamp`, `valid_at`, and `invalid_at`.

Graph retrieval can still emit `RetrievedItem` values for common result display.
The item metadata should include the backing node IDs, edge IDs, path, layer,
and temporal filters used.

## Config Shape

Move from A-Mem-specific `qa_mode` to component composition.

```yaml
experiment_id: amem_construct_bm25_robust_qa
dataset: data/locomo10.json

construction:
  adapter: amem
  runs: 1
  params:
    keyword_pruning_mode: nltk
    embedding_model: all-MiniLM-L6-v2
  hooks:
    - type: timing
    - type: normalized_memory_export

retrieval:
  adapter: pipeline
  stages:
    - type: bm25
      name: bm25_candidates
      top_k: 50
      query: original_question
    - type: cross_encoder
      name: cross_encoder_rerank
      top_k: 10
      query: original_question
      model: cross-encoder/ms-marco-MiniLM-L6-v2
      batch_size: 32

context:
  adapter: memory_fields
  fields: [timestamp, content, keywords]
  include_links: false

qa:
  adapter: robust_plain_text
  runs: 30
  backend: ollama
  model: llama3.2:1b
  params:
    temperature_c5: 0.5
  hooks:
    - type: answer_parser
      parser: plain_text

metrics:
  adapters: [f1, bleu1]

limits:
  ratio: 1.0
  sample_limit: null
  turn_limit: null
  qa_limit: null

run:
  resume: true
  max_workers: 10
```

Compatibility can be preserved by translating the current YAML schema into this
new internal config during a transition period.

Graph and tool-calling experiments should use the same top-level component
shape:

```yaml
experiment_id: mragent_graph_tool_qa
dataset: data/locomo10.json

construction:
  adapter: mragent_graph
  runs: 1
  params:
    rewrite_turns: true
    extract_keywords: true
    graph_layers: [episode, topic, personal_event]

retrieval:
  adapter: graph_toolset
  tools:
    - query_event_keywords
    - query_topic_events
    - query_personal_information
    - query_conversation_time
    - query_event_context
  params:
    max_tool_calls: 8

context:
  adapter: tool_trace_context
  include_tool_outputs: true

qa:
  adapter: tool_calling_agent
  runs: 1
  backend: openai
  model: gpt-4o-mini

metrics:
  adapters: [f1, exact_match, llm_judge]
```

Flattened graph baselines should be explicit so they are not confused with the
native graph method:

```yaml
construction:
  adapter: mragent_graph

retrieval:
  adapter: pipeline
  view: records_from_graph
  stages:
    - type: embedding
      top_k: 10

qa:
  adapter: plain_rag_prompt
```

## Artifact Layout

Keep the existing `artifacts/` root and construction/QA run directories.

```text
artifacts/
  caches/<experiment_id>/
    manifest.json
    construction_run_00/
      metadata.json
      private/
        memory_cache_sample_0.pkl
        retriever_cache_sample_0.pkl
        retriever_cache_embeddings_sample_0.npy
      normalized/
        memory_store_sample_0.json
        memory_records_sample_0.jsonl
        memory_nodes_sample_0.jsonl
        memory_edges_sample_0.jsonl
        memory_store_manifest.json

  results/<experiment_id>/
    manifest.json
    construction_run_00/
      qa_run_00/
        results.jsonl
        results.json
        summary.json
      summary_across_runs.csv
      summary_across_runs.json
```

`private/` is adapter-owned and may contain pickle files. `normalized/` is the
stable interoperability layer. Dashboard and cross-method analysis should read
normalized stores and results only.

For large graphs, `memory_store_sample_0.json` may contain metadata and file
references instead of embedding every node and edge inline. JSONL files should be
the scalable row-oriented representation.

## Dashboard Changes

The Gradio data loader should stop assuming this path:

```text
construction_run_XX/robust/qa_run_XX/results.json
```

Instead, it should discover normalized result files and use stable fields:

- `question_key`
- `experiment_id`
- `construction.adapter`
- `retrieval.adapter` and stage list
- `context.adapter`
- `qa.adapter`
- `prediction`
- `reference`
- `metrics`
- `retrieval.items`
- `retrieval.tool_calls`
- `context.text`
- `prompt`

Mode-specific visualizations can still exist, but the default comparison view
should be method-neutral.

The dashboard should support three views over the same normalized result:

- flat QA comparison, using prediction/reference/metrics;
- retrieved support comparison, using `retrieval.items`;
- graph/tool trace inspection, using `retrieval.tool_calls`, node IDs, edge IDs,
  paths, and temporal filters where available.

## Migration Plan

1. Add benchmark schemas, config parsing, registry, and artifact helpers,
   including `MemoryStore`, `MemoryRecord`, `MemoryNode`, `MemoryEdge`, and
   `RetrievalToolCall`.
2. Add an A-Mem construction adapter that wraps `RobustAgenticMemorySystem` and
   writes both private pickle caches and normalized `MemoryStore` files.
3. Add retrieval adapters that operate on normalized stores and records:
   - embedding;
   - BM25;
   - CrossEncoder rerank;
   - limit/final selection.
4. Add graph-aware access adapters:
   - graph traversal retrieval;
   - temporal graph retrieval;
   - interactive retrieval toolsets.
5. Add context builders:
   - `amem_full`;
   - `content_keywords`;
   - generic `memory_fields`.
   - `tool_trace_context`;
   - graph path / node evidence context.
6. Add QA adapters:
   - current robust plain-text QA;
   - closed-book baseline;
   - tool-calling QA;
   - future paper-specific prompts.
7. Refactor `scripts/build_memories.py` into a thin runner over construction
   adapters while preserving old flags.
8. Refactor `scripts/evaluate_memories.py` into a thin runner over retrieval,
   context, and QA adapters while preserving old flags.
9. Update dashboard loader to consume normalized stores and results.
10. Deprecate old `qa_mode` branches after compatibility tests pass.

## Testing Strategy

Unit tests:

- schema JSON round trips;
- config translation from old schema to new component schema;
- registry rejects unknown adapters and invalid stage configs;
- normalized memory export preserves A-Mem content, timestamp, keywords, tags,
  and links;
- graph store export preserves node IDs, edge IDs, layer names, temporal fields,
  and private refs;
- retrieval stages produce deterministic ordering on toy records;
- graph retrieval produces deterministic paths on toy graphs;
- tool-calling QA records every tool call and retrieved support item;
- context builders include only configured fields;
- QA result writer emits dashboard-readable rows.

Compatibility tests:

- existing `tests/test_experiment_common.py`;
- existing `tests/test_experiment_config.py`;
- existing `tests/test_experiment_entrypoints.py`;
- existing `tests/test_retrieval_pipeline.py`;
- a smoke run using the old config path and the new internal runner.

Manual smoke test:

```bash
uv run python scripts/build_memories.py \
  --experiment-id smoke_component_arch \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --construction-runs 1 \
  --keyword-pruning-mode nltk \
  --sample-limit 1 \
  --turn-limit 2 \
  --max-workers 1 \
  --resume

uv run python scripts/evaluate_memories.py \
  --experiment-id smoke_component_arch \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --qa-mode robust \
  --qa-runs 1 \
  --sample-limit 1 \
  --qa-limit 1 \
  --max-workers 1 \
  --resume
```

## Non-Goals

- Do not rewrite A-Mem internals before adapters exist.
- Do not remove existing two-stage entrypoints immediately.
- Do not require every method to use the same private cache format.
- Do not make hooks powerful enough to obscure major experimental conditions.
- Do not force graph methods to degrade into flat RAG unless that is an explicit
  baseline condition.
- Do not require non-graph RAG baselines to populate graph nodes and edges.

## Open Decisions

- Whether normalized memory stores should be one set of JSON/JSONL files per
  sample or one set per construction run. The initial recommendation is per
  sample to match current cache files and resume semantics.
- Whether dashboard should read `results.jsonl` directly or a compact
  `results.json` aggregate. The initial recommendation is to write both:
  JSONL for scalable row-level analysis and JSON for current compatibility.
- Whether graph stores should use plain JSONL only or optionally support a local
  graph database export for very large runs. The initial recommendation is
  JSON/JSONL first, with private refs pointing to method-owned graph databases
  if needed.
