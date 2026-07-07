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

## Design Principles

Use three formal component boundaries:

- `construction`: converts a conversation sample into memory artifacts.
- `retrieval`: converts a question and memory records into retrieved items.
- `qa`: converts a question and context into a prediction.

Keep retrieval internally composable with explicit stages because retrieval
variants are often the experimental condition. Use hooks for lower-level
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
methods.

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
class RetrievedItem:
    memory_id: str
    rank: int
    text: str
    score: float | None = None
    source_stage: str = ""
    trace: tuple[Mapping[str, Any], ...] = ()
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

`MemoryRecord` is the key compatibility layer. Each construction method may keep
private caches, but it must export normalized records so retrieval and dashboard
code can be method-neutral.

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
    ) -> ConstructionOutput:
        ...


class RetrievalAdapter(Protocol):
    name: str

    def retrieve(
        self,
        question: str,
        records: Sequence[MemoryRecord],
        config: Mapping[str, Any],
        hooks: Sequence[Hook],
    ) -> RetrievalOutput:
        ...


class ContextBuilder(Protocol):
    name: str

    def build_context(
        self,
        question: str,
        retrieved: Sequence[RetrievedItem],
        records_by_id: Mapping[str, MemoryRecord],
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
    ) -> QAOutput:
        ...
```

The runner owns dataset iteration, resume behavior, artifact paths, metrics, and
result writing. Adapters own method-specific behavior.

## Hooks

Hooks are for behavior that cuts across components or makes small transformations.

Recommended hook events:

- `before_sample_construction`
- `after_memory_record_created`
- `after_sample_construction`
- `before_retrieval`
- `after_retrieval_stage`
- `after_retrieval`
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

Do not use hooks to silently replace a major component such as the retrieval
algorithm, construction semantics, or QA task definition. Those should be
explicit adapters or retrieval stages.

## Retrieval Pipeline

Keep the existing `src/amem/retrieval_pipeline.py` direction, but make it consume
`MemoryRecord` instead of A-Mem memory objects where possible.

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
        memory_records_sample_0.jsonl
        memory_records_manifest.json

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
normalized results only.

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
- `context.text`
- `prompt`

Mode-specific visualizations can still exist, but the default comparison view
should be method-neutral.

## Migration Plan

1. Add benchmark schemas, config parsing, registry, and artifact helpers.
2. Add an A-Mem construction adapter that wraps `RobustAgenticMemorySystem` and
   writes both private pickle caches and normalized `MemoryRecord` JSONL files.
3. Add retrieval adapters that operate on normalized records:
   - embedding;
   - BM25;
   - CrossEncoder rerank;
   - limit/final selection.
4. Add context builders:
   - `amem_full`;
   - `content_keywords`;
   - generic `memory_fields`.
5. Add QA adapters:
   - current robust plain-text QA;
   - closed-book baseline;
   - future paper-specific prompts.
6. Refactor `scripts/build_memories.py` into a thin runner over construction
   adapters while preserving old flags.
7. Refactor `scripts/evaluate_memories.py` into a thin runner over retrieval,
   context, and QA adapters while preserving old flags.
8. Update dashboard loader to consume normalized results.
9. Deprecate old `qa_mode` branches after compatibility tests pass.

## Testing Strategy

Unit tests:

- schema JSON round trips;
- config translation from old schema to new component schema;
- registry rejects unknown adapters and invalid stage configs;
- normalized memory export preserves A-Mem content, timestamp, keywords, tags,
  and links;
- retrieval stages produce deterministic ordering on toy records;
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

## Open Decisions

- Whether normalized memory records should be one JSONL file per sample or one
  JSONL file per construction run. The initial recommendation is per sample to
  match current cache files and resume semantics.
- Whether dashboard should read `results.jsonl` directly or a compact
  `results.json` aggregate. The initial recommendation is to write both:
  JSONL for scalable row-level analysis and JSON for current compatibility.
