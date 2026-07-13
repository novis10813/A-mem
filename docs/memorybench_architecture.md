# MemoryBench Architecture and Configuration

MemoryBench executes one ordered pipeline:

```text
DatasetAdapter → ConstructionAdapter → RetrievalAdapter → ContextAdapter → QAAdapter → MetricAdapters
```

Adapters are registered by component family. `python -m memorybench list-components` is the
authoritative list of components that are currently runnable; future protocols such as interactive
tool retrieval are not advertised until an implementation and artifact trace exist.

## Phase configuration

`pipeline.stages` selects `construction`, `retrieve_qa`, or both. Dataset configuration only loads
and normalizes data. Construction and retrieve/QA have separate selectors because turn limits and
question/category filters serve different purposes.

For evaluation against an existing new-format cache:

```yaml
pipeline:
  stages: [retrieve_qa]
  dataset:
    adapter: locomo
    path: data/locomo10.json
  retrieve_qa:
    memory_source:
      experiment_id: built_memories
      construction_runs: [0]
    retrieval:
      adapter: staged
      stages:
        - adapter: bm25
          top_k: 10
    context:
      adapter: records
    qa:
      adapter: extractive
    # metrics and selection are optional
```

MemoryBench rejects missing sources and dataset fingerprint mismatches before writing a new run.

## Flexible retrieval stages

Staged retrieval currently supports BM25, embedding, embedding rerank, CrossEncoder, limit, and
query transform. Every stage owns its `params`, `top_k`, and optional `llm`, so query generation can
use a different provider/model from QA:

```yaml
retrieval:
  adapter: staged
  stages:
    - adapter: query_transform
      top_k: 50
      llm:
        provider: ollama
        model: llama3.2:1b
    - adapter: embedding
      top_k: 50
      params:
        model: all-MiniLM-L6-v2
    - adapter: cross_encoder
      top_k: 10
```

Each result stores input/output rankings, scores, effective query, resolved stage config, latency,
and LLM usage where applicable.

## Dataset taxonomy

Dataset adapters declare their own taxonomy dimensions. Questions store multi-valued textual labels;
LoCoMo uses `multi_hop`, `temporal`, `open_domain`, `single_hop`, and `adversarial`. Dashboard
breakdowns use these native labels. Cross-dataset mappings are optional metadata and never assumed
to be semantically equivalent.

## Artifact layout

```text
artifacts/experiments/<experiment_id>/
  manifest.json
  construction/run_000/
    status.json
    errors.jsonl
    usage.jsonl
    samples/<sample_key>/
      store.json
      records.jsonl
      nodes.jsonl
      edges.jsonl
      layers.jsonl
      status.json
      private/
  retrieve_qa/construction_000/run_000/
    status.json
    results.jsonl
    errors.jsonl
    usage.jsonl
    summary.json
    questions/<question_key>.json
```

All unit files are atomically replaced. A partial run exits with code `2`; fatal configuration or
startup failures exit `1`; a completed run exits `0`.

## Dashboard

The Gradio research workbench reads normalized artifacts only. It provides Overview, QA Compare,
Retrieval Trace, Memory Explorer, and Usage & Latency views. Reported and estimated token usage are
aggregated separately.
