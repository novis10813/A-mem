# MemoryBench

MemoryBench is a component-oriented framework for reproducible memory-system experiments:

```text
Dataset → Construction → Retrieve/QA → Artifacts → Dashboard
```

The first release provides a LoCoMo adapter, stable question/evidence IDs, turn-level RAG,
graph-capable memory stores, staged retrieval traces, canonical resumable artifacts, and a
read-only research dashboard. The legacy `amem` namespace, cache layout, and entrypoints are
intentionally unsupported.

## Quick start

```bash
uv sync --extra dev
uv run python -m memorybench validate --config configs/turn_rag_smoke.yaml
uv run python -m memorybench run --config configs/turn_rag_smoke.yaml
uv run python -m memorybench list-components
```

Dashboard dependencies are optional:

```bash
uv sync --extra dashboard
uv run python -m memorybench dashboard --artifact-root artifacts/experiments
```

Canonical experiments live under `artifacts/experiments/<experiment_id>/`. Each experiment
contains a resolved manifest, per-run construction stores, per-question QA JSONL rows, status,
retrieval traces, metrics, usage, errors, and provenance. Files are atomically committed and
resume is accepted only for completed units with the same config fingerprint.

For a smoke run without network services, use `configs/turn_rag_smoke.yaml`.
