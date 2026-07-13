# MemoryBench

MemoryBench is a component-oriented framework for reproducible memory-system experiments:

```text
Dataset → Construction → Retrieve/QA → Artifacts → Dashboard
```

The first release provides a LoCoMo adapter, stable question/evidence IDs, native A-Mem and
turn-level RAG construction, graph-capable memory stores, staged retrieval traces, canonical
resumable artifacts, and a read-only research dashboard. The legacy `amem` namespace, cache
layout, and entrypoints are intentionally unsupported.

## Quick start

```bash
uv sync --extra dev
uv run python -m memorybench validate --config configs/turn_rag_smoke.yaml
uv run python -m memorybench run --config configs/turn_rag_smoke.yaml
uv run python -m memorybench run --config configs/amem_fake_smoke.yaml
uv run python -m memorybench list-components
```

Dashboard dependencies are optional:

```bash
uv sync --extra dashboard
uv run python -m memorybench dashboard --artifact-root artifacts/experiments
```

Canonical experiments live under `artifacts/experiments/<experiment_id>/`. Construction is
committed per sample and retrieve/QA per stable question ID. Each experiment contains a resolved
manifest with dataset/Git provenance, sharded records/nodes/edges/layers, per-question QA rows,
retrieval traces, metrics, reported-or-estimated usage, structured errors, and summaries. Resume
skips only completed units with the same config fingerprint and retries failed units.

For runs without network services, use `configs/turn_rag_smoke.yaml` and
`configs/amem_fake_smoke.yaml`. See [the architecture and config guide](docs/memorybench_architecture.md)
for component contracts, phase selection, external memory sources, taxonomy, and artifact layout.
