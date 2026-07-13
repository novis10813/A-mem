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

## FinanceBench (local PDF corpus)

Prepare the public FinanceBench question-linked PDF corpus once. Preparation downloads 84 source PDFs for the 150 public questions, extracts complete pages, and writes only ignored artifacts. Benchmark runs are offline after this step.

```bash
uv sync --extra dev --extra providers --extra financebench
uv run python -m memorybench prepare-financebench --output artifacts/datasets/financebench --workers 4
uv run python -m memorybench validate --config configs/financebench_llamacpp_smoke.yaml
uv run python -m memorybench run --config configs/financebench_llamacpp_smoke.yaml
```

The FinanceBench configs use the existing `vllm` provider label solely to call a llama.cpp OpenAI-compatible server at `http://127.0.0.1:8080/v1` with model `llama3.2`; this branch does not add a llama.cpp provider. Start full runs with `runtime.max_workers: 1`, then measure two and four workers before changing the configuration. The included exact match, F1, and BLEU-1 values are diagnostic metrics, not official FinanceBench scores.
