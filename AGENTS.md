# AGENTS.md

## Repository

MemoryBench is rooted in this directory. Library and CLI code belongs under
`src/memorybench/`; tests belong under `tests/`; `scripts/` may contain only shell wrappers
around `python -m memorybench`. Generated output belongs under `artifacts/` and must not be
committed.

This repository intentionally has no `amem` package, legacy Python entrypoints, or old cache
compatibility. Native A-Mem implementation code lives under `memorybench.amem_native` and its
normalized adapters under `memorybench.methods.amem`.

## Commands

Use `uv` and Python 3.13:

```bash
uv sync --extra dev
uv run python -m memorybench validate --config configs/turn_rag_smoke.yaml
uv run python -m memorybench run --config configs/turn_rag_smoke.yaml
uv run python -m pytest -v
uv run python -m compileall -q src/memorybench tests
```

Keep optional dashboard, provider, retrieval, and heavy metric dependencies lazy. Core runs
must not initialize NLTK, BERTScore, or SentenceTransformer unless their adapter is selected.
