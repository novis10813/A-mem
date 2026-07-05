# AGENTS.md

Guidance for coding agents working in this repository.

## Repository Layout

This is a reproduction-oriented fork of A-Mem. The actual repo root is this directory; do not assume code lives under `external/`.

- Core evaluation code:
  - `test_advanced_robust.py`: robust LoCoMo evaluation entrypoint.
  - `src/amem/memory_layer_robust.py`: robust memory system and LLM backend controllers.
  - `src/amem/reranking.py`: optional second-stage retrieval rerankers.
  - `src/amem/llm_text_parsers.py`: plain-text parsers and keyword pruning modes.
  - `src/amem/ablation.py`: fixed-memory embedding-field ablations.
- Preferred two-stage experiment entrypoints:
  - `scripts/build_memories.py`: memory-construction stage; builds reusable caches.
  - `scripts/evaluate_memories.py`: QA-evaluation stage; reads existing caches.
  - `scripts/run_experiment.sh`: convenience wrapper that builds first, then evaluates.
- Experiment and analysis tools live in `scripts/`.
- Design notes and baseline comparisons live in `docs/`.
- Pytest tests live in `tests/`; root-level `test_advanced*.py` files are evaluation entrypoints kept for compatibility.
- Library code lives in `src/amem/`; root-level modules such as `memory_layer_robust.py` are thin compatibility shims.
- Dataset lives in `data/`.
- Generated caches/results/logs are local artifacts under `artifacts/` and should not be committed.
- Root-level reference papers are ignored via `.gitignore`; keep papers under `papers/`.

## Environment

Use `uv` for Python commands.

```bash
uv sync
uv run python -m pytest tests/test_ablation.py -v
```

The project targets Python `>=3.13,<3.14` via `pyproject.toml`.

For Ollama experiments, make sure Ollama is running and the model is available:

```bash
ollama serve
ollama pull llama3.2:1b
```

Check availability:

```bash
curl -sf http://localhost:11434/api/tags
```

## Generated Artifacts

Do not commit generated experiment artifacts:

- `artifacts/`
- `.venv/`, `__pycache__/`, `.pytest_cache/`

Legacy root-level artifact paths remain ignored for compatibility:

- `cached_memories*/`
- `logs/`
- `results_*.json`
- `results_*/`
- `results_experiments/`
- `output/`
- `results_content_keyword_pruning/`
- `results_content_keyword_rebuild/`

If new generated-output directories are introduced, add them to `.gitignore` instead of committing them.

## Keyword Pruning Modes

Keyword pruning is controlled in `src/amem/llm_text_parsers.py`:

- `none`: normalize and deduplicate only.
- `simple`: rule filtering without stemming.
- `nltk`: rule filtering with PorterStemmer grounding.

The robust evaluator exposes this as:

```bash
uv run python test_advanced_robust.py \
  --backend ollama \
  --model llama3.2:1b \
  --dataset data/locomo10.json \
  --keyword_pruning_mode nltk
```

## Retrieval and Reranking Designs

Detailed design notes:

- `docs/retrieval_reranker_design_zh.md`: robust retrieval, content+keyword retrieval, and CrossEncoder reranker data flow.
- `docs/baseline_comparison_zh.md`: baseline matrix and guidance for comparing construction, retrieval, keyword pruning, and reranking variants.

Robust retrieval baseline:

- `retrieve_k` is the final number of memories placed into the answer prompt.
- With reranking disabled (`--rerank-mode off`), robust retrieval uses the existing embedding retriever directly.
- BM25 first-stage retrieval is evaluation-only for robust QA via `--retrieval-mode bm25`; use `--cache-experiment-id` to compare BM25 and embedding on the same memory cache.

CrossEncoder reranker:

- Only applies to robust QA evaluation.
- First-stage embedding retrieval takes `--rerank-top-n` candidates.
- CrossEncoder scores `(original question, candidate memory text)` pairs.
- Final answer context keeps the top `--retrieve-k` reranked memories.
- Reranker runs during QA evaluation and does not change memory cache construction.

Example robust reranker evaluation from an existing two-stage cache:

```bash
uv run python scripts/evaluate_memories.py \
  --experiment-id ollama_llama3.2-1b_none_rerank_k10 \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --qa-mode robust \
  --qa-runs 1 \
  --retrieve-k 10 \
  --retrieval-mode embedding \
  --rerank-mode cross_encoder \
  --rerank-top-n 50 \
  --rerank-batch-size 32 \
  --resume
```

## Experiment Tools

`scripts/` is the experiment, analysis, and visualization tool area.

Useful commands:

```bash
bash scripts/k_sweep_ollama.sh --model llama3.2:1b --k-values "5 10 15"
uv run python scripts/print_k_results.py --results-dir artifacts/results/k_sweep_ollama
```

Preferred two-stage workflow:

```bash
uv run python scripts/build_memories.py \
  --experiment-id ollama_llama3.2-1b_nltk \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --construction-runs 1 \
  --keyword-pruning-mode nltk \
  --max-workers 10 \
  --resume
```

```bash
uv run python scripts/evaluate_memories.py \
  --experiment-id ollama_llama3.2-1b_nltk \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --qa-mode content_keywords \
  --keyword-conditions none,nltk \
  --qa-runs 30 \
  --retrieve-k 10 \
  --resume
```

Or run both stages through the wrapper:

```bash
bash scripts/run_experiment.sh \
  --experiment-id ollama_llama3.2-1b_nltk_k10 \
  --backend ollama \
  --model llama3.2:1b \
  --construction-runs 1 \
  --qa-runs 30 \
  --qa-mode both \
  --keyword-pruning-mode nltk \
  --keyword-conditions none,nltk \
  --retrieve-k 10 \
  --resume
```

Run both stages with robust CrossEncoder reranking:

```bash
bash scripts/run_experiment.sh \
  --experiment-id ollama_llama3.2-1b_none_rerank_k10 \
  --backend ollama \
  --model llama3.2:1b \
  --construction-runs 1 \
  --qa-runs 1 \
  --qa-mode robust \
  --keyword-pruning-mode none \
  --retrieve-k 10 \
  --retrieval-mode embedding \
  --rerank-mode cross_encoder \
  --rerank-top-n 50 \
  --rerank-batch-size 32 \
  --max-workers 10 \
  --resume
```

Two-stage run semantics:

- `construction-runs=1, qa-runs=30`: one memory cache, 30 QA runs. Use this for fixed-cache evaluation.
- `construction-runs=30, qa-runs=1`: 30 independently rebuilt memory caches, one QA run each. Use this for construction variance.
- `construction-runs=30, qa-runs=30`: every cache receives 30 QA runs, for 900 QA outputs total.

Artifact layout:

```text
artifacts/
  caches/<experiment_id>/
    manifest.json
    construction_run_00/
      metadata.json
      memory_cache_sample_0.pkl
      retriever_cache_sample_0.pkl
      retriever_cache_embeddings_sample_0.npy

  results/<experiment_id>/
    manifest.json
    construction_run_00/
      content_keywords/
        qa_run_00/
          none.json
          nltk.json
        summary_across_runs.csv
        summary_across_runs.json
      robust/
        qa_run_00/
          results.json
        summary_across_runs.csv
        summary_across_runs.json

  logs/<experiment_id>/
    build.log
    evaluate_content_keywords.log
```

Legacy root-level generated directories such as `cached_memories*/`, `results_*/`, and `logs/` remain ignored for compatibility, but new experiments should write under `artifacts/`.

The new two-stage scripts are preferred for new fixed-cache and rebuild experiments because construction variance and QA variance are explicit. The older scripts below remain available for compatibility.

Fixed-cache content+keyword pruning analysis:

```bash
uv run python scripts/run_content_keyword_pruning_experiment.py \
  --runs 30 \
  --memory-cache-dir artifacts/caches/cached_memories_robust_ollama_llama3.2:1b \
  --output-dir artifacts/results/content_keyword_pruning/ollama_llama3.2-1b_content_keywords_k10_30runs \
  --backend ollama \
  --model llama3.2:1b \
  --retrieve_k 10 \
  --max-workers 10 \
  --resume
```

End-to-end memory-rebuild content+keyword pruning analysis:

```bash
uv run python scripts/run_content_keyword_rebuild_experiment.py \
  --runs 30 \
  --output-dir artifacts/results/content_keyword_rebuild/ollama_llama3.2-1b_content_keywords_k10_30runs \
  --backend ollama \
  --model llama3.2:1b \
  --retrieve_k 10 \
  --max-workers 10 \
  --resume
```

The fixed-cache experiment isolates keyword transformation on the same memory notes. The rebuild experiment recreates memories per pruning mode and captures system-level effects from construction, retrieval, and evolution.

## Long-Running Runs

Use `tmux` for long experiments. Prefer `--resume` so interrupted runs can continue from completed `construction_run_XX` and `qa_run_XX` outputs.

```bash
tmux new-session -d -s amem_two_stage 'cd /mnt/raid1/novis/a-mem && PYTHONUNBUFFERED=1 bash scripts/run_experiment.sh --experiment-id ollama_llama3.2-1b_nltk_k10 --backend ollama --model llama3.2:1b --construction-runs 1 --qa-runs 30 --qa-mode both --keyword-pruning-mode nltk --keyword-conditions none,nltk --retrieve-k 10 --max-workers 10 --resume'
tmux attach -t amem_two_stage
```

## Verification Before Finishing

For script changes, run at least:

```bash
uv run python -m py_compile scripts/<script>.py
```

For new two-stage entrypoint changes, run focused tests and compile checks:

```bash
uv run python -m py_compile scripts/experiment_common.py scripts/build_memories.py scripts/evaluate_memories.py
uv run python -m pytest tests/test_experiment_common.py tests/test_experiment_entrypoints.py -v
```

For reranker changes, run:

```bash
uv run python -m py_compile src/amem/reranking.py src/amem/memory_layer_robust.py scripts/evaluate_memories.py test_advanced_robust.py
uv run python -m pytest tests/test_reranking.py tests/test_memory_pipeline.py tests/test_reproduction_package.py -v
```

For docs-only changes, run:

```bash
uv run python -m pytest tests/test_reproduction_package.py -v
```

Use `uv run python -m pytest ...` if direct `uv run pytest ...` resolves to a non-project pytest.

Compatibility checks:

```bash
uv run python -m pytest tests/test_ablation.py -v
uv run python -m pytest tests/test_llm_text_parsers.py tests/test_parallel_evaluation.py -v
```

For experiment runners, do a smoke run before launching full jobs. Use limits such as:

```bash
uv run python scripts/build_memories.py \
  --experiment-id smoke_two_stage \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --construction-runs 1 \
  --keyword-pruning-mode nltk \
  --sample-limit 1 \
  --turn-limit 2 \
  --max-workers 1 \
  --resume
```

```bash
uv run python scripts/evaluate_memories.py \
  --experiment-id smoke_two_stage \
  --dataset data/locomo10.json \
  --backend ollama \
  --model llama3.2:1b \
  --qa-mode content_keywords \
  --keyword-conditions none,nltk \
  --qa-runs 1 \
  --sample-limit 1 \
  --qa-limit 1 \
  --max-workers 1 \
  --resume
```

## Coding Notes

- Keep changes small and scoped to the experiment or bug at hand.
- Prefer existing helpers and result formats over inventing new structures.
- Keep content+keyword experiments strict: retrieval documents and answer context should contain only timestamp/content/keywords unless the experiment explicitly says otherwise.
- Avoid committing local notes, papers, output HTML, caches, logs, or partial experiment results.
