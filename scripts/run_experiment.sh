#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_experiment.sh --experiment-id ID [options]

Options:
  --dataset PATH
  --cache-root PATH
  --results-root PATH
  --log-root PATH
  --backend NAME
  --model NAME
  --construction-runs N
  --qa-runs N
  --qa-mode content_keywords|robust|both
  --keyword-pruning-mode none|simple|nltk
  --keyword-conditions LIST
  --retrieve-k N
  --retrieval-mode embedding|bm25
  --rerank-mode off|cross_encoder
  --rerank-model NAME
  --rerank-top-n N
  --rerank-batch-size N
  --temperature-c5 FLOAT
  --ratio FLOAT
  --sample-limit N
  --turn-limit N
  --qa-limit N
  --max-workers N
  --resume
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

experiment_id=""
dataset="data/locomo10.json"
cache_root="artifacts/caches"
results_root="artifacts/results"
log_root="artifacts/logs"
backend="ollama"
model="llama3.2:1b"
construction_runs="1"
qa_runs="1"
qa_mode="content_keywords"
keyword_pruning_mode="nltk"
keyword_conditions="none,nltk"
retrieve_k="10"
retrieval_mode="embedding"
rerank_mode="off"
rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2"
rerank_top_n="50"
rerank_batch_size="32"
temperature_c5="0.5"
ratio="1.0"
sample_limit=""
turn_limit=""
qa_limit=""
max_workers="1"
resume=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-id) experiment_id="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --cache-root) cache_root="$2"; shift 2 ;;
    --results-root) results_root="$2"; shift 2 ;;
    --log-root) log_root="$2"; shift 2 ;;
    --backend) backend="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --construction-runs) construction_runs="$2"; shift 2 ;;
    --qa-runs) qa_runs="$2"; shift 2 ;;
    --qa-mode) qa_mode="$2"; shift 2 ;;
    --keyword-pruning-mode) keyword_pruning_mode="$2"; shift 2 ;;
    --keyword-conditions) keyword_conditions="$2"; shift 2 ;;
    --retrieve-k|--retrieve_k) retrieve_k="$2"; shift 2 ;;
    --retrieval-mode) retrieval_mode="$2"; shift 2 ;;
    --rerank-mode) rerank_mode="$2"; shift 2 ;;
    --rerank-model) rerank_model="$2"; shift 2 ;;
    --rerank-top-n) rerank_top_n="$2"; shift 2 ;;
    --rerank-batch-size) rerank_batch_size="$2"; shift 2 ;;
    --temperature-c5|--temperature_c5) temperature_c5="$2"; shift 2 ;;
    --ratio) ratio="$2"; shift 2 ;;
    --sample-limit) sample_limit="$2"; shift 2 ;;
    --turn-limit) turn_limit="$2"; shift 2 ;;
    --qa-limit) qa_limit="$2"; shift 2 ;;
    --max-workers) max_workers="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$experiment_id" ]]; then
  echo "--experiment-id is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$dataset" ]]; then
  echo "Dataset not found: $dataset" >&2
  exit 1
fi

if [[ "$backend" == "ollama" ]]; then
  curl -sf http://localhost:11434/api/tags >/dev/null
fi

log_dir="${log_root}/${experiment_id}"
mkdir -p "$log_dir"

build_cmd=(
  uv run python scripts/build_memories.py
  --experiment-id "$experiment_id"
  --dataset "$dataset"
  --cache-root "$cache_root"
  --backend "$backend"
  --model "$model"
  --construction-runs "$construction_runs"
  --keyword-pruning-mode "$keyword_pruning_mode"
  --ratio "$ratio"
  --max-workers "$max_workers"
)
eval_cmd=(
  uv run python scripts/evaluate_memories.py
  --experiment-id "$experiment_id"
  --dataset "$dataset"
  --cache-root "$cache_root"
  --results-root "$results_root"
  --backend "$backend"
  --model "$model"
  --construction-runs "$construction_runs"
  --qa-runs "$qa_runs"
  --qa-mode "$qa_mode"
  --keyword-conditions "$keyword_conditions"
  --retrieve-k "$retrieve_k"
  --retrieval-mode "$retrieval_mode"
  --rerank-mode "$rerank_mode"
  --rerank-model "$rerank_model"
  --rerank-top-n "$rerank_top_n"
  --rerank-batch-size "$rerank_batch_size"
  --temperature-c5 "$temperature_c5"
  --ratio "$ratio"
  --max-workers "$max_workers"
)

if [[ -n "$sample_limit" ]]; then
  build_cmd+=(--sample-limit "$sample_limit")
  eval_cmd+=(--sample-limit "$sample_limit")
fi
if [[ -n "$turn_limit" ]]; then
  build_cmd+=(--turn-limit "$turn_limit")
fi
if [[ -n "$qa_limit" ]]; then
  eval_cmd+=(--qa-limit "$qa_limit")
fi
if [[ "$resume" -eq 1 ]]; then
  build_cmd+=(--resume)
  eval_cmd+=(--resume)
fi

"${build_cmd[@]}" 2>&1 | tee "$log_dir/build.log"
"${eval_cmd[@]}" 2>&1 | tee "$log_dir/evaluate_${qa_mode}.log"
