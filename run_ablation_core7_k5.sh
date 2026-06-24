#!/usr/bin/env bash
set -euo pipefail

# Run the fixed-memory embedding-field ablation with retrieve_k=5.
# This script assumes Ollama is already running and has llama3.2:latest installed.

cd "$(dirname "$0")"

PYTHON_CMD="${PYTHON_CMD:-uv run python}"
DATASET="${DATASET:-data/locomo10.json}"
BACKEND="${BACKEND:-ollama}"
MODEL="${MODEL:-llama3.2:latest}"
MEMORY_CACHE_DIR="${MEMORY_CACHE_DIR:-cached_memories_robust_ollama_llama3.2:latest-v3}"
OUTPUT_DIR="${OUTPUT_DIR:-results_ablation/ollama_llama3.2-latest-v3_core7_k5}"
RETRIEVE_K="${RETRIEVE_K:-5}"
TEMPERATURE_C5="${TEMPERATURE_C5:-0.5}"
VARIANTS="${VARIANTS:-core7}"

if [[ ! -d "$MEMORY_CACHE_DIR" ]]; then
  echo "Memory cache directory not found: $MEMORY_CACHE_DIR" >&2
  echo "Run test_advanced_robust.py first to create cached memories." >&2
  exit 1
fi

$PYTHON_CMD ablation.py \
  --dataset "$DATASET" \
  --backend "$BACKEND" \
  --model "$MODEL" \
  --memory-cache-dir "$MEMORY_CACHE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --retrieve_k "$RETRIEVE_K" \
  --temperature_c5 "$TEMPERATURE_C5" \
  --variants "$VARIANTS"
