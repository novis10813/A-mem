#!/usr/bin/env bash
set -euo pipefail

# Find a good retrieval-k for A-MEM with the Ollama backend.
#
# Usage:
#   bash scripts/k_sweep_ollama.sh [OPTIONS]
#
# Options:
#   --model    <name>      Ollama model tag (default: llama3.2:1b)
#   --dataset  <path>      Dataset JSON path (default: data/locomo10.json)
#   --k-values <list>      Space-separated k list (default: "5 10 15")
#   --outdir   <path>      Output directory (default: results_k_sweep_ollama)
#   --log-dir  <path>      Log directory (default: logs/k_sweep_ollama)
#   --help                 Show this message and exit
#
# Example:
#   bash scripts/k_sweep_ollama.sh --model llama3.2:1b --k-values "5 10 15"

MODEL="llama3.2:1b"
DATASET="data/locomo10.json"
K_VALUES=(5 10 15)
OUTDIR="results_k_sweep_ollama"
LOGDIR="logs/k_sweep_ollama"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --k-values) IFS=' ' read -r -a K_VALUES <<< "$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --log-dir) LOGDIR="$2"; shift 2 ;;
        --help)
            sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
    esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ ! -f "test_advanced_robust.py" ]]; then
    echo "[ERROR] Run this script from the A-MEM repo root." >&2
    echo "        Example: bash scripts/k_sweep_ollama.sh --model llama3.2:1b" >&2
    exit 1
fi

if [[ ! -f "$DATASET" ]]; then
    echo "[ERROR] Dataset not found: $DATASET" >&2
    exit 1
fi

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[ERROR] Ollama does not appear to be running at http://localhost:11434" >&2
    echo "        Start it with: ollama serve" >&2
    exit 1
fi

MODEL_SHORT=$(echo "$MODEL" | tr ':/' '-')
mkdir -p "$OUTDIR" "$LOGDIR"

log "A-MEM Ollama k-sweep"
log "model=$MODEL dataset=$DATASET k_values=${K_VALUES[*]} outdir=$OUTDIR logdir=$LOGDIR"

FAILED_KS=()
for k in "${K_VALUES[@]}"; do
    OUTFILE="${OUTDIR}/${MODEL_SHORT}_k${k}.json"
    LOGFILE="${LOGDIR}/${MODEL_SHORT}_k${k}.log"

    if [[ -f "$OUTFILE" ]]; then
        log "[SKIP] k=$k output exists: $OUTFILE"
        continue
    fi

    log "[RUN] k=$k -> $OUTFILE"
    if uv run test_advanced_robust.py \
        --backend ollama \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output "$OUTFILE" \
        --retrieve_k "$k" \
        >"$LOGFILE" 2>&1; then
        log "[OK] k=$k"
    else
        log "[WARN] k=$k failed; see $LOGFILE"
        FAILED_KS+=("$k")
    fi
done

log "Summary"
uv run python scripts/print_k_results.py --results-dir "$OUTDIR" --model "$MODEL_SHORT" || true

if (( ${#FAILED_KS[@]} > 0 )); then
    echo "[ERROR] Failed k values: ${FAILED_KS[*]}" >&2
    exit 1
fi
