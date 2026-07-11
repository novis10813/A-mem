#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CONFIG.yaml" >&2
  exit 64
fi

uv run python -m memorybench validate --config "$1"
uv run python -m memorybench run --config "$1"
