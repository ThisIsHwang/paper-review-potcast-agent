#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE="${1:-$(date +%F)}"
TOP_K="${2:-10}"
EXTRA_ARGS=("${@:3}")

if [ -f "$ROOT/scripts/env.sh" ]; then
  # shellcheck source=/dev/null
  source "$ROOT/scripts/env.sh"
else
  echo "env.sh not found; ensure venv is activated and env vars are set."
fi

python "$ROOT/main.py" --date "$DATE" --top-k "$TOP_K" --video-only "${EXTRA_ARGS[@]}"
