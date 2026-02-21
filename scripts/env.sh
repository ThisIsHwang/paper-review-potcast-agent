#!/usr/bin/env bash
# shell options: be strict but stay compatible with zsh sourcing
if [ -n "${BASH_VERSION:-}" ]; then
  set -euo pipefail
else
  set -eu
  set -o pipefail 2>/dev/null || true
fi

# BASH_SOURCE is undefined in zsh; fall back to $0
_THIS_FILE="${BASH_SOURCE[0]-$0}"
ROOT="$(cd "$(dirname "${_THIS_FILE}")/.." && pwd)"
VENV_PATH="$ROOT/.venv"

if [ ! -d "$VENV_PATH" ]; then
  echo "venv not found at $VENV_PATH. Run scripts/setup_venv.sh first."
  exit 1
fi

source "$VENV_PATH/bin/activate"

ENV_FILE="$ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
  echo "Loaded environment from .env"
else
  echo "No .env file found. Create one from .env.example."
fi

export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$ROOT/outputs}"
export PYTHONPATH="$ROOT"

echo "Environment ready. venv activated. PYTHONPATH=$PYTHONPATH"
