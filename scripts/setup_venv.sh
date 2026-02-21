#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

echo "Creating venv at $ROOT/.venv using ${PYTHON_BIN}..."
if [ ! -d "$ROOT/.venv" ]; then
  "${PYTHON_BIN}" -m venv "$ROOT/.venv"
else
  echo "venv already exists, reusing."
fi

source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements.txt"

echo "Done. Activate with: source $ROOT/.venv/bin/activate"
