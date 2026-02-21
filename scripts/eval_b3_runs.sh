#!/usr/bin/env bash
# Evaluate all b3 ablation prediction files under a run directory.
# Usage:
#   scripts/eval_b3_runs.sh runs_ablation_gpt4o_head100 data/wmt24pp/en-ko_KR.head100.jsonl
# Defaults:
#   RUN_DIR=runs_ablation_gpt4o_head100
#   REF=data/wmt24pp/en-ko_KR.head100.jsonl

set -euo pipefail

RUN_DIR="${1:-runs_ablation_gpt4o_head100}"
REF="${2:-data/wmt24pp/en-ko_KR.head100.jsonl}"
PY_BIN="./bleu/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  exit 1
fi

shopt -s nullglob
files=("$RUN_DIR"/*/predictions.jsonl)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No predictions.jsonl files found under $RUN_DIR" >&2
  exit 1
fi

for f in "${files[@]}"; do
  echo "=== $f ==="
  "$PY_BIN" b3.py eval \
    --input "$f" \
    --metric bleu \
    --exclude_bad_source \
    --ref "$REF"
  echo
done
