#!/usr/bin/env bash
# Upload the tracking/ package as a private Kaggle dataset.
# First run creates it; subsequent runs version it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$ROOT/kaggle/code-dataset-metadata.json" "$STAGE/dataset-metadata.json"

rsync -a \
  --exclude '__pycache__/' --exclude '*.pyc' \
  "$ROOT/tracking" "$STAGE/"

if kaggle datasets status sharma37/object-tracking-code >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "update $(date -u +%FT%TZ)" --dir-mode zip
else
  kaggle datasets create -p "$STAGE" --dir-mode zip
fi
echo "[push_code] synced → sharma37/object-tracking-code"
