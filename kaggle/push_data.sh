#!/usr/bin/env bash
# Upload ../data as the Kaggle dataset referenced in kernel-metadata.json.
# First run: creates it. Subsequent runs: new version.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$ROOT/kaggle/dataset-metadata.json" "$STAGE/"
rsync -a "$ROOT/data/" "$STAGE/data/"

if kaggle datasets status sharma37/object-tracking-data >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "update $(date -u +%FT%TZ)" --dir-mode zip
else
  kaggle datasets create -p "$STAGE" --dir-mode zip
fi
