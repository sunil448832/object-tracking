#!/usr/bin/env bash
# Bundle tracking/ + kaggle.json into colab/bundle.zip for single upload.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/colab/bundle.zip"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp -r "$ROOT/tracking" "$STAGE/tracking"
find "$STAGE/tracking" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
cp "$HOME/.kaggle/kaggle.json" "$STAGE/kaggle.json"

rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" tracking kaggle.json)
echo "[pack] → $OUT"
