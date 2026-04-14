#!/usr/bin/env bash
# Poll status, then download /kaggle/working/ outputs once done.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/kaggle_output"
mkdir -p "$OUT"
kaggle kernels status sharma37/object-tracking-run
kaggle kernels output sharma37/object-tracking-run -p "$OUT"
echo "[pull] downloaded to $OUT"
