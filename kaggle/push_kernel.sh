#!/usr/bin/env bash
# Push code dataset + kernel. Kaggle queues the kernel on a GPU worker.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$DIR/push_code.sh"
kaggle kernels push -p "$DIR"
echo "[push] watch: https://www.kaggle.com/code/sharma37/object-tracking-run"
