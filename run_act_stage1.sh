#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$ROOT/training/act/setup.sh"
unset PYTHONPATH
export HF_HOME="$ROOT/.runtime/act_hf"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export TORCH_HOME="$ROOT/.runtime/torch" MPLCONFIGDIR="$ROOT/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR" "$HF_HOME"
exec "$ROOT/.runtime/act-env311/bin/python" -u "$ROOT/training/act/stage1.py" "$@"
