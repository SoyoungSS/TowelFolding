#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONPATH
export HF_HOME="$ROOT/.runtime/act_hf" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCH_HOME="$ROOT/.runtime/torch" MPLCONFIGDIR="$ROOT/.runtime/matplotlib"
exec "$ROOT/.runtime/act-env311/bin/python" "$ROOT/training/act/predict.py" "$@"
