#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export MPLCONFIGDIR="$ROOT/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR" "$ROOT/.runtime/torch/hub/checkpoints"
WEIGHTS="$ROOT/.runtime/torch/hub/checkpoints/resnet18-f37072fd.pth"
if [[ ! -f "$WEIGHTS" ]]; then
  echo '首次运行需要下载约45MB的官方ResNet-18预训练权重。'
  curl -L --fail --output "$WEIGHTS.part" https://download.pytorch.org/models/resnet18-f37072fd.pth
  mv "$WEIGHTS.part" "$WEIGHTS"
fi
"$HOME/miniforge3/envs/dexonomy-isaac/bin/python" "$ROOT/learning/python/vision_features.py" "$@"
