#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export MPLCONFIGDIR="$ROOT/.runtime/matplotlib"
mkdir -p "$MPLCONFIGDIR"
"$ROOT/run_collect.sh" --check
"$HOME/miniforge3/envs/dexonomy-isaac/bin/python" "$ROOT/learning/python/check_vision.py"
