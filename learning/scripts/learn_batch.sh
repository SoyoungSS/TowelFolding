#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
"$ROOT/run_collect.sh" --check
"$HOME/miniforge3/envs/dexonomy-isaac/bin/python" "$ROOT/learning/python/build_batch.py" "$@"
