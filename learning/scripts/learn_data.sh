#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
"$ROOT/run_collect.sh" --check
"$HOME/miniforge3/envs/dexonomy-isaac/bin/python" "$ROOT/learning/python/view_data.py" "$@"
echo '将上面打印的 index.html 用浏览器打开，即可开始学习。'
