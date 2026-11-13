#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.runtime/act-env311/bin/python"
if [[ ! -x "$PY" ]]; then
  "${ACT_BASE_PYTHON:-$HOME/miniforge3/envs/dexonomy-isaac/bin/python}" -m venv "$ROOT/.runtime/act-env311"
fi
unset PYTHONPATH
"$PY" -c 'import sys; assert sys.version_info[:2] == (3,11), "ACT环境需要Python 3.11"'
if [[ -f "$ROOT/.runtime/act-env311/READY" ]]; then
  "$PY" - "$ROOT/environments/act-requirements.txt" <<'PYCODE'
import sys
from importlib.metadata import version
for line in open(sys.argv[1]):
    line=line.strip()
    if not line or line.startswith('#'):continue
    name,expected=line.split('==')
    assert version(name)==expected, f"{name}版本改变：{version(name)} != {expected}"
import torch,torchvision,lerobot,av
print("ACT environment ready: LeRobot",version("lerobot"),"PyTorch",torch.__version__)
PYCODE
  exit 0
fi
mkdir -p "$ROOT/.runtime/act_setup"
WHEEL_ARGS=()
if [[ -d "$ROOT/.runtime/act_setup/wheels" ]]; then WHEEL_ARGS+=(--find-links "$ROOT/.runtime/act_setup/wheels"); fi
if [[ -n "${ACT_WHEELHOUSE:-}" ]]; then WHEEL_ARGS+=(--find-links "$ACT_WHEELHOUSE"); fi
"$PY" -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 "${WHEEL_ARGS[@]}" --no-deps --timeout 120 --index-url https://download.pytorch.org/whl/cu128 --disable-pip-version-check
"$PY" -m pip install -r "$ROOT/environments/act-requirements.txt" "${WHEEL_ARGS[@]}" --timeout 120 --disable-pip-version-check
"$PY" -m pip check
"$PY" -m pip freeze > "$ROOT/.runtime/act_setup/installed-freeze.txt"
touch "$ROOT/.runtime/act-env311/READY"
