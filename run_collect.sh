#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PYTHON="$HOME/miniforge3/envs/dexonomy-isaac/bin/python"
NEWTON_PYTHON="$HOME/miniforge3/envs/towel-newton/bin/python"
for interpreter in "$ISAAC_PYTHON" "$NEWTON_PYTHON"; do
  if [[ ! -x "$interpreter" ]]; then echo "找不到环境：$interpreter" >&2; exit 1; fi
done
# Runtime dependencies must survive cleanup of generated datasets in outputs/.
DEPS="$ROOT/.runtime/collector_deps"
export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"
if ! "$ISAAC_PYTHON" -c 'import pyarrow; assert pyarrow.__version__ == "19.0.1"' >/dev/null 2>&1; then
  echo "正在准备采集依赖 PyArrow 19.0.1（首次运行需要联网，不修改 Isaac 环境）……"
  "$ISAAC_PYTHON" -m pip install pyarrow==19.0.1 --no-deps --upgrade --target "$DEPS" --disable-pip-version-check || {
    echo "采集依赖安装失败，请检查网络连接后重新运行 ./run_collect.sh。" >&2
    exit 1
  }
fi
"$ISAAC_PYTHON" -c 'import pyarrow, cv2; print("采集依赖检查通过：PyArrow " + pyarrow.__version__)'
if [[ "${1:-}" == "--check" ]]; then exit 0; fi
if [[ "${1:-}" == "--episodes" ]]; then
  if [[ $# -lt 2 ]]; then echo '用法：./run_collect.sh --episodes N [--offset-mm 3] [--output DIR]' >&2; exit 2; fi
  count="$2"
  shift 2
  cd "$ROOT"
  exec "$ISAAC_PYTHON" -u "$ROOT/scripts/isaac/collect_batch.py" --episodes "$count" \
    --newton-python "$NEWTON_PYTHON" --output "$ROOT/outputs/collect/batch_$(date +%Y%m%d_%H%M%S)_$$" "$@"
fi
RECOVERY_ARGS=()
if [[ $# -ne 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--recovery-plan" || ! -f "$2" ]]; then
    echo '用法：./run_collect.sh [--check | --episodes N | --recovery-plan /path/to/plan.npz]' >&2
    exit 2
  fi
  RECOVERY_ARGS=(--recovery-plan "$(realpath -- "$2")" --exit-on-done)
fi
OUTPUT="$ROOT/outputs/collect/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUTPUT"
echo "启动实时窗口并采集一次固定工况对折。输出：$OUTPUT"
echo "成功后出现 DATASET SAVED；窗口可手动关闭。提前关闭会留下未完成数据。"
cd "$ROOT"
"$ISAAC_PYTHON" -u scripts/isaac/live_bridge.py --collect --fps 30 --seed 0 --newton-python "$NEWTON_PYTHON" --output "$OUTPUT" "${RECOVERY_ARGS[@]}" 2>&1 | tee "$OUTPUT/console.log"
[[ -f "$OUTPUT/dataset/SUCCESS.json" ]] || { echo "本次采集未成功，请查看 $OUTPUT/console.log" >&2; exit 1; }
echo "采集成功：$OUTPUT/dataset"
