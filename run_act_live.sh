#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$ROOT/training/act/setup.sh"
unset PYTHONPATH
cd "$ROOT"
echo '启动ACT实时闭环：专家关闭，优先使用已通过重复闭环验收的模型和执行参数。'
echo '结束后查看ACT EVALUATION COMPLETE和evaluation.json；任务失败不等于程序异常。'
exec "$HOME/miniforge3/envs/dexonomy-isaac/bin/python" -u "$ROOT/scripts/isaac/act_live.py" "$@"
