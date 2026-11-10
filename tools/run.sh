#!/usr/bin/env bash
set -euo pipefail
# 运行当前检出的源码，日志和缓存也留在这个检出目录。
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME=${WHEELBIPE_PYTHON:-/data/isaac620-migration-20261103/env620/bin/python}
if [[ $# -eq 0 || ! -x "$RUNTIME" ]]; then
    echo "用法：bash tools/run.sh /path/to/entry.py [args...]；请确认 WHEELBIPE_PYTHON 可执行。" >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/project/source/agent_tasks:$ROOT/project/source/agent_world:$ROOT/project/source/agent_rl:$ROOT/project"
export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${WHEELBIPE_GPU:-0}" HEADLESS=1
export XDG_CACHE_HOME="$ROOT/runs/cache" XDG_DATA_HOME="$ROOT/runs/data" XDG_CONFIG_HOME="$ROOT/runs/config"
export WHEELBIPE_DIAGNOSTICS_DIR="$ROOT/diagnostics/${WHEELBIPE_EXPERIMENT:-probe}/${WHEELBIPE_PHASE:-main}"
mkdir -p "$WHEELBIPE_DIAGNOSTICS_DIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME"
cd "$ROOT"
exec taskset -c "${WHEELBIPE_CPUSET:-0-7}" "$RUNTIME" "$@"
