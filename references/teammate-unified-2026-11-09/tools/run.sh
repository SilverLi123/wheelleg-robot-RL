#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/wheelbipe-unified-parallel-20261108
RUNTIME=/data/isaac620-migration-20261103
export PYTHONPATH="$ROOT/project/source/agent_tasks:$ROOT/project/source/agent_world:$ROOT/project/source/agent_rl:$ROOT/project"
export OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${WHEELBIPE_GPU:-0}" HEADLESS=1
export XDG_CACHE_HOME="$RUNTIME/cache/620" XDG_DATA_HOME="$RUNTIME/cache/data620" XDG_CONFIG_HOME="$RUNTIME/cache/config620"
export WHEELBIPE_DIAGNOSTICS_DIR="$ROOT/diagnostics/${WHEELBIPE_EXPERIMENT:-probe}/${WHEELBIPE_PHASE:-main}"
mkdir -p "$WHEELBIPE_DIAGNOSTICS_DIR"
cd "$ROOT"
exec taskset -c "${WHEELBIPE_CPUSET:-0-7}" "$RUNTIME/env620/bin/python" "$@"
