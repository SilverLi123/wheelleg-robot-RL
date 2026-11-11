#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu=${1:-}
checkpoint=${2:-}
if [[ ! "$gpu" =~ ^[0-9]+$ || -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "Usage: bash $0 GPU_INDEX CHECKPOINT [STEPS] [SEED]" >&2
    exit 2
fi
checkpoint=$(realpath "$checkpoint")
steps=${3:-600}
seed=${4:-20261111}
if [[ ! "$steps" =~ ^[1-9][0-9]*$ || ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "STEPS must be positive and SEED integer." >&2
    exit 2
fi

python_bin=/data/isaac620-migration-20261103/env620/bin/python
read -r gpu_uuid util memory < <(
    nvidia-smi -i "$gpu" \
        --query-gpu=uuid,utilization.gpu,memory.used \
        --format=csv,noheader,nounits | tr -d ','
)
if [[ -z "$gpu_uuid" || "$util" -ne 0 || "$memory" -gt 20 ]]; then
    echo "GPU $gpu is busy: $gpu_uuid, ${util}%, ${memory} MiB" >&2
    exit 3
fi

out=$(mktemp -d "$root/runs/eval_mixed_${seed}_XXXXXX")
mkdir -p "$out/cache" "$out/data" "$out/config" "$out/tmp" "$out/diagnostics" "$out/kit"
export PYTHONPATH="$root/project/source/agent_tasks:$root/project/source/agent_world:$root/project/source/agent_rl:$root/project"
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export HEADLESS=1 OMNI_KIT_ACCEPT_EULA=yes
export WHEELBIPE_EXPERIMENT=${WHEELBIPE_EXPERIMENT:-mixed_slopes_20261111}
export WHEELBIPE_STAGE=2
export WHEELBIPE_INTEGRATION_SUBSTEPS=2
export WHEELBIPE_SAVE_INTERVAL=10
export WHEELBIPE_EVAL_STEPS="$steps"
export WHEELBIPE_EVAL_SEED="$seed"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$out/cache"
export XDG_DATA_HOME="$out/data"
export XDG_CONFIG_HOME="$out/config"
export TMPDIR="$out/tmp"
export WHEELBIPE_DIAGNOSTICS_DIR="$out/diagnostics"

printf 'OUTPUT %s\n' "$out"
printf 'GPU %s %s\n' "$gpu" "$gpu_uuid"
cd "$out"
"$python_bin" -B "$root/tools/mixed_slopes_eval.py" \
    "$checkpoint" \
    "$out/report.json" 2>&1 | tee "$out/console.log"
