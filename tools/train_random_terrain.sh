#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode=${1:-}
gpu=${2:-}
if [[ ! "$mode" =~ ^(smoke|train)$ || ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 {smoke|train} GPU_INDEX" >&2
    exit 2
fi

python_bin=/data/isaac620-migration-20261103/env620/bin/python
default_checkpoint=/data/wheeled-legged/projects/v14_6_downhill_pilot_8KMxPj/runs/train_udMln2/logs/rsl_rl/wheelbipe_v14_6_dr_20261108/2026-11-11_06-29-46_downhill_pilot/model_50.pt
checkpoint=${WHEELBIPE_INIT_CHECKPOINT:-$default_checkpoint}
checkpoint_sha=${WHEELBIPE_INIT_SHA:-}

test -x "$python_bin"
test -f "$checkpoint"
if [[ -n "$checkpoint_sha" ]]; then
    printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum -c
fi

read -r gpu_uuid util memory < <(
    nvidia-smi -i "$gpu" \
        --query-gpu=uuid,utilization.gpu,memory.used \
        --format=csv,noheader,nounits | tr -d ','
)
if [[ -z "$gpu_uuid" || "$util" -ne 0 || "$memory" -gt 20 ]]; then
    echo "GPU $gpu is busy: $gpu_uuid, ${util}%, ${memory} MiB" >&2
    exit 3
fi

mkdir -p "$root/runs"
lock_name=${WHEELBIPE_TRAIN_LOCK:-random_terrain_train}
exec 9>"$root/runs/${lock_name}.lock"
flock -n 9 || {
    echo "A random-terrain training command is already running." >&2
    exit 3
}

out=$(mktemp -d "$root/runs/${mode}_XXXXXX")
mkdir -p "$out/cache" "$out/data" "$out/config" "$out/tmp" "$out/diagnostics" "$out/kit"

export PYTHONPATH="$root/project/source/agent_tasks:$root/project/source/agent_world:$root/project/source/agent_rl:$root/project"
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export HEADLESS=1 OMNI_KIT_ACCEPT_EULA=yes
export WHEELBIPE_EXPERIMENT=${WHEELBIPE_EXPERIMENT:-random_terrain_20261112}
export WHEELBIPE_STAGE=2
export WHEELBIPE_INTEGRATION_SUBSTEPS=2
export WHEELBIPE_SAVE_INTERVAL=${WHEELBIPE_SAVE_INTERVAL:-10}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$out/cache"
export XDG_DATA_HOME="$out/data"
export XDG_CONFIG_HOME="$out/config"
export TMPDIR="$out/tmp"
export WHEELBIPE_DIAGNOSTICS_DIR="$out/diagnostics"

num_envs=${WHEELBIPE_NUM_ENVS:-4096}
iterations=${WHEELBIPE_ITERATIONS:-120}
run_name=${WHEELBIPE_RUN_NAME:-random_terrain_4096_${iterations}}
seed=${WHEELBIPE_SEED:-20261112}
finetune_noise_std=${WHEELBIPE_FINETUNE_NOISE_STD:-0.18}
if [[ "$mode" == smoke ]]; then
    num_envs=${WHEELBIPE_SMOKE_ENVS:-16}
    iterations=1
    run_name=random_terrain_smoke
fi

printf 'OUTPUT %s\n' "$out"
printf 'GPU %s %s\n' "$gpu" "$gpu_uuid"
printf 'ENVIRONMENTS %s ITERATIONS %s\n' "$num_envs" "$iterations"
cd "$out"
"$python_bin" -B "$root/project/scripts/rsl_rl/train.py" \
    --task Robotics-Wheelbipe-Own-Explore-v0 \
    --num_envs "$num_envs" \
    --max_iterations "$iterations" \
    --seed "$seed" \
    --checkpoint "$checkpoint" \
    --checkpoint_architecture \
    --finetune_noise_std "$finetune_noise_std" \
    --audit_updates \
    --logger tensorboard \
    --run_name "$run_name" \
    --device cuda:0 \
    --livestream 0 \
    "--kit_args=--portable-root=$out/kit" 2>&1 | tee "$out/console.log"
