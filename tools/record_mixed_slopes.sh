#!/usr/bin/env bash
set -euo pipefail

# Record a frozen actor on the mixed terrain without changing the actor or its
# physical settings. Each column is one full-difficulty evaluation tile:
# 0=flat, 4=18% upslope, 8=18% downslope.
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu=${1:-}
checkpoint=${2:-}
output=${3:-}
steps=${4:-800}

if [[ ! "$gpu" =~ ^[0-9]+$ || -z "$checkpoint" || -z "$output" ]]; then
    echo "Usage: bash $0 GPU_INDEX CHECKPOINT OUTPUT_DIR [STEPS]" >&2
    exit 2
fi
if [[ ! "$steps" =~ ^[1-9][0-9]*$ ]]; then
    echo "STEPS must be positive." >&2
    exit 2
fi
checkpoint=$(realpath "$checkpoint")
output=$(realpath -m "$output")
test -f "$checkpoint" || {
    echo "Checkpoint not found: $checkpoint" >&2
    exit 2
}

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

mkdir -p "$output"
export PYTHONPATH="$root/project/source/agent_tasks:$root/project/source/agent_world:$root/project/source/agent_rl:$root/project"
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export HEADLESS=1 OMNI_KIT_ACCEPT_EULA=yes
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$output/cache"
export XDG_DATA_HOME="$output/data"
export XDG_CONFIG_HOME="$output/config"
export TMPDIR="$output/tmp"
export WHEELBIPE_DIAGNOSTICS_DIR="$output/diagnostics"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$TMPDIR" "$WHEELBIPE_DIAGNOSTICS_DIR" "$output/kit"

printf 'CHECKPOINT %s\nGPU %s %s\nOUTPUT %s\nSTEPS %s\n' \
    "$checkpoint" "$gpu" "$gpu_uuid" "$output" "$steps" | tee "$output/run.txt"

for spec in flat:0 up18:4 down18:8; do
    name=${spec%%:*}
    column=${spec##*:}
    mkdir -p "$output/$name"
    "$python_bin" -B "$root/tools/record.py" \
        --checkpoint "$checkpoint" \
        --output "$output/$name" \
        --steps "$steps" \
        --mode policy \
        --fixed-command 0.5 0 \
        --record \
        --seed 20261111 \
        --profile mixed_slopes_20261111 \
        --allow-profile-override \
        --terrain-column "$column" \
        --terrain-level 0 \
        --device cuda:0 \
        "--kit_args=--portable-root=$output/kit" \
        2>&1 | tee "$output/$name/console.log"
done

printf 'Recording complete: %s\n' "$output"
