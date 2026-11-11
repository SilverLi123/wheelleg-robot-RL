#!/usr/bin/env bash
set -euo pipefail

# Record a frozen actor on one representative tile from each slope direction,
# or on all nine full-difficulty evaluation tiles without changing the actor or
# its physical settings.
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu=${1:-}
checkpoint=${2:-}
output=${3:-}
steps=${4:-800}
scope=${5:-representative}

if [[ ! "$gpu" =~ ^[0-9]+$ || -z "$checkpoint" || -z "$output"
      || ! "$scope" =~ ^(representative|all)$ ]]; then
    echo "Usage: bash $0 GPU_INDEX CHECKPOINT OUTPUT_DIR [STEPS] [representative|all]" >&2
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
# record.py must load the exact observation, action and launch metadata written
# beside the original training checkpoint. Check this before spending time on
# Isaac Sim startup so a copied checkpoint without its params/ directory fails
# immediately and cannot be mistaken for a policy or rendering problem.
params=$(dirname "$checkpoint")/params
for name in launch.json env.yaml agent.yaml; do
    if [[ ! -f "$params/$name" ]]; then
        echo "Checkpoint metadata missing: $params/$name" >&2
        echo "Use the original training checkpoint path, not a copied .pt file." >&2
        exit 2
    fi
done

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
printf 'SCOPE %s\n' "$scope" | tee -a "$output/run.txt"

if [[ "$scope" == all ]]; then
    cases=(
        flat:0 up06:1 up10:2 up14:3 up18:4
        down06:5 down10:6 down14:7 down18:8
    )
else
    cases=(flat:0 up18:4 down18:8)
fi

for spec in "${cases[@]}"; do
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
        --mixed-eval-terrain \
        --device cuda:0 \
        "--kit_args=--portable-root=$output/kit" \
        2>&1 | tee "$output/$name/console.log"
done

if [[ "$scope" == all ]]; then
    "$python_bin" -B "$root/tools/assemble_mixed_slopes_video.py" \
        "$output" \
        2>&1 | tee "$output/assemble.log"
fi

printf 'Recording complete: %s\n' "$output"
