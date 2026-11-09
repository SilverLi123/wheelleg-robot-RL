#!/usr/bin/env bash
# Cloud desktop entry; physics validation and measurements are recorded in audit/.
# The platform's conda/Isaac setup scripts read optional unset shell variables.
set -eo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
interactive=0
command_args=(--standstill)
forwarded=()
for arg in "$@"; do
    case "$arg" in
        --interactive) interactive=1 ;;
        --random-commands) command_args=() ;;
        --keyboard) command_args=(); forwarded+=("$arg") ;;
        *) forwarded+=("$arg") ;;
    esac
done
source /root/gpufree-data/wheelbipe-deploy/activate.sh
export WHEELBIPE_PYTHON=/root/gpufree-data/envs/wheelbipe_sim51/bin/python
export WHEELBIPE_RUN_ROOT="${WHEELBIPE_DESKTOP_RUN_ROOT:-/root/gpufree-data/own_robot_runs_20261028/desktop_solver64}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
checkpoint="${WHEELBIPE_CHECKPOINT:-/root/gpufree-data/own_robot_runs_20261028/train3000_512/run/logs/rsl_rl/wheelbipe_own_flat/2026-10-28_11-57-13_authorized_3000/model_2999.pt}"
if [[ ! -f "$checkpoint" ]]; then echo "Missing checkpoint: $checkpoint" >&2; exit 2; fi
device_args=(--device cuda:0 --follow_camera)
if [[ "$interactive" == 1 ]]; then
    device_args=(--device cpu --disable_fabric --kit_args
        "--/physics/mouseInteractionEnabled=true --/physics/mouseGrab=true --/physics/forceGrab=true --/physics/updateToUsd=true --/physics/updateVelocitiesToUsd=true --/persistent/physics/visualizationDisplayJoints=true")
fi
cd -- "$PROJECT_ROOT"
exec bash scripts/run_project.sh scripts/rsl_rl/play.py \
    --task Robotics-Wheelbipe-Own-Flat-Play-v0 --num_envs 1 \
    --checkpoint "$checkpoint" --seed 42 --real-time \
    --solver_position_iterations 64 --solver_velocity_iterations 8 \
    "${device_args[@]}" "${command_args[@]}" "${forwarded[@]}"
