#!/usr/bin/env bash
# 在本机执行。--check 只读；不带 --check 才在 pi 上启动一次平地推理。
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ! ${1:-} =~ ^[0-9]+$ || ( $# == 2 && ${2:-} != --check ) ]]; then
    echo "用法: bash $0 GPU编号 [--check]" >&2
    exit 2
fi

ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=5 \
    -p 6102 ubuntu@10.14.3.245 bash -s -- "$1" "${2:-run}" <<'REMOTE'
set -euo pipefail
gpu=$1
mode=$2
team=/data/wheelbipe-v14-6-training-20261108
project="$team/project"
runtime=/data/isaac620-migration-20261103/env620/bin/python
checkpoint="$team/logs/rsl_rl/wheelbipe_v14_6_dr_20261108/2026-11-07_18-03-53_flat_seed42_long/model_2000.pt"
params="$(dirname "$checkpoint")/params"
output_root=/data/wheeled-legged/logs

# 锁定已评测的权重及配套配置，不跟随可能更新的 candidate.pt。
cd "$team"
sha256sum --check --quiet <<HASHES
6d321ea1b17ae240719f492bcde40f25e849ff1911399bd9e674023df0cec867  $checkpoint
f35d7d54406cb275983df707cfdbd6a2f1a515d221ed65abb966b1099baa47cc  $params/launch.json
1f114ffa70e6a5fe68effba4e5f3392c2707ab3f357937bcab8eff9ed9bec119  $params/env.yaml
0e9f887c1e9497052eed37b55147de1d0473a7cd2b50d10e7cb832855646a619  $params/agent.yaml
ae341103f249c5bf8d7b5955e58e4f5df397f69f5dc7237d3a2a0ccccbdacdc9  profiles.json
9a1f8c6348abc540589f5da8c22225f5a263862254da09f0b468ed9dcc562613  tools/evaluate_grid.py
bfbc01692f9c7f785ed115a7ec3e67f5340e507396e862859c2acaab62835e65  tools/assess.py
HASHES

# 旧 launch 清单只记录训练时的源码。唯一已知差异是闭链异常采集模块；
# 锁定它的当前版本，并拒绝其他变动，避免无意中测到新的控制逻辑。
"$runtime" -B - "$project" "$params/launch.json" <<'PY'
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

project, launch = map(Path, sys.argv[1:])
saved = json.loads(launch.read_text())
sources = saved["source_sha256"].copy()
sources["source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/physics_failure_diagnostics.py"] = (
    "a94bbbf4474920f90c847b4a13ce5f1876851fc4aa3da10a42d4d583cdc468b4"
)
for name, expected in (sources | saved["own_asset_sha256"]).items():
    if hashlib.sha256((project / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"输入文件变化：{name}")
for name in ("isaacsim", "isaaclab", "torch", "rsl-rl-lib"):
    if version(name) != saved["installed_versions"][name]:
        raise SystemExit(f"共享环境版本变化：{name}")
print("权重、配置、项目源码、模型资产和共享环境版本检查通过。")
PY

# 只选一张确认空闲的卡；UUID 避免 nvidia-smi 与 CUDA 编号不一致。
gpu_state=$(nvidia-smi -i "$gpu" --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader,nounits)
IFS=, read -r uuid util mem <<< "$gpu_state"
compute_apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
if [[ ! $uuid =~ ^GPU-[a-f0-9-]+$ || ! $util =~ ^[[:space:]]*0$ || ! $mem =~ ^[[:space:]]*[0-9]+$ ||
      ${mem//[[:space:]]/} -gt 128 || -n $compute_apps ]]; then
    echo "GPU $gpu 未通过空闲检查：$gpu_state；请选择其他卡或稍后再试。" >&2
    exit 1
fi
[[ -d $output_root && -w $output_root ]]
echo "GPU $gpu 检查时空闲：$gpu_state"
[[ $mode != --check ]] || exit 0

run=$(mktemp -d "$output_root/v14_6_flat_2000_XXXXXX")
export CUDA_VISIBLE_DEVICES="$uuid" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$project/source/agent_tasks:$project/source/agent_world:$project/source/agent_rl:$project"
export OMNI_KIT_ACCEPT_EULA=yes OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WHEELBIPE_STAGE=3 WHEELBIPE_INTEGRATION_SUBSTEPS=2
export XDG_CACHE_HOME="$run/cache" XDG_DATA_HOME="$run/data" XDG_CONFIG_HOME="$run/config"
export WHEELBIPE_DIAGNOSTICS_DIR="$run/diagnostics" TMPDIR="$run/tmp"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME" "$WHEELBIPE_DIAGNOSTICS_DIR" "$TMPDIR" "$run/kit"
cd "$run"
echo "结果目录：$run"

# 原评测种子和指令网格：8 个环境各 6 秒，只有 Actor 推理。
"$runtime" -B "$team/tools/evaluate_grid.py" \
    --profile flat_seed42 --checkpoint "$checkpoint" --output "$run" \
    --seconds 6 --seed 20261108 --device cuda:0 --viz none --headless --livestream 0 \
    "--kit_args=--portable-root=$run/kit" 2>&1 | tee "$run/evaluate.log"
"$runtime" -B "$team/tools/assess.py" "$run" | tee "$run/assess.log"

# 原评分只数 done；同时确认没有超时或异常，避免把短回合判成稳定。
"$runtime" -B - "$run" <<'VERIFY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
summary, metrics, manifest = (
    json.loads((root / f"{name}.json").read_text()) for name in ("summary", "metrics", "manifest")
)
with np.load(root / "trajectory.npz", allow_pickle=False) as data:
    policy = data["policy"]
    shapes_ok = policy.shape[:2] == (300, 8) and data["physics"].shape[:2] == (2400, 8)
    reset = any(policy[:, :, slice(*manifest["policy_fields"][key])].any()
                for key in ("done", "timeout", "abnormal"))
if not (summary["status"] == "PASS" and shapes_ok and metrics["finite"]
        and metrics["effort_limits_pass"] and not reset):
    raise SystemExit(f"平地短测未通过，请查看 {root}")
print(f"平地短测基础检查通过，跟踪误差见 {root / 'metrics.json'}")
VERIFY
REMOTE
