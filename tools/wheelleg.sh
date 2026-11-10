#!/usr/bin/env bash
# 日常入口：只检查输入并转发，不修改底层脚本或实验参数。
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat <<'HELP'
用法：
  bash tools/wheelleg.sh [--dry-run] baseline GPU编号 [--check]  # Mac
  bash tools/wheelleg.sh [--dry-run] train [train.py 参数]       # pi
  bash tools/wheelleg.sh [--dry-run] eval [validate.py 参数]     # pi
  bash tools/wheelleg.sh [--dry-run] view video [record.py 参数] # pi
  bash tools/wheelleg.sh view interactive  # Sim 6.2 统一入口暂不可用
--dry-run 只显示转发命令，不连接服务器、不检查输入、不启动仿真。
HELP
}

dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
mode=${1:---help}
case "$mode" in -h|--help|help) usage; exit 0 ;; esac
shift
case "$mode" in
    baseline) command=(bash "$root/tools/reproduce_v14_6_flat.sh") ;;
    train) command=(bash "$root/tools/run.sh" "$root/project/scripts/rsl_rl/train.py") ;;
    eval) command=(bash "$root/tools/run.sh" "$root/tools/validate.py") ;;
    view)
        case "${1:-}" in
            video) shift; mode=video; command=(bash "$root/tools/run.sh" "$root/tools/record.py" --record) ;;
            interactive)
                echo "当前 Sim 6.2 统一入口暂不支持交互回放：原 Own 回放入口要求 Sim 5.1；未启动仿真。可使用 view video。" >&2
                exit 2 ;;
            *) usage >&2; exit 2 ;;
        esac ;;
    *) usage >&2; exit 2 ;;
esac
if [[ $dry_run == 1 ]]; then
    printf ' %q' "${command[@]}" "$@"; printf '\n'
    exit 0
fi

if [[ $mode != baseline ]]; then
    runtime=${WHEELBIPE_PYTHON:-/data/isaac620-migration-20261103/env620/bin/python}
    if [[ ! -x $runtime ]]; then
        echo "请在 pi 的独立检出目录运行；找不到运行环境：$runtime" >&2
        exit 2
    fi

    # 只读 JSON/YAML 和文件哈希，不加载 checkpoint，也不导入 Isaac Sim。
    # 微调允许换 profile，具体观测、动作及续训检查仍由 train.py 完成。
    cd "$root"
    "$runtime" -B - "$mode" "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

mode, *argv = sys.argv[1:]
p = argparse.ArgumentParser(allow_abbrev=False)
p.add_argument("--checkpoint", type=Path, required=True)
if mode == "train":
    p.add_argument("--task", choices=["Robotics-Wheelbipe-Own-Explore-v0"], required=True)
    p.add_argument("--max_iterations", type=int, required=True)
else:
    p.add_argument("--profile", required=True)
a, _ = p.parse_known_args(argv)
profile = os.environ.get("WHEELBIPE_EXPERIMENT") if mode == "train" else a.profile
profiles = json.loads(Path("profiles.json").read_text())
if profile not in profiles:
    p.error(f"未知或未指定实验配置：{profile!r}；可用配置：{', '.join(profiles)}")
if mode == "train" and a.max_iterations < 1:
    p.error("--max_iterations 必须为正数")
params = a.checkpoint.resolve().parent / "params"
for file in (a.checkpoint, *(params / name for name in ("env.yaml", "agent.yaml", "launch.json"))):
    if not file.is_file():
        p.error(f"缺少 checkpoint 或其配套文件：{file}")
launch = json.loads((params / "launch.json").read_text())
if "-Own-" not in launch.get("task", ""):
    p.error("checkpoint 的任务不是 Own 机器人")
asset_dir = Path("source/agent_world/agent_world/assets/usd_files/wheelbipe_own")
for name in ("robot.usdc", "reset_bank.json"):
    if not (Path("project") / asset_dir / name).is_file():
        p.error(f"检出目录缺少训练资产：{asset_dir / name}")
for file in (Path("project") / asset_dir).iterdir():
    if file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest() != launch.get(
        "own_asset_sha256", {}
    ).get(str(file.relative_to("project"))):
        p.error(f"训练资产与 checkpoint 不一致：{file}")
if mode in ("eval", "video"):
    saved = yaml.load((params / "env.yaml").read_text(), Loader=yaml.BaseLoader)
    expected = yaml.load(yaml.safe_dump(profiles[profile]), Loader=yaml.BaseLoader)
    if saved.get("experiment_profile_name") != profile or saved.get("experiment_profile_settings") != expected:
        p.error("评估/录制要求 checkpoint 配套的 profile；平地 model_2000.pt 请走 baseline")
    policy = yaml.safe_load((params / "agent.yaml").read_text())
    if policy.get("empirical_normalization") or policy.get("policy", {}).get("activation") != "elu":
        p.error("评估/录制入口需要未启用经验归一化的 ELU actor")
PY
fi
exec "${command[@]}" "$@"
