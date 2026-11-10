"""Intervention test: overlap env0 with env1; perturb only env0; compare env1 to baseline."""
import argparse
import json
import traceback
from pathlib import Path

from preflight import require_target
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
versions = require_target()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import agent_tasks
from isaaclab_tasks.utils import parse_env_cfg


def main():
    if args.num_envs < 2:
        raise ValueError("Isolation requires at least two environments")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg("Robotics-Wheelbipe-Own-Flat-v0", device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = 42
    env = gym.make("Robotics-Wheelbipe-Own-Flat-v0", cfg=cfg)
    e = env.unwrapped
    native = e._native_robot
    def run(overlap):
        env.reset()
        if overlap:
            state = native.data.root_state_w[1:2].clone()
            native.write_root_state_to_sim(state, env_ids=torch.tensor([0], device=e.device))
            e.sim.forward()
            e.scene.update(0.0)
            e._invalidate_step_caches()
        action = torch.zeros(args.num_envs, 6, device=e.device)
        if overlap:
            action[0, 4:] = 0.5
        rows = []
        for step in range(args.steps):
            _, _, terminated, timeout, _ = env.step(action)
            if terminated[1] or timeout[1]:
                raise RuntimeError("Protected environment reset during isolation probe")
            rows.append({"protected_root_velocity": native.data.root_state_w[1, 7:].tolist(),
                "protected_q": native.data.joint_pos[1].tolist(),
                "intervention_root_velocity": native.data.root_state_w[0, 7:].tolist()})
        return rows
    try:
        baseline = run(False)
        intervention = run(True)
        v_error = max(abs(a - b) for x, y in zip(baseline, intervention)
                      for a, b in zip(x["protected_root_velocity"], y["protected_root_velocity"]))
        q_error = max(abs(a - b) for x, y in zip(baseline, intervention)
                      for a, b in zip(x["protected_q"], y["protected_q"]))
        report = {"status": "PASS" if v_error < 0.001 and q_error < 0.001 else "FAIL",
            "environment": versions, "num_envs": args.num_envs,
            "method": "env0 placed exactly over env1 and wheel commands applied only to env0; protected env1 compared to zero-action baseline",
            "max_protected_velocity_error": v_error, "max_protected_q_error": q_error,
            "baseline": baseline, "intervention": intervention}
        (args.output / "isolation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k not in ("environment", "baseline", "intervention")}, indent=2))
        if report["status"] != "PASS":
            raise RuntimeError("Environment collision/constraint isolation failed")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        with torch.inference_mode():
            main()
    except BaseException:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        app.close(wait_for_replicator=False)
