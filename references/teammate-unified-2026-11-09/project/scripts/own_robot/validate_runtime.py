"""Sim5.1/Lab2.3 acceptance harness. Never counts an old runtime as a pass.

Writes per-policy-step telemetry and physics-substep closure/force telemetry.
Includes a raw continuous physics mode without RL autoresets for the 60s test.
"""
import argparse
import json
import math
import time
import traceback
from pathlib import Path

from preflight import require_target

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Robotics-Wheelbipe-V14-Flat-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seconds", type=float, default=60)
parser.add_argument("--mode", choices=["hold", "sweep", "reset", "continuous"], default="hold")
parser.add_argument("--resets", type=int, default=1000)
parser.add_argument("--reset-hold-steps", type=int, default=10, help="Integrate each sampled reset through landing before the next reset.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--physics-hz", type=int, default=200)
parser.add_argument("--position-iterations", type=int, default=128)
parser.add_argument("--velocity-iterations", type=int, default=8)
parser.add_argument("--output", type=Path, required=True)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
versions = require_target()
app = AppLauncher(args).app

import csv
import gymnasium as gym
import torch
from pxr import UsdPhysics
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.io import dump_yaml
import agent_tasks  # noqa: F401


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "environment.json").write_text(json.dumps(versions, indent=2))
    own = "-Own-" in args.task
    if args.physics_hz % 50:
        raise ValueError("Physics frequency must be divisible by 50")
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    if not cfg.sim.device.startswith("cuda"):
        raise ValueError("Final acceptance requires CUDA GPU physics")
    if own:
        cfg.own_physics_hz = args.physics_hz
        cfg.sim.dt = 1.0 / args.physics_hz
        cfg.decimation = args.physics_hz // 50
        cfg.sim.render_interval = cfg.decimation
        cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = args.position_iterations
        cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
        cfg.own_random_reset = args.mode == "reset"
    dump_yaml(str(args.output / "env.yaml"), cfg)
    env = gym.make(args.task, cfg=cfg)
    e = env.unwrapped
    native = getattr(e, "_native_robot", e.robot)
    obs, _ = env.reset()
    assert gym.spaces.flatdim(e.single_action_space) == 6
    assert obs["policy"].shape == (args.num_envs, 35)
    assert obs["critic"].shape == (args.num_envs, 78)
    loops = []
    # Inspect every cloned target, not just env_0.
    for i in range(args.num_envs):
        prim = e.sim.stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        found = []
        from pxr import Usd
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdPhysics.SphericalJoint):
                j = UsdPhysics.Joint(p)
                body_targets = [j.GetBody0Rel().GetTargets()[0], j.GetBody1Rel().GetTargets()[0]]
                if not all(str(t).startswith(f"/World/envs/env_{i}/Robot/") for t in body_targets):
                    raise RuntimeError(f"Cross-environment loop target: {p.GetPath()}")
                if not j.GetJointEnabledAttr().Get() or not j.GetExcludeFromArticulationAttr().Get():
                    raise RuntimeError("Loop disabled or inserted into articulation")
                found.append({"name": p.GetName(), "ids": [native.body_names.index(t.name) for t in body_targets],
                              "anchors": [list(j.GetLocalPos0Attr().Get()), list(j.GetLocalPos1Attr().Get())]})
        assert len(found) == 6
        loops.append(found)
    if own:
        (args.output / "runtime_indices.json").write_text(json.dumps(e.index_manifest(), indent=2))
    action = torch.zeros(args.num_envs, 6, device=e.device)
    closure_file = (args.output / "closure.csv").open("w")
    closure_writer = csv.writer(closure_file)
    closure_writer.writerow(["physics_step", "env_id", *[j["name"] for j in loops[0]]])
    policy_file = (args.output / "policy.jsonl").open("w")
    reward_values = {}
    reset_counts = {}
    max_error = 0.0
    force_error = 0.0
    above_2mm_run, longest_2mm = 0, 0
    pair_ids = torch.tensor([[j["ids"] for j in row] for row in loops], device=e.device)
    anchor_tensor = torch.tensor([[j["anchors"] for j in row] for row in loops], device=e.device)
    env_indices = torch.arange(args.num_envs, device=e.device)[:, None, None]

    def sample_physics(step):
        nonlocal max_error, force_error, above_2mm_run, longest_2mm
        pos = native.data.body_pos_w[env_indices, pair_ids]
        quat = native.data.body_quat_w[env_indices, pair_ids]
        points = pos + quat_apply(quat, anchor_tensor)
        error = torch.linalg.vector_norm(points[:, :, 0] - points[:, :, 1], dim=-1)
        if not torch.isfinite(error).all():
            raise FloatingPointError("Nonfinite closure error")
        peak = float(error.max())
        max_error = max(max_error, peak)
        above_2mm_run = above_2mm_run + 1 if peak > 0.002 else 0
        longest_2mm = max(longest_2mm, above_2mm_run)
        host_error = error.tolist()
        closure_writer.writerows([step, env_id, *row] for env_id, row in enumerate(host_error))
        if own:
            # The cached applied effort corresponds to the state at the last
            # write_data_to_sim, before integration (not the post-step velocity).
            measured = native.data.applied_torque[:, e._spring_idx]
            delta = (measured - e._spring_commanded_net).abs().max()
            force_error = max(force_error, float(delta))

    # Wrap the native scene update only for recording. This samples each substep,
    # including spikes that a 50 Hz observation would miss.
    original_update = e.scene.update
    original_step = e.sim.step
    physics_step = 0
    physics_armed = False
    def sim_step(*args, **kwargs):
        nonlocal physics_armed
        result = original_step(*args, **kwargs)
        physics_armed = True
        return result
    def update(dt):
        nonlocal physics_step, physics_armed
        original_update(dt)
        if physics_armed:
            sample_physics(physics_step)
            physics_step += 1
            physics_armed = False
    e.scene.update = update
    e.sim.step = sim_step
    torch.cuda.synchronize()
    start = time.perf_counter()
    steps = math.ceil(args.seconds / e.step_dt)
    if args.mode == "reset":
        steps = math.ceil(args.resets / args.num_envs) * args.reset_hold_steps
    try:
        for step in range(steps):
            if not app.is_running():
                raise RuntimeError("Simulator stopped before validation finished")
            if args.mode == "reset" and step % args.reset_hold_steps == 0:
                env.reset()
            action.zero_()
            if args.mode == "sweep":
                # 8 phases: each motor separately, both wheels forward, differential yaw.
                phase = int(step * e.step_dt // 4) % 8
                pulse = 0.10 * math.sin(2 * math.pi * (step * e.step_dt % 4) / 4)
                if phase < 6:
                    action[:, phase] = pulse
                elif phase == 6:
                    action[:, 4:] = 0.1
                else:
                    action[:, 4] = -0.1
                    action[:, 5] = 0.1
            if args.mode == "continuous":
                # No RL dones/reset can hide a long-time constraint failure.
                e._pre_physics_step(action)
                for _ in range(cfg.decimation):
                    e._apply_action()
                    e.scene.write_data_to_sim()
                    e.sim.step(render=False)
                    e.scene.update(cfg.sim.dt)
                rewards = torch.zeros(args.num_envs, device=e.device)
            else:
                obs, rewards, done, timeout, info = env.step(action)
                if not all(torch.isfinite(v).all() for v in obs.values()) or not torch.isfinite(rewards).all():
                    raise FloatingPointError("Nonfinite runtime output")
                for key, mask in getattr(e, "_termination_reason_masks", {}).items():
                    reset_counts[key] = reset_counts.get(key, 0) + int((mask & done).sum())
                for key, value in getattr(e, "_own_raw_rewards", {}).items():
                    reward_values.setdefault(key, []).append([float(value.min()), float(value.max())])
            q = native.data.joint_pos
            dq = native.data.joint_vel
            effort = native.data.applied_torque
            if not all(torch.isfinite(x).all() for x in (q, dq, effort)):
                raise FloatingPointError("Nonfinite physical state")
            for env_id in range(args.num_envs):
                policy_file.write(json.dumps({"step": step, "env_id": env_id, "root_state": native.data.root_state_w[env_id].tolist(),
                    "native_q": q[env_id].tolist(), "native_dq": dq[env_id].tolist(),
                    "applied_effort": effort[env_id].tolist(), "action": action[env_id].tolist(),
                    "spring_expected_effort": e._spring_commanded_net[env_id].tolist() if own else None,
                    "contact_forces": e.contact_sensor.data.net_forces_w[env_id].tolist()}) + "\n")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        closure_ok = longest_2mm * cfg.sim.dt <= 0.1
        force_ok = force_error <= 0.1
        accepted = (closure_ok and force_ok) if own else True
        summary = {"status": "PASSED_PHYSICS_GATES_REVIEW_RESPONSE" if own and accepted else
                   "FAILED_PHYSICS_GATES" if own else "BASELINE_EXECUTION_RECORDED",
            "physics_gates": {"closure": closure_ok, "spring_effort": force_ok},
            "task": args.task, "mode": args.mode, "num_envs": args.num_envs,
            "policy_hz": 1 / e.step_dt, "physics_hz": 1 / cfg.sim.dt,
            "sim_seconds": steps * e.step_dt, "wall_seconds": elapsed,
            "requested_random_resets": args.resets if args.mode == "reset" else None,
            "sampled_random_resets": math.ceil(args.resets / args.num_envs) * args.num_envs if args.mode == "reset" else None,
            "reset_hold_policy_steps": args.reset_hold_steps if args.mode == "reset" else None,
            "env_steps_per_second": steps * args.num_envs / elapsed,
            "max_closure_m": max_error, "longest_above_2mm_s": longest_2mm * cfg.sim.dt,
            "spring_effort_error_N": force_error, "reset_causes": reset_counts,
            "reward_extrema": {k: [min(x[0] for x in v), max(x[1] for x in v)] for k, v in reward_values.items()},
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_end": __import__("preflight").command(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"])}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        # Suggested engineering gates, explicitly adjustable after reviewing data.
        if not accepted:
            raise RuntimeError("Closure or spring effort gate failed; inspect telemetry")
        (args.output / "validation_complete.json").write_text(json.dumps({
            "completed": True, "own_physics_gates_passed": accepted if own else None,
            "baseline_execution_only": not own}, indent=2))
    finally:
        closure_file.close()
        policy_file.close()
        e.scene.update = original_update
        e.sim.step = original_step
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
        # Telemetry is flushed above; no Replicator capture is pending.
        app.close(wait_for_replicator=False)
