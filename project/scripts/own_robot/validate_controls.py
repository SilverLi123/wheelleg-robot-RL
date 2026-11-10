"""Paired free-body action probes against zero-action trajectories on target GPU."""
import argparse
import json
import math
import traceback
from pathlib import Path

from preflight import require_target
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--physics-hz", type=int, default=200)
parser.add_argument("--position-iterations", type=int, default=128)
parser.add_argument("--velocity-iterations", type=int, default=8)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--attitude-fixture", action="store_true", help="Apply only base world-X/Y attitude torque for an upright rolling test; never used in training.")
parser.add_argument("--warmup-steps", type=int, default=0)
parser.add_argument("--rolling-only", action="store_true", help="Run longer forward/yaw probes after individual channels were checked.")
parser.add_argument("--only-test", choices=["all", "forward", "yaw"], default="all")
parser.add_argument("--wheel-action", type=float, default=0.2, help="Wheel action magnitude; target angular velocity is 10 times this value.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
versions = require_target()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import agent_tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg("Robotics-Wheelbipe-Own-Flat-v0", device=args.device or "cuda:0", num_envs=1)
    cfg.seed = 42
    cfg.sim.dt, cfg.decimation = 1 / args.physics_hz, args.physics_hz // 50
    cfg.sim.render_interval = cfg.decimation
    cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = args.position_iterations
    cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
    env = gym.make("Robotics-Wheelbipe-Own-Flat-v0", cfg=cfg)
    e = env.unwrapped
    native = e._native_robot
    masses = native.root_physx_view.get_masses().to(e.device)
    masses /= masses.sum(-1, keepdim=True)
    wheel_bodies, _ = native.find_bodies(["left_wheel_link", "right_wheel_link"], preserve_order=True)
    if args.attitude_fixture:
        base_bodies, _ = native.find_bodies("base_link")
        original_apply = e._apply_action
        def apply_with_fixture():
            original_apply()
            roll, pitch, yaw = euler_xyz_from_quat(native.data.root_quat_w)
            torque = torch.zeros(1, 1, 3, device=e.device)
            torque[:, 0, 0] = (-100 * wrap_to_pi(roll) - 10 * native.data.root_ang_vel_w[:, 0]).clamp(-20, 20)
            torque[:, 0, 1] = (-100 * wrap_to_pi(pitch) - 10 * native.data.root_ang_vel_w[:, 1]).clamp(-20, 20)
            native.set_external_force_and_torque(torch.zeros_like(torque), torque,
                                                body_ids=base_bodies, is_global=True)
        e._apply_action = apply_with_fixture
    trace = (args.output / "trajectory.jsonl").open("w", buffering=1)
    trajectory_number = 0
    def trajectory(action):
        nonlocal trajectory_number
        trajectory_number += 1
        env.reset()
        for _ in range(args.warmup_steps):
            _, _, terminated, timeout, _ = env.step(torch.zeros_like(action))
            if terminated.any() or timeout.any():
                raise RuntimeError("Fixture could not settle without resetting")
        rows = []
        for step in range(args.steps):
            # Record the actual pre-reset state. DirectRLEnv.step() performs an
            # autoreset before returning and would erase the failure evidence.
            e._pre_physics_step(action)
            for _ in range(cfg.decimation):
                e._apply_action()
                e.scene.write_data_to_sim()
                e.sim.step(render=False)
                e.scene.update(cfg.sim.dt)
            e.episode_length_buf += 1
            e._sync_policy_state()
            terminated, timeout = e._get_dones()
            com_vel = (native.data.body_lin_vel_w * masses[..., None]).sum(dim=1)
            com_pos = (native.data.body_com_pos_w * masses[..., None]).sum(dim=1)
            rows.append({"step": step, "policy_q": e.robot.data.joint_pos[0, e._actuate_idx].tolist(),
                "policy_dq": e.robot.data.joint_vel[0, e._actuate_idx].tolist(),
                "native_q_target": native.data.joint_pos_target[0, e._actuate_idx].tolist(),
                "native_dq_target": native.data.joint_vel_target[0, e._actuate_idx].tolist(),
                "root_velocity_w": native.data.root_lin_vel_w[0].tolist(),
                "com_velocity_w": com_vel[0].tolist(), "com_position_w": com_pos[0].tolist(),
                "root_heading": float(native.data.heading_w[0]),
                "wheel_angular_velocity_w": native.data.body_ang_vel_w[0, wheel_bodies].tolist(),
                "wheel_center_position_w": native.data.body_pos_w[0, wheel_bodies].tolist(),
                "native_wheel_effort": native.data.applied_torque[0, e._wheel_idx].tolist(),
                "root_quaternion_wxyz": native.data.root_quat_w[0].tolist(),
                "contact_forces_w": e.contact_sensor.data.net_forces_w[0].tolist(),
                "root_yaw_rate": float(native.data.root_ang_vel_w[0, 2]),
                "spring_position": native.data.joint_pos[0, e._spring_idx].tolist(),
                "terminated": bool(terminated[0]),
                "termination_causes": {k: bool(v[0]) for k, v in e._termination_reason_masks.items()}})
            trace.write(json.dumps({"trajectory": trajectory_number, "action": action[0].tolist(), **rows[-1]}) + "\n")
            if terminated.any() or timeout.any():
                raise RuntimeError(f"Probe terminated at step {step}: {rows[-1]['termination_causes']}; inspect trajectory.jsonl")
        return rows
    results = []
    action = torch.zeros(1, 6, device=e.device)
    try:
        for channel in ([] if args.rolling_only else range(6)):
            baseline = trajectory(action)
            probe_action = action.clone()
            probe_action[:, channel] = 0.1 if channel < 4 else args.wheel_action
            probe = trajectory(probe_action)
            # Joint displacement integrates the response across the entire
            # landing transient. A single 50Hz velocity sample can alias the
            # faster contact oscillation; retain those velocities in the log.
            key = "policy_q"
            response = probe[-1][key][channel] - baseline[-1][key][channel]
            results.append({"name": f"channel_{channel}", "quantity": key, "response": response,
                "instantaneous_velocity_delta": probe[-1]["policy_dq"][channel] - baseline[-1]["policy_dq"][channel],
                "pass": response > 1e-4, "baseline": baseline, "probe": probe})
        a = args.wheel_action
        for name, values, quantity in (("both_wheels_forward", [a, a], "com_position_w"),
                                       ("differential_yaw", [-a, a], "root_heading")):
            if args.only_test == "forward" and name != "both_wheels_forward":
                continue
            if args.only_test == "yaw" and name != "differential_yaw":
                continue
            baseline = trajectory(action)
            probe_action = action.clone()
            probe_action[:, 4:] = probe_action.new_tensor(values)
            probe = trajectory(probe_action)
            if quantity == "com_position_w":
                response = probe[-1][quantity][0] - baseline[-1][quantity][0]
            else:
                def unwrapped_heading(rows):
                    values = [x[quantity] for x in rows]
                    return values[0] + sum(math.atan2(math.sin(b-a), math.cos(b-a)) for a, b in zip(values, values[1:]))
                response = unwrapped_heading(probe) - unwrapped_heading(baseline)
            results.append({"name": name, "quantity": quantity, "response": response,
                "pass": response > 1e-4, "baseline": baseline, "probe": probe})
        report = {"status": "PASS" if all(x["pass"] for x in results) else "FAIL", "environment": versions,
            "fixture": "base world-X/Y torques only, K100 D10 limit20Nm; no force or yaw torque" if args.attitude_fixture else "none",
            "warmup_steps": args.warmup_steps,
            "wheel_action": args.wheel_action,
            "definition": "positive integrated joint rotation, +X COM displacement and +Z heading vs paired zero action; retain instantaneous velocities for diagnosis",
            "steps_per_probe": args.steps, "dt_policy": e.step_dt, "results": results}
        (args.output / "control_tests.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"status": report["status"], "results": [{k: r[k] for k in ("name", "quantity", "response", "pass")} for r in results]}, indent=2))
        if report["status"] != "PASS":
            raise RuntimeError("Control sign/response probe failed; inspect paired trajectories")
    finally:
        trace.close()
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
