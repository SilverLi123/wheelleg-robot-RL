"""Evaluate velocity, yaw-rate, e-stop, and cross-slope roll tracking at 400 Hz.

Runs a compact vectorized scenario matrix in one Isaac Lab rollout so baseline
and fine-tuned checkpoints can be compared across 0.5-2.0 m/s forward speeds,
0.0-1.5 rad/s yaw rates, and a 1.5 -> 0.0 m/s emergency stop without duplicating
the full block-evaluator harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np


SPEED_SWEEP_SCENARIOS = (
    {"name": "flat_v050", "terrain": "flat", "speed_m_s": 0.50, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "flat_v100", "terrain": "flat", "speed_m_s": 1.00, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "flat_v150", "terrain": "flat", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "flat_v200", "terrain": "flat", "speed_m_s": 2.00, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "up10_v150", "terrain": "up10", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "down10_v150", "terrain": "down10", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "up18_v120", "terrain": "up18", "speed_m_s": 1.20, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "flat_v065_yaw050", "terrain": "flat", "speed_m_s": 0.65, "yaw_rate_rad_s": 0.50, "mode": "const", "target_forward_m": 0.0},
    {"name": "flat_v065_yaw100", "terrain": "flat", "speed_m_s": 0.65, "yaw_rate_rad_s": 1.00, "mode": "const", "target_forward_m": 0.0},
    {"name": "flat_v065_yaw150", "terrain": "flat", "speed_m_s": 0.65, "yaw_rate_rad_s": 1.50, "mode": "const", "target_forward_m": 0.0},
    {"name": "flat_v120_yaw100", "terrain": "flat", "speed_m_s": 1.20, "yaw_rate_rad_s": 1.00, "mode": "const", "target_forward_m": 0.0},
    {"name": "up10_v065_yaw080", "terrain": "up10", "speed_m_s": 0.65, "yaw_rate_rad_s": 0.80, "mode": "const", "target_forward_m": 0.0},
    {"name": "down10_v065_yaw080", "terrain": "down10", "speed_m_s": 0.65, "yaw_rate_rad_s": 0.80, "mode": "const", "target_forward_m": 0.0},
    {"name": "flat_estop_150", "terrain": "flat", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "estop", "brake_time_s": 2.5, "target_forward_m": 0.0},
)

HIGHSPEED_MANEUVER_SCENARIOS = (
    {"name": "flat_v100", "terrain": "flat", "speed_m_s": 1.00, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "flat_v150", "terrain": "flat", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "flat_v200", "terrain": "flat", "speed_m_s": 2.00, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "up10_v150", "terrain": "up10", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "down10_v150", "terrain": "down10", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "up18_v150", "terrain": "up18", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "down18_v150", "terrain": "down18", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "const", "target_forward_m": 10.5},
    {"name": "flat_v150_yaw100", "terrain": "flat", "speed_m_s": 1.50, "yaw_rate_rad_s": 1.00, "mode": "const", "target_forward_m": 0.0},
    {"name": "flat_v200_sturn075", "terrain": "flat", "speed_m_s": 2.00, "yaw_rate_rad_s": 0.75, "mode": "sturn", "alternating_yaw_period_s": 2.0, "target_forward_m": 0.0},
    {"name": "flat_v150_sturn", "terrain": "flat", "speed_m_s": 1.50, "yaw_rate_rad_s": 1.00, "mode": "sturn", "alternating_yaw_period_s": 2.0, "target_forward_m": 0.0},
    {"name": "down10_v120_sturn080", "terrain": "down10", "speed_m_s": 1.20, "yaw_rate_rad_s": 0.80, "mode": "sturn", "alternating_yaw_period_s": 2.0, "target_forward_m": 0.0},
    {"name": "flat_estop_150", "terrain": "flat", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "estop", "brake_time_s": 2.5, "target_forward_m": 0.0},
    {"name": "flat_estop_200", "terrain": "flat", "speed_m_s": 2.00, "yaw_rate_rad_s": 0.0, "mode": "estop", "brake_time_s": 2.5, "target_forward_m": 0.0},
    {"name": "down10_estop_150", "terrain": "down10", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "estop", "brake_time_s": 2.5, "target_forward_m": 0.0},
    {"name": "flat_reversal_150_n080", "terrain": "flat", "speed_m_s": 1.50, "yaw_rate_rad_s": 0.0, "mode": "reversal", "brake_time_s": 2.5, "reversal_speed_m_s": -0.80, "target_forward_m": 0.0},
)

SUITES = {
    "standard": SPEED_SWEEP_SCENARIOS,
    "highspeed": HIGHSPEED_MANEUVER_SCENARIOS,
}


def command_for_scenario(scenario: dict, elapsed_s: float) -> tuple[float, float]:
    """Return commanded ``(vx_m_s, wz_rad_s)`` at ``elapsed_s``."""
    mode = scenario.get("mode", "const")
    if mode == "const":
        return float(scenario["speed_m_s"]), float(scenario["yaw_rate_rad_s"])
    if mode == "estop":
        brake_time_s = float(scenario["brake_time_s"])
        if elapsed_s < brake_time_s:
            return float(scenario["speed_m_s"]), 0.0
        return 0.0, 0.0
    if mode == "sturn":
        period_s = float(scenario["alternating_yaw_period_s"])
        if period_s <= 0.0:
            raise ValueError(f"alternating_yaw_period_s must be positive, got {period_s}")
        half = 0.5 * period_s
        sign = 1.0 if (elapsed_s % period_s) < half else -1.0
        return float(scenario["speed_m_s"]), sign * float(scenario["yaw_rate_rad_s"])
    if mode == "reversal":
        brake_time_s = float(scenario["brake_time_s"])
        if elapsed_s < brake_time_s:
            return float(scenario["speed_m_s"]), 0.0
        return float(scenario["reversal_speed_m_s"]), 0.0
    raise ValueError(f"Unsupported speed sweep mode: {mode!r}")


def resolve_scenario_blocks(
    layout: tuple[tuple[str, ...], ...],
    repeats: int,
    scenarios: tuple[dict, ...] = SPEED_SWEEP_SCENARIOS,
) -> tuple[list[dict], list[int]]:
    """Expand ``scenarios`` into per-env scenario specs and block ids."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    terrain_to_block = {}
    num_cols = len(layout[0])
    for row_idx, row in enumerate(layout):
        for col_idx, name in enumerate(row):
            terrain_to_block[name] = row_idx * num_cols + col_idx
    envs = []
    block_ids = []
    for scenario in scenarios:
        terrain = scenario["terrain"]
        if terrain not in terrain_to_block:
            raise KeyError(f"Terrain {terrain!r} missing from layout")
        block_id = terrain_to_block[terrain]
        for repeat_idx in range(repeats):
            envs.append({**scenario, "repeat": repeat_idx, "block_id": block_id})
            block_ids.append(block_id)
    return envs, block_ids


def expected_terrain_roll_rad(gx: float | np.ndarray, gy: float | np.ndarray, yaw_rad: float | np.ndarray):
    """Return expected body roll (rad) when aligned perpendicular to local slope."""
    return np.arctan(gy * np.cos(yaw_rad) - gx * np.sin(yaw_rad))


def _load_actor_from_checkpoint(checkpoint_path: Path, device: str):
    import torch

    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {
        key.removeprefix("actor."): value
        for key, value in saved["model_state_dict"].items()
        if key.startswith("actor.")
    }
    weights = sorted(
        (int(key.split(".")[0]), value)
        for key, value in state.items()
        if key.endswith(".weight")
    )
    layers = []
    for index, (_, weight) in enumerate(weights):
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        if index < len(weights) - 1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to(device).eval()
    actor.load_state_dict(state, strict=True)
    return actor, saved


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="random_terrain_stage2_steep_agile_20261114")
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=71903)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--suite",
        choices=tuple(SUITES.keys()),
        default="standard",
        help="Scenario suite: standard (default 14x3 matrix) or highspeed (1.5-2.0 m/s, turns, S-turns, e-stops, reversal).",
    )
    parser.add_argument(
        "--steep-grades",
        action="store_true",
        help="Override GRADES with RM steep grades (+/-10%, +/-18%, +/-24%, +/-30%).",
    )
    return parser


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.seconds <= 1.5:
        parser.error("--seconds must exceed 1.5 s warmup")

    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    os.environ["WHEELBIPE_EXPERIMENT"] = args.profile
    os.environ["WHEELBIPE_DIAGNOSTICS_DIR"] = str(args.output / "diagnostics")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostics").mkdir(exist_ok=True)
    os.chdir(args.output)

    started = time.monotonic()
    report = {
        "status": "INITIALIZING",
        "profile": args.profile,
        "seed": args.seed,
        "seconds": args.seconds,
        "repeats": args.repeats,
        "suite": args.suite,
        "steep_grades": bool(args.steep_grades),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    }
    app = None
    try:
        app = AppLauncher(args, fast_shutdown=True).app
        import torch
        from isaaclab.utils.io import dump_yaml
        from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
        from agent_tasks.direct.wheelbipe.wheelbipe_own import random_terrain_map
        from agent_tasks.direct.wheelbipe.wheelbipe_own.experiment import (
            WheelbipeOwnExperimentEnv,
            WheelbipeOwnExperimentPlayCfg,
        )
        from agent_world.sim620_compat import torch_data as T
        from scripts.rsl_rl.checkpoint_utils import prepare_checkpoint

        if args.steep_grades:
            random_terrain_map.GRADES = {
                "flat": 0.00,
                "up06": 0.10,
                "up10": 0.18,
                "up14": 0.24,
                "up18": 0.30,
                "down06": -0.10,
                "down10": -0.18,
                "down14": -0.24,
                "down18": -0.30,
            }
        report["grades_map"] = dict(random_terrain_map.GRADES)

        cfg = WheelbipeOwnExperimentPlayCfg()
        cfg.seed = args.seed
        cfg.sim.device = args.device
        layout = random_terrain_map.make_layout(
            int(cfg.experiment_profile_settings.get("random_terrain_layout_seed", random_terrain_map.LAYOUT_SEED))
        )
        scenarios = SUITES[args.suite]
        env_specs, block_ids = resolve_scenario_blocks(layout, args.repeats, scenarios=scenarios)
        cfg.scene.num_envs = len(env_specs)
        prepare_checkpoint(cfg, None, str(args.checkpoint))
        cfg.experiment_profile_settings = dict(cfg.experiment_profile_settings)
        if float(cfg.experiment_profile_settings.get("random_terrain_grade_scale", 1.0)) != 1.0:
            cfg.experiment_profile_settings["random_terrain_grade_scale"] = 1.0
            random_terrain_map.configure(cfg)
        cfg.experiment_profile_settings["evaluation_block_ids"] = block_ids
        cfg.experiment_profile_settings["evaluation_force_yaw"] = True
        cfg.episode_length_s = args.seconds + 2.0

        actor, saved = _load_actor_from_checkpoint(args.checkpoint, cfg.sim.device)
        env = WheelbipeOwnExperimentEnv(cfg)
        env._u_eval = True

        count = cfg.scene.num_envs
        command = torch.zeros((count, 3), device=env.device)
        cmd_gen = env.command_generator
        orig_compute = cmd_gen.compute
        orig_resample = cmd_gen._resample_command

        def apply_command():
            cmd_gen.command.copy_(command)
            if hasattr(cmd_gen, "is_standing_env"):
                cmd_gen.is_standing_env[:] = False

        def compute(dt):
            orig_compute(dt)
            apply_command()

        def resample(env_ids):
            orig_resample(env_ids)
            apply_command()

        cmd_gen.compute = compute
        cmd_gen._resample_command = resample

        for idx, spec in enumerate(env_specs):
            vx, wz = command_for_scenario(spec, 0.0)
            command[idx, 0] = vx
            command[idx, 2] = wz

        obs, _ = env.reset()
        apply_command()
        obs = env._get_observations()
        start_xy = T(env.robot.data.root_pos_w)[:, :2].clone()
        last_xy = start_xy.clone()

        original_get_dones = env._get_dones
        latest_step = [None]

        def capture_get_dones():
            terminated, timed_out = original_get_dones()
            pos = T(env.robot.data.root_pos_w)
            quat = T(env.robot.data.root_quat_w)
            lin_b = T(env.robot.data.root_lin_vel_b)
            ang_b = T(env.robot.data.root_ang_vel_b)
            roll, pitch, yaw = euler_xyz_from_quat(quat)
            roll = wrap_to_pi(roll)
            pitch = wrap_to_pi(pitch)
            local_xy = pos[:, :2] - T(env.terrain.env_origins)[:, :2]
            offsets = local_xy.new_tensor(((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25)))
            ground = random_terrain_map._heights(env, local_xy[:, None, :] + offsets)
            gx = (ground[:, 0] - ground[:, 1]) / 0.5
            gy = (ground[:, 2] - ground[:, 3]) / 0.5
            exp_roll = torch.atan(gy * yaw.cos() - gx * yaw.sin())
            rel_roll = wrap_to_pi(roll - exp_roll)
            loop_err = env._u_closure_errors().amax(dim=-1)
            latest_step[0] = (pos[:, :2].clone(), lin_b.clone(), ang_b[:, 2].clone(), roll.clone(), rel_roll.clone(), pitch.clone(), loop_err.clone())
            return terminated, timed_out

        env._get_dones = capture_get_dones

        active = torch.ones(count, dtype=torch.bool, device=env.device)
        failed = torch.zeros_like(active)
        timed_out = torch.zeros_like(active)
        completed = torch.zeros_like(active)
        steps_survived = torch.zeros(count, dtype=torch.long, device=env.device)
        metric_steps = torch.zeros(count, dtype=torch.long, device=env.device)
        vx_err_sum = torch.zeros(count, device=env.device)
        vx_sum = torch.zeros(count, device=env.device)
        wz_err_sum = torch.zeros(count, device=env.device)
        world_roll_sum = torch.zeros(count, device=env.device)
        rel_roll_sum = torch.zeros(count, device=env.device)
        max_abs_world_roll = torch.zeros(count, device=env.device)
        max_abs_rel_roll = torch.zeros(count, device=env.device)
        max_abs_pitch = torch.zeros(count, device=env.device)
        max_up_vz_b = torch.zeros(count, device=env.device)
        max_loop_err = torch.zeros(count, device=env.device)
        failure_causes = [set() for _ in range(count)]

        estop_brake_x = [None] * count
        estop_stop_x = [None] * count
        estop_stop_t = [None] * count
        estop_brake_pitch = [0.0] * count
        estop_post_vx_err_sum = [0.0] * count
        estop_post_steps = [0] * count

        warmup_steps = max(round(1.0 / env.step_dt), 0)
        steps = max(round(args.seconds / env.step_dt), 1)
        target_fwd_t = torch.tensor([float(s.get("target_forward_m", 0.0)) for s in env_specs], device=env.device)

        with torch.inference_mode():
            for step in range(steps):
                t_s = step * env.step_dt
                for idx, spec in enumerate(env_specs):
                    vx_cmd, wz_cmd = command_for_scenario(spec, t_s)
                    command[idx, 0] = vx_cmd
                    command[idx, 2] = wz_cmd
                apply_command()

                action = actor(obs["policy"])
                latest_step[0] = None
                obs, _, done, timeout, _ = env.step(action)
                active_before = active.clone()
                active &= ~done & ~timeout

                pos_xy, lin_b, yaw_rate, roll, rel_roll, pitch, loop_err = latest_step[0]
                not_reset = active_before & ~done & ~timeout
                last_xy[not_reset] = pos_xy[not_reset]
                steps_survived[active_before] += 1
                failed |= done & active_before
                timed_out |= timeout & active_before

                reached = (target_fwd_t > 0.0) & (step >= warmup_steps) & not_reset & ((pos_xy[:, 0] - start_xy[:, 0]) >= target_fwd_t)
                completed |= reached
                active &= ~reached

                max_loop_err[active_before] = torch.maximum(max_loop_err[active_before], loop_err[active_before])
                max_up_vz_b[active_before] = torch.maximum(max_up_vz_b[active_before], lin_b[active_before, 2].clamp_min(0.0))

                if step >= warmup_steps:
                    metric_steps[active_before] += 1
                    vx_err_sum[active_before] += (lin_b[active_before, 0] - command[active_before, 0]).abs()
                    vx_sum[active_before] += lin_b[active_before, 0]
                    wz_err_sum[active_before] += (yaw_rate[active_before] - command[active_before, 2]).abs()
                    world_roll_sum[active_before] += roll[active_before]
                    rel_roll_sum[active_before] += rel_roll[active_before]
                    max_abs_world_roll[active_before] = torch.maximum(max_abs_world_roll[active_before], roll[active_before].abs())
                    max_abs_rel_roll[active_before] = torch.maximum(max_abs_rel_roll[active_before], rel_roll[active_before].abs())
                    max_abs_pitch[active_before] = torch.maximum(max_abs_pitch[active_before], pitch[active_before].abs())

                for idx, spec in enumerate(env_specs):
                    if spec.get("mode") != "estop" or not active_before[idx]:
                        continue
                    brake_t = float(spec["brake_time_s"])
                    if t_s >= brake_t:
                        if estop_brake_x[idx] is None:
                            estop_brake_x[idx] = float(pos_xy[idx, 0])
                        estop_brake_pitch[idx] = max(estop_brake_pitch[idx], abs(float(np.degrees(pitch[idx].item()))))
                        if estop_stop_t[idx] is None and abs(float(lin_b[idx, 0])) <= 0.05:
                            estop_stop_t[idx] = t_s - brake_t
                            estop_stop_x[idx] = float(pos_xy[idx, 0])
                        if t_s >= brake_t + 1.5:
                            estop_post_vx_err_sum[idx] += abs(float(lin_b[idx, 0]))
                            estop_post_steps[idx] += 1

                for key, mask in getattr(env._termination_reason_masks, "items", lambda: [])():
                    for idx in torch.nonzero(active_before & done & mask, as_tuple=False).flatten().tolist():
                        failure_causes[idx].add(key)

                if not bool(active.any()):
                    break

        rows = []
        grouped = {}
        for idx, spec in enumerate(env_specs):
            success = not bool(failed[idx] or timed_out[idx])
            steps_m = max(int(metric_steps[idx]), 1)
            row = {
                "scenario": spec["name"],
                "terrain": spec["terrain"],
                "repeat": spec["repeat"],
                "speed_cmd_m_s": spec["speed_m_s"],
                "yaw_cmd_rad_s": spec["yaw_rate_rad_s"],
                "mode": spec["mode"],
                "success": success,
                "completed_block": bool(completed[idx]),
                "failure_causes": sorted(failure_causes[idx]),
                "steps_survived": int(steps_survived[idx]),
                "forward_m": float(last_xy[idx, 0] - start_xy[idx, 0]),
                "lateral_drift_m": float(last_xy[idx, 1] - start_xy[idx, 1]),
                "mean_vx_m_s": float(vx_sum[idx] / steps_m),
                "vx_mae_m_s": float(vx_err_sum[idx] / steps_m),
                "yaw_mae_rad_s": float(wz_err_sum[idx] / steps_m),
                "mean_world_roll_deg": float(np.degrees(world_roll_sum[idx].item() / steps_m)),
                "mean_terrain_rel_roll_deg": float(np.degrees(rel_roll_sum[idx].item() / steps_m)),
                "max_abs_world_roll_deg": float(np.degrees(max_abs_world_roll[idx].item())),
                "max_abs_terrain_rel_roll_deg": float(np.degrees(max_abs_rel_roll[idx].item())),
                "max_abs_pitch_deg": float(np.degrees(max_abs_pitch[idx].item())),
                "max_upward_vz_b_m_s": float(max_up_vz_b[idx]),
                "max_loop_error_m": float(max_loop_err[idx]),
            }
            if spec["mode"] == "estop":
                row["estop_brake_max_abs_pitch_deg"] = estop_brake_pitch[idx]
                row["estop_stop_time_s"] = estop_stop_t[idx]
                row["estop_stop_distance_m"] = (
                    None if (estop_stop_x[idx] is None or estop_brake_x[idx] is None)
                    else abs(estop_stop_x[idx] - estop_brake_x[idx])
                )
                row["estop_hold_vx_mae_m_s"] = (
                    None if estop_post_steps[idx] == 0
                    else estop_post_vx_err_sum[idx] / estop_post_steps[idx]
                )
            rows.append(row)
            grouped.setdefault(spec["name"], []).append(row)

        scenario_summary = {}
        for name, items in grouped.items():
            summary_entry = {
                "terrain": items[0]["terrain"],
                "attempts": len(items),
                "successes": sum(1 for r in items if r["success"]),
                "mean_vx_m_s": float(np.mean([r["mean_vx_m_s"] for r in items])),
                "vx_mae_m_s": float(np.mean([r["vx_mae_m_s"] for r in items])),
                "yaw_mae_rad_s": float(np.mean([r["yaw_mae_rad_s"] for r in items])),
                "mean_world_roll_deg": float(np.mean([r["mean_world_roll_deg"] for r in items])),
                "mean_terrain_rel_roll_deg": float(np.mean([r["mean_terrain_rel_roll_deg"] for r in items])),
                "max_abs_world_roll_deg": float(np.max([r["max_abs_world_roll_deg"] for r in items])),
                "max_abs_terrain_rel_roll_deg": float(np.max([r["max_abs_terrain_rel_roll_deg"] for r in items])),
                "max_abs_pitch_deg": float(np.max([r["max_abs_pitch_deg"] for r in items])),
                "max_loop_error_m": float(np.max([r["max_loop_error_m"] for r in items])),
            }
            if items[0]["mode"] == "estop":
                stop_times = [r["estop_stop_time_s"] for r in items if r["estop_stop_time_s"] is not None]
                stop_dists = [r["estop_stop_distance_m"] for r in items if r["estop_stop_distance_m"] is not None]
                hold_maes = [r["estop_hold_vx_mae_m_s"] for r in items if r["estop_hold_vx_mae_m_s"] is not None]
                summary_entry["estop_brake_max_abs_pitch_deg"] = float(np.max([r["estop_brake_max_abs_pitch_deg"] for r in items]))
                summary_entry["estop_stop_time_s"] = float(np.mean(stop_times)) if stop_times else None
                summary_entry["estop_stop_distance_m"] = float(np.mean(stop_dists)) if stop_dists else None
                summary_entry["estop_hold_vx_mae_m_s"] = float(np.mean(hold_maes)) if hold_maes else None
            scenario_summary[name] = summary_entry

        dump_yaml(str(args.output / "env.yaml"), cfg)
        report.update(
            status="PASS" if all(r["success"] for r in rows) else "WARN_POLICY",
            checkpoint_iteration=int(saved["iter"]),
            max_loop_error_m=float(max(r["max_loop_error_m"] for r in rows)),
            scenario_summary=scenario_summary,
            rows=rows,
        )
        env.close()
    except BaseException:
        report["status"] = "ERROR"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if app is not None:
            app.close(exit_code=0 if report["status"] in ("PASS", "WARN_POLICY") else 1)


if __name__ == "__main__":
    main()
