"""Evaluate a frozen policy on every patch of the random-terrain map.

The evaluator deliberately fixes the starting point and heading on each patch.
That makes checkpoint comparisons independent of the random reset distribution
used during training while keeping the physical model and domain randomization
unchanged.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", default="random_terrain_20261112")
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--seed", type=int, default=20261112)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--speed", type=float, default=0.65)
parser.add_argument("--yaw-rate", type=float, default=0.0)
parser.add_argument(
    "--steep-grades",
    action="store_true",
    help="Override GRADES with RM steep grades (+/-10%, +/-18%, +/-24%, +/-30%).",
)
parser.add_argument(
    "--target-forward-m",
    type=float,
    default=0.0,
    help="If > 0, mark an environment as completed once forward progress reaches this distance (m) after warmup.",
)
parser.add_argument(
    "--alternating-yaw-period-s",
    type=float,
    default=0.0,
    help="If > 0, alternate the sign of --yaw-rate with this full period (s) to evaluate S-turn slaloms within a single block.",
)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.repeats < 1:
    parser.error("--repeats must be positive")
if not 0.0 <= args.speed <= 2.5:
    parser.error("--speed must be in [0.0, 2.5] m/s")

args.checkpoint = args.checkpoint.expanduser().resolve()
args.output = args.output.expanduser().resolve()
os.environ["WHEELBIPE_EXPERIMENT"] = args.profile
os.environ["WHEELBIPE_DIAGNOSTICS_DIR"] = str(args.output / "diagnostics")
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "diagnostics").mkdir(exist_ok=True)
# Existing runtime diagnostics use paths relative to the working directory.
# Enter the immutable result directory so every sidecar stays in this run.
os.chdir(args.output)
started = time.monotonic()
report = {
    "status": "INITIALIZING",
    "profile": args.profile,
    "seed": args.seed,
    "seconds": args.seconds,
    "speed_m_s": args.speed,
    "yaw_rate_rad_s": args.yaw_rate,
    "steep_grades": bool(args.steep_grades),
    "target_forward_m": float(args.target_forward_m),
    "alternating_yaw_period_s": float(args.alternating_yaw_period_s),
    "repeats": args.repeats,
    "checkpoint": str(args.checkpoint),
    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
}
app = None

try:
    app = AppLauncher(args, fast_shutdown=True).app
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
    from isaaclab.utils.io import dump_yaml
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
    cfg.scene.num_envs = len(random_terrain_map.TERRAIN_TYPES) * args.repeats

    # Validate the original profile before adding evaluation-only settings.
    # These settings affect reset placement, never the policy or physics.
    prepare_checkpoint(cfg, None, str(args.checkpoint))
    block_ids = list(range(len(random_terrain_map.TERRAIN_TYPES))) * args.repeats
    cfg.experiment_profile_settings = dict(cfg.experiment_profile_settings)
    if float(cfg.experiment_profile_settings.get("random_terrain_grade_scale", 1.0)) != 1.0:
        cfg.experiment_profile_settings["random_terrain_grade_scale"] = 1.0
        random_terrain_map.configure(cfg)
    cfg.experiment_profile_settings["evaluation_block_ids"] = block_ids
    cfg.experiment_profile_settings["evaluation_force_yaw"] = True
    cfg.episode_length_s = args.seconds + 2.0

    saved = torch.load(args.checkpoint, map_location=cfg.sim.device, weights_only=False)
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
    actor = torch.nn.Sequential(*layers).to(cfg.sim.device).eval()
    actor.load_state_dict(state, strict=True)

    env = WheelbipeOwnExperimentEnv(cfg)
    env._u_eval = True
    expected_blocks = torch.tensor(block_ids, device=env.device)
    if not torch.equal(env._random_terrain_force_block, expected_blocks):
        raise RuntimeError("Environment did not preserve the requested terrain blocks")

    reset_contact_names = [
        env.contact_sensor.body_names[index]
        for index in env._reset_contact_link_idx
    ]
    gimbal_body_idx, _ = env._native_robot.find_bodies(
        "gimbal_yaw_link", preserve_order=True
    )
    if len(gimbal_body_idx) != 1:
        raise RuntimeError("Expected exactly one gimbal_yaw_link")

    from pxr import Usd, UsdGeom

    def collision_mesh_local_hull(body_name):
        """Return the body-local collision convex-hull vertices.

        Dynamic rigid bodies in PhysX use the collision mesh's convex hull.
        The hull avoids both the body-origin error and AABB corners that lie
        outside the real collision surface of a concave mesh.
        """
        from scipy.spatial import ConvexHull

        time = Usd.TimeCode.Default()
        body = env.sim.stage.GetPrimAtPath(
            f"/World/envs/env_0/Robot/{body_name}"
        )
        if not body.IsValid():
            raise RuntimeError(f"Robot body prim is missing: {body_name}")
        body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(time)
        body_inverse = body_world.GetInverse()
        points_body = []
        mesh_paths = []
        for prim in Usd.PrimRange(body):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            # The V14-6 USD keeps collision meshes below a dedicated
            # `collisions` scope. Physics uses those meshes, while the
            # visually similar geometry below `visuals` is not a contact bound.
            if "collisions" not in prim.GetPath().pathString.split("/"):
                continue
            mesh = UsdGeom.Mesh(prim)
            mesh_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                time
            )
            local_to_body = mesh_world * body_inverse
            points = mesh.GetPointsAttr().Get(time)
            if not points:
                continue
            points_body.extend(local_to_body.Transform(point) for point in points)
            mesh_paths.append(str(prim.GetPath()))
        if not points_body:
            raise RuntimeError(
                f"No collision mesh points found below {body.GetPath()}"
            )
        points_np = np.asarray(
            [tuple(float(value) for value in point) for point in points_body],
            dtype=np.float64,
        )
        points_np = np.unique(points_np, axis=0)
        hull = ConvexHull(points_np)
        hull_points = points_np[hull.vertices]
        minimum = hull_points.min(axis=0)
        maximum = hull_points.max(axis=0)
        return minimum, maximum, hull_points, mesh_paths

    (
        gimbal_local_min,
        gimbal_local_max,
        gimbal_local_hull_points,
        gimbal_collision_meshes,
    ) = collision_mesh_local_hull("gimbal_yaw_link")
    original_get_dones = env._get_dones
    latest_reset_contact_forces = [None]
    latest_gimbal_mesh_clearance = [None]
    latest_step_metrics = [None]

    def gimbal_mesh_clearance_m():
        body_pos_w = T(env._native_robot.data.body_pos_w)
        body_quat_w = T(env._native_robot.data.body_quat_w)
        origin = T(env.terrain.env_origins)
        positions = body_pos_w[:, gimbal_body_idx[0]].cpu().numpy()
        quaternions = body_quat_w[:, gimbal_body_idx[0]].cpu().numpy()[:, [3, 0, 1, 2]]
        origins = origin.cpu().numpy()
        points_per_body = len(gimbal_local_hull_points)

        def ground_at_world_xy(world_xy):
            origin_xy = np.repeat(
                origins[:, :2],
                points_per_body,
                axis=0,
            )
            local_xy_np = world_xy - origin_xy
            local_xy = torch.as_tensor(
                local_xy_np,
                dtype=torch.float32,
                device=env.device,
            )
            relative_z = T(random_terrain_map._heights(env, local_xy))
            return (
                relative_z.cpu().numpy()
                + np.repeat(origins[:, 2], points_per_body)
            )

        clearance = random_terrain_map.collision_points_minimum_clearance(
            body_pos_w=positions,
            body_quat_w=quaternions,
            local_points=gimbal_local_hull_points,
            height_at_local_xy=ground_at_world_xy,
        )
        return torch.as_tensor(
            clearance,
            dtype=body_pos_w.dtype,
            device=env.device,
        )

    def capture_get_dones():
        # `env.step()` can reset terminated environments before returning.
        # Capture the forces while Isaac Lab is still evaluating termination,
        # rather than reading an already-cleared sensor after the step.
        terminated, timed_out = original_get_dones()
        latest_reset_contact_forces[0] = torch.linalg.vector_norm(
            T(env.contact_sensor.data.net_forces_w)[
                :, env._reset_contact_link_idx
            ],
            dim=-1,
        ).clone()
        latest_gimbal_mesh_clearance[0] = gimbal_mesh_clearance_m().clone()
        roll_s, pitch_s, _ = euler_xyz_from_quat(T(env.robot.data.root_quat_w))
        roll_s = wrap_to_pi(roll_s)
        pitch_s = wrap_to_pi(pitch_s)
        mech_p_s = (
            T(env.robot.data.applied_torque)[:, env._actuate_idx]
            * T(env.robot.data.joint_vel)[:, env._actuate_idx]
        ).abs().sum(dim=-1)
        latest_step_metrics[0] = (roll_s.clone(), pitch_s.clone(), mech_p_s.clone())
        return terminated, timed_out

    env._get_dones = capture_get_dones

    count = cfg.scene.num_envs
    command = torch.zeros((count, 3), device=env.device)
    command[:, 0] = args.speed
    command[:, 2] = args.yaw_rate
    command_generator = env.command_generator
    original_compute = command_generator.compute
    original_resample = command_generator._resample_command

    def apply_command():
        command_generator.command.copy_(command)
        if hasattr(command_generator, "is_standing_env"):
            command_generator.is_standing_env[:] = False

    def compute(dt):
        original_compute(dt)
        apply_command()

    def resample(env_ids):
        original_resample(env_ids)
        apply_command()

    command_generator.compute = compute
    command_generator._resample_command = resample

    obs, _ = env.reset()
    apply_command()
    obs = env._get_observations()
    start_xy = T(env.robot.data.root_pos_w)[:, :2].clone()
    last_xy = start_xy.clone()
    initial_roll, initial_pitch, _ = euler_xyz_from_quat(T(env.robot.data.root_quat_w))
    initial_roll = wrap_to_pi(initial_roll)
    initial_pitch = wrap_to_pi(initial_pitch)

    active = torch.ones(count, dtype=torch.bool, device=env.device)
    failed = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    completed_block = torch.zeros_like(active)
    steps_survived = torch.zeros(count, dtype=torch.long, device=env.device)
    speed_error_sum = torch.zeros(count, device=env.device)
    speed_sum = torch.zeros(count, device=env.device)
    yaw_error_sum = torch.zeros(count, device=env.device)
    roll_sum_rad = torch.zeros(count, device=env.device)
    max_abs_roll_rad = torch.zeros(count, device=env.device)
    max_abs_pitch_rad = torch.zeros(count, device=env.device)
    mech_power_sum_w = torch.zeros(count, device=env.device)
    peak_mech_power_w = torch.zeros(count, device=env.device)
    metric_steps = torch.zeros(count, dtype=torch.long, device=env.device)
    pitch_min = initial_pitch.clone()
    pitch_max = initial_pitch.clone()
    warmup_steps = max(round(1.0 / env.step_dt), 0)
    steps = max(round(args.seconds / env.step_dt), 1)
    failure_causes = [set() for _ in range(count)]
    failure_contact_forces = [dict() for _ in range(count)]
    failure_gimbal_mesh_clearance = [None] * count
    minimum_gimbal_mesh_clearance = torch.full(
        (count,), float("inf"), device=env.device
    )
    trajectory = []

    with torch.inference_mode():
        for step in range(steps):
            if args.alternating_yaw_period_s > 0.0:
                phase = ((step * env.step_dt) % args.alternating_yaw_period_s) < (
                    0.5 * args.alternating_yaw_period_s
                )
                current_yaw_cmd = args.yaw_rate if phase else -args.yaw_rate
                command[:, 2] = current_yaw_cmd
                apply_command()
            else:
                current_yaw_cmd = args.yaw_rate
            action = actor(obs["policy"])
            latest_reset_contact_forces[0] = None
            latest_gimbal_mesh_clearance[0] = None
            latest_step_metrics[0] = None
            obs, reward, done, timeout, _ = env.step(action)
            active_before = active.clone()
            active &= ~done & ~timeout

            speed = T(env.robot.data.root_lin_vel_b)[:, 0]
            yaw_rate = T(env.robot.data.root_ang_vel_b)[:, 2]
            _, pitch, _ = euler_xyz_from_quat(T(env.robot.data.root_quat_w))
            pitch = wrap_to_pi(pitch)
            root_xy = T(env.robot.data.root_pos_w)[:, :2]
            not_reset = active_before & ~done & ~timeout
            last_xy[not_reset] = root_xy[not_reset]
            steps_survived[active_before] += 1
            failed |= done & active_before
            timed_out |= timeout & active_before
            if args.target_forward_m > 0.0 and step >= warmup_steps:
                reached_target = not_reset & (
                    (root_xy[:, 0] - start_xy[:, 0]) >= args.target_forward_m
                )
                completed_block |= reached_target
                active &= ~reached_target
            # Preserve the same reset-body force data that _get_dones() used.
            # `_undesired_contact_link_idx` is broader than the reset set, and
            # the contact sensor may have been cleared by the automatic reset
            # before env.step() returns.
            reset_forces = latest_reset_contact_forces[0]
            if reset_forces is None:
                raise RuntimeError(
                    "Failed to capture contact forces before environment reset"
                )
            step_gimbal_mesh_clearance = latest_gimbal_mesh_clearance[0]
            if step_gimbal_mesh_clearance is None:
                raise RuntimeError(
                    "Failed to capture gimbal mesh clearance before environment reset"
                )
            minimum_gimbal_mesh_clearance[active_before] = torch.minimum(
                minimum_gimbal_mesh_clearance[active_before],
                step_gimbal_mesh_clearance[active_before],
            )
            for index in torch.nonzero(
                done & active_before, as_tuple=False
            ).flatten().tolist():
                failure_gimbal_mesh_clearance[index] = float(
                    step_gimbal_mesh_clearance[index]
                )
                for body_name, force in zip(
                    reset_contact_names,
                    reset_forces[index].tolist(),
                ):
                    if force > 0.1:
                        failure_contact_forces[index][body_name] = max(
                            force,
                            failure_contact_forces[index].get(body_name, 0.0),
                        )
            pitch_min = torch.minimum(pitch_min, pitch)
            pitch_max = torch.maximum(pitch_max, pitch)

            if step >= warmup_steps:
                speed_error_sum[active_before] += (
                    speed[active_before] - args.speed
                ).abs()
                speed_sum[active_before] += speed[active_before]
                yaw_error_sum[active_before] += (
                    yaw_rate[active_before] - current_yaw_cmd
                ).abs()
                metric_steps[active_before] += 1
                step_roll, step_pitch, step_mech_p = latest_step_metrics[0]
                roll_sum_rad[active_before] += step_roll[active_before]
                max_abs_roll_rad[active_before] = torch.maximum(
                    max_abs_roll_rad[active_before], step_roll[active_before].abs()
                )
                max_abs_pitch_rad[active_before] = torch.maximum(
                    max_abs_pitch_rad[active_before], step_pitch[active_before].abs()
                )
                mech_power_sum_w[active_before] += step_mech_p[active_before]
                peak_mech_power_w[active_before] = torch.maximum(
                    peak_mech_power_w[active_before], step_mech_p[active_before]
                )

            if step % 10 == 0 or step == steps - 1:
                trajectory.append(
                    torch.stack(
                        (
                            root_xy[:, 0],
                            root_xy[:, 1],
                            speed,
                            yaw_rate,
                            pitch,
                            active.to(pitch.dtype),
                        ),
                        dim=-1,
                    ).cpu().numpy()
                )

            for key, mask in getattr(env._termination_reason_masks, "items", lambda: [])():
                newly_done = active_before & done & mask
                for index in torch.nonzero(newly_done, as_tuple=False).flatten().tolist():
                    failure_causes[index].add(key)

            if not bool(active.any()):
                break

    rows = []
    aggregate = defaultdict(
        lambda: {
            "attempts": 0,
            "successes": 0,
            "vx_mae_sum": 0.0,
            "yaw_mae_sum": 0.0,
            "lateral_drift_sum": 0.0,
            "mean_roll_sum": 0.0,
            "max_abs_roll_max": 0.0,
            "max_abs_pitch_max": 0.0,
            "mech_power_sum": 0.0,
            "peak_mech_power_max": 0.0,
        }
    )
    layout = tuple(tuple(row) for row in env._random_terrain_layout)
    for index, block_id in enumerate(block_ids):
        terrain = random_terrain_map.terrain_at_block(block_id, layout=layout)
        success = not bool(failed[index] or timed_out[index])
        lateral_drift = float(last_xy[index, 1] - start_xy[index, 1])
        forward = float(last_xy[index, 0] - start_xy[index, 0])
        steps_m = max(int(metric_steps[index]), 1)
        row = {
            "environment": index,
            "block_id": block_id,
            "terrain": terrain,
            "grade": float(random_terrain_map.GRADES[terrain]),
            "success": success,
            "failed": bool(failed[index]),
            "timed_out": bool(timed_out[index]),
            "completed_block": bool(completed_block[index]),
            "failure_causes": sorted(failure_causes[index]),
            "failure_reset_contact_forces_n": dict(
                sorted(failure_contact_forces[index].items())
            ),
            "failure_gimbal_mesh_clearance_m": (
                failure_gimbal_mesh_clearance[index]
            ),
            "minimum_gimbal_mesh_clearance_m": float(
                minimum_gimbal_mesh_clearance[index]
            ),
            "steps_survived": int(steps_survived[index]),
            "forward_m": forward,
            "lateral_drift_m": lateral_drift,
            "vx_mae_m_s": float(speed_error_sum[index] / steps_m),
            "mean_vx_m_s": float(speed_sum[index] / steps_m),
            "abs_yaw_rate_mae_rad_s": float(yaw_error_sum[index] / steps_m),
            "initial_pitch_deg": float(np.degrees(initial_pitch[index].item())),
            "pitch_range_deg": [
                float(np.degrees(pitch_min[index].item())),
                float(np.degrees(pitch_max[index].item())),
            ],
            "mean_roll_deg": float(np.degrees(roll_sum_rad[index].item() / steps_m)),
            "max_abs_roll_deg": float(np.degrees(max_abs_roll_rad[index].item())),
            "max_abs_pitch_deg": float(np.degrees(max_abs_pitch_rad[index].item())),
            "mean_mech_power_w": float(mech_power_sum_w[index].item() / steps_m),
            "peak_mech_power_w": float(peak_mech_power_w[index].item()),
        }
        rows.append(row)
        group = aggregate[terrain]
        group["attempts"] += 1
        group["successes"] += int(success)
        group["vx_mae_sum"] += row["vx_mae_m_s"]
        group["yaw_mae_sum"] += row["abs_yaw_rate_mae_rad_s"]
        group["lateral_drift_sum"] += abs(lateral_drift)
        group["mean_roll_sum"] += row["mean_roll_deg"]
        group["max_abs_roll_max"] = max(group["max_abs_roll_max"], row["max_abs_roll_deg"])
        group["max_abs_pitch_max"] = max(group["max_abs_pitch_max"], row["max_abs_pitch_deg"])
        group["mech_power_sum"] += row["mean_mech_power_w"]
        group["peak_mech_power_max"] = max(group["peak_mech_power_max"], row["peak_mech_power_w"])

    terrain_summary = {}
    for terrain, values in aggregate.items():
        attempts = values["attempts"]
        terrain_summary[terrain] = {
            "grade": float(random_terrain_map.GRADES[terrain]),
            "attempts": attempts,
            "successes": values["successes"],
            "success_rate": values["successes"] / attempts,
            "vx_mae_m_s": values["vx_mae_sum"] / attempts,
            "yaw_mae_rad_s": values["yaw_mae_sum"] / attempts,
            "mean_abs_lateral_drift_m": values["lateral_drift_sum"] / attempts,
            "mean_roll_deg": values["mean_roll_sum"] / attempts,
            "max_abs_roll_deg": values["max_abs_roll_max"],
            "max_abs_pitch_deg": values["max_abs_pitch_max"],
            "mean_mech_power_w": values["mech_power_sum"] / attempts,
            "peak_mech_power_w": values["peak_mech_power_max"],
        }

    overall_metrics = {
        "attempts": len(rows),
        "successes": sum(1 for r in rows if r["success"]),
        "success_rate": sum(1 for r in rows if r["success"]) / max(len(rows), 1),
        "vx_mae_m_s": float(np.mean([r["vx_mae_m_s"] for r in rows])),
        "yaw_mae_rad_s": float(np.mean([r["abs_yaw_rate_mae_rad_s"] for r in rows])),
        "mean_roll_deg": float(np.mean([r["mean_roll_deg"] for r in rows])),
        "mean_abs_roll_bias_deg": float(np.mean([abs(r["mean_roll_deg"]) for r in rows])),
        "max_abs_roll_deg": float(np.max([r["max_abs_roll_deg"] for r in rows])),
        "max_abs_pitch_deg": float(np.max([r["max_abs_pitch_deg"] for r in rows])),
        "mean_mech_power_w": float(np.mean([r["mean_mech_power_w"] for r in rows])),
        "peak_mech_power_w": float(np.max([r["peak_mech_power_w"] for r in rows])),
        "min_gimbal_mesh_clearance_m": float(np.min([r["minimum_gimbal_mesh_clearance_m"] for r in rows])),
    }

    np.savez_compressed(
        args.output / "trajectory.npz",
        samples=np.stack(trajectory),
        blocks=np.asarray(block_ids),
        start_xy=start_xy.cpu().numpy(),
    )
    dump_yaml(str(args.output / "env.yaml"), cfg)
    report.update(
        status="PASS" if all(row["success"] for row in rows) else "WARN_POLICY",
        checkpoint_iteration=int(saved["iter"]),
        gimbal_collision_local_aabb_m={
            "minimum": gimbal_local_min.tolist(),
            "maximum": gimbal_local_max.tolist(),
            "mesh_prims": gimbal_collision_meshes,
            "convex_hull_vertices": len(gimbal_local_hull_points),
        },
        observation_shapes={key: list(value.shape) for key, value in obs.items()},
        random_terrain_layout=random_terrain_map.health(env),
        overall_metrics=overall_metrics,
        terrain_summary=terrain_summary,
        rows=rows,
        finite_outputs=True,
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
