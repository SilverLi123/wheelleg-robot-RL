"""Evaluate a frozen policy on inter-block transitions of the 3x3 random-terrain map.

This probe tests two groups of transitions on the connected 3x3 map (seed 20261112):
1. `valid_transitions` (8 cases):
   - 6 longitudinal X-seam transitions along the serpentine path (`0->1`, `1->2`,
     `5->4`, `4->3`, `6->7`, `7->8`), starting on the exit slope (`block_x = +-4.5 m`)
     so the robot traverses the source exit ramp, the zero-height seam at
     `block_x = +-8.0 m`, and the target entry ramp.
   - 2 flat-corridor row crossings (`row_corridor_2_to_5`, `row_corridor_3_to_6`)
     at `block_x = +-7.2 m` where adjacent rows meet at zero height.
2. `crest_wall_probes` (2 cases):
   - `row_crest_2_to_5`: crossing from block 2 (`up14`) crest (`block_x = 0.0 m`)
     across the 1.5 m Y-taper (-56% lateral drop) into block 5 (`flat`).
   - `row_crest_3_to_6`: exact reproduction of the step 11,759 continuous-route
     failure (`local_x = -17.31 m`, `block_x = -1.31 m`), crossing from block 3
     (`up06`) crest across the Y-taper (-24% drop then +72% / 35.8 deg wall)
     into block 6 (`up18`) crest.
"""

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

TRANSITION_CASES = (
    # 6 longitudinal block-to-block transitions along the serpentine path
    {
        "name": "col_0_to_1_down10_to_up10",
        "group": "valid_transition",
        "source_block": 0,
        "target_block": 1,
        "local_x": -11.5,
        "local_y": -12.0,
        "yaw_rad": 0.0,
        "description": "Exit down10 (+10% climb) -> X-seam -> climb up10 (+10%)",
    },
    {
        "name": "col_1_to_2_up10_to_up14",
        "group": "valid_transition",
        "source_block": 1,
        "target_block": 2,
        "local_x": 4.5,
        "local_y": -12.0,
        "yaw_rad": 0.0,
        "description": "Exit up10 (-10% descent) -> X-seam -> climb up14 (+14%)",
    },
    {
        "name": "col_5_to_4_flat_to_down14",
        "group": "valid_transition",
        "source_block": 5,
        "target_block": 4,
        "local_x": 10.5,
        "local_y": 0.0,
        "yaw_rad": math.pi,
        "description": "Exit flat -> -X seam -> descend into down14 (-14%)",
    },
    {
        "name": "col_4_to_3_down14_to_up06",
        "group": "valid_transition",
        "source_block": 4,
        "target_block": 3,
        "local_x": -4.5,
        "local_y": 0.0,
        "yaw_rad": math.pi,
        "description": "Exit down14 (+14% climb) -> -X seam -> climb up06 (+6%)",
    },
    {
        "name": "col_6_to_7_up18_to_down06",
        "group": "valid_transition",
        "source_block": 6,
        "target_block": 7,
        "local_x": -11.5,
        "local_y": 12.0,
        "yaw_rad": 0.0,
        "description": "Exit up18 (-18% steep descent) -> X-seam -> descend down06 (-6%)",
    },
    {
        "name": "col_7_to_8_down06_to_down18",
        "group": "valid_transition",
        "source_block": 7,
        "target_block": 8,
        "local_x": 4.5,
        "local_y": 12.0,
        "yaw_rad": 0.0,
        "description": "Exit down06 (+6% climb) -> X-seam -> descend down18 (-18%)",
    },
    # 2 flat-corridor row crossings (|block_x| = 7.2 m, zero-height corridor)
    {
        "name": "row_corridor_2_to_5",
        "group": "valid_transition",
        "source_block": 2,
        "target_block": 5,
        "local_x": 23.2,
        "local_y": -8.0,
        "yaw_rad": math.pi / 2.0,
        "description": "Row 0->1 crossing at flat run-off corridor (block_x = +7.2 m)",
    },
    {
        "name": "row_corridor_3_to_6",
        "group": "valid_transition",
        "source_block": 3,
        "target_block": 6,
        "local_x": -23.2,
        "local_y": 4.0,
        "yaw_rad": math.pi / 2.0,
        "description": "Row 1->2 crossing at flat run-off corridor (block_x = -7.2 m)",
    },
    # 2 pathological crest-to-crest lateral Y-taper wall crossings
    {
        "name": "row_crest_2_to_5",
        "group": "crest_wall_probe",
        "source_block": 2,
        "target_block": 5,
        "local_x": 16.0,
        "local_y": -8.0,
        "yaw_rad": math.pi / 2.0,
        "description": "Lateral Y-taper drop (-56%) off block 2 (up14) crest into block 5 (flat)",
    },
    {
        "name": "row_crest_3_to_6",
        "group": "crest_wall_probe",
        "source_block": 3,
        "target_block": 6,
        "local_x": -17.31,
        "local_y": 4.0,
        "yaw_rad": math.pi / 2.0,
        "description": "Step 11,759 failure point: block 3 (up06) crest -> +72% Y-taper wall on block 6 (up18)",
    },
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", default="random_terrain_20261112")
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--seed", type=int, default=71903)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--speed", type=float, default=0.65)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.repeats < 1:
    parser.error("--repeats must be positive")
if not 0.0 <= args.speed <= 0.65:
    parser.error("--speed must stay within the training command range [0, 0.65]")

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
    "speed_m_s": args.speed,
    "repeats": args.repeats,
    "checkpoint": str(args.checkpoint),
    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
}
app = None

try:
    app = AppLauncher(args, fast_shutdown=True).app
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat
    from isaaclab.utils.io import dump_yaml
    from agent_tasks.direct.wheelbipe.wheelbipe_own import random_terrain_map
    from agent_tasks.direct.wheelbipe.wheelbipe_own.experiment import (
        WheelbipeOwnExperimentEnv,
        WheelbipeOwnExperimentPlayCfg,
    )
    from agent_world.sim620_compat import torch_data as T
    from scripts.rsl_rl.checkpoint_utils import prepare_checkpoint

    cfg = WheelbipeOwnExperimentPlayCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = len(TRANSITION_CASES) * args.repeats

    prepare_checkpoint(cfg, None, str(args.checkpoint))
    spawn_xy_yaw = [
        (case["local_x"], case["local_y"], case["yaw_rad"])
        for _ in range(args.repeats)
        for case in TRANSITION_CASES
    ]
    case_indices = [
        idx for _ in range(args.repeats) for idx in range(len(TRANSITION_CASES))
    ]
    cfg.experiment_profile_settings = dict(cfg.experiment_profile_settings)
    if float(cfg.experiment_profile_settings.get("random_terrain_grade_scale", 1.0)) != 1.0:
        cfg.experiment_profile_settings["random_terrain_grade_scale"] = 1.0
        random_terrain_map.configure(cfg)
    cfg.experiment_profile_settings["evaluation_spawn_xy_yaw"] = spawn_xy_yaw
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

    reset_contact_names = [
        env.contact_sensor.body_names[index]
        for index in env._reset_contact_link_idx
    ]
    gimbal_body_idx, _ = env._native_robot.find_bodies(
        "gimbal_yaw_link", preserve_order=True
    )
    base_body_idx, _ = env._native_robot.find_bodies(
        "base_link", preserve_order=True
    )

    from pxr import Usd, UsdGeom

    def collision_mesh_local_hull(body_name):
        from scipy.spatial import ConvexHull

        time_code = Usd.TimeCode.Default()
        body = env.sim.stage.GetPrimAtPath(
            f"/World/envs/env_0/Robot/{body_name}"
        )
        if not body.IsValid():
            raise RuntimeError(f"Robot body prim is missing: {body_name}")
        body_world = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(time_code)
        body_inverse = body_world.GetInverse()
        points_body = []
        for prim in Usd.PrimRange(body):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if "collisions" not in prim.GetPath().pathString.split("/"):
                continue
            mesh = UsdGeom.Mesh(prim)
            mesh_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
            local_to_body = mesh_world * body_inverse
            points = mesh.GetPointsAttr().Get(time_code)
            if not points:
                continue
            points_body.extend(local_to_body.Transform(point) for point in points)
        points_np = np.asarray(
            [tuple(float(value) for value in point) for point in points_body],
            dtype=np.float64,
        )
        points_np = np.unique(points_np, axis=0)
        hull = ConvexHull(points_np)
        return points_np[hull.vertices]

    gimbal_hull_points = collision_mesh_local_hull("gimbal_yaw_link")
    base_hull_points = collision_mesh_local_hull("base_link")
    original_get_dones = env._get_dones
    latest_reset_contact_forces = [None]
    latest_gimbal_clearance = [None]
    latest_base_clearance = [None]

    def body_mesh_clearance_m(body_idx, hull_points):
        body_pos_w = T(env._native_robot.data.body_pos_w)
        body_quat_w = T(env._native_robot.data.body_quat_w)
        origin = T(env.terrain.env_origins)
        positions = body_pos_w[:, body_idx].cpu().numpy()
        quaternions = body_quat_w[:, body_idx].cpu().numpy()[:, [3, 0, 1, 2]]
        origins = origin.cpu().numpy()
        points_per_body = len(hull_points)

        def ground_at_world_xy(world_xy):
            origin_xy = np.repeat(origins[:, :2], points_per_body, axis=0)
            local_xy_np = world_xy - origin_xy
            local_xy = torch.as_tensor(
                local_xy_np, dtype=torch.float32, device=env.device
            )
            relative_z = T(random_terrain_map._heights(env, local_xy))
            return relative_z.cpu().numpy() + np.repeat(origins[:, 2], points_per_body)

        clearance = random_terrain_map.collision_points_minimum_clearance(
            body_pos_w=positions,
            body_quat_w=quaternions,
            local_points=hull_points,
            height_at_local_xy=ground_at_world_xy,
        )
        return torch.as_tensor(clearance, dtype=body_pos_w.dtype, device=env.device)

    def capture_get_dones():
        terminated, timed_out = original_get_dones()
        latest_reset_contact_forces[0] = torch.linalg.vector_norm(
            T(env.contact_sensor.data.net_forces_w)[:, env._reset_contact_link_idx],
            dim=-1,
        ).clone()
        latest_gimbal_clearance[0] = body_mesh_clearance_m(
            gimbal_body_idx[0], gimbal_hull_points
        ).clone()
        latest_base_clearance[0] = body_mesh_clearance_m(
            base_body_idx[0], base_hull_points
        ).clone()
        return terminated, timed_out

    env._get_dones = capture_get_dones

    count = cfg.scene.num_envs
    command = torch.zeros((count, 3), device=env.device)
    command[:, 0] = args.speed
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
    origin_xy = T(env.terrain.env_origins)[:, :2].clone()
    start_xy = T(env.robot.data.root_pos_w)[:, :2].clone()
    last_xy = start_xy.clone()
    _, initial_pitch, initial_yaw = euler_xyz_from_quat(T(env.robot.data.root_quat_w))
    heading_vec = torch.stack((initial_yaw.cos(), initial_yaw.sin()), dim=-1)

    active = torch.ones(count, dtype=torch.bool, device=env.device)
    failed = torch.zeros_like(active)
    steps_survived = torch.zeros(count, dtype=torch.long, device=env.device)
    speed_error_sum = torch.zeros(count, device=env.device)
    speed_sum = torch.zeros(count, device=env.device)
    yaw_error_sum = torch.zeros(count, device=env.device)
    metric_steps = torch.zeros(count, dtype=torch.long, device=env.device)
    warmup_steps = max(round(0.5 / env.step_dt), 0)
    steps = max(round(args.seconds / env.step_dt), 1)
    failure_causes = [set() for _ in range(count)]
    failure_contact_forces = [dict() for _ in range(count)]
    failure_local_xy = [None] * count
    min_gimbal_clearance = torch.full((count,), float("inf"), device=env.device)
    min_base_clearance = torch.full((count,), float("inf"), device=env.device)

    with torch.inference_mode():
        for step in range(steps):
            action = actor(obs["policy"])
            latest_reset_contact_forces[0] = None
            latest_gimbal_clearance[0] = None
            latest_base_clearance[0] = None
            obs, reward, done, timeout, _ = env.step(action)
            active_before = active.clone()
            active &= ~done & ~timeout

            speed = T(env.robot.data.root_lin_vel_b)[:, 0]
            yaw_rate = T(env.robot.data.root_ang_vel_b)[:, 2]
            root_xy = T(env.robot.data.root_pos_w)[:, :2]
            last_xy[active_before] = root_xy[active_before]
            steps_survived[active_before] += 1
            failed |= done & active_before

            reset_forces = latest_reset_contact_forces[0]
            step_gimbal = latest_gimbal_clearance[0]
            step_base = latest_base_clearance[0]
            min_gimbal_clearance[active_before] = torch.minimum(
                min_gimbal_clearance[active_before], step_gimbal[active_before]
            )
            min_base_clearance[active_before] = torch.minimum(
                min_base_clearance[active_before], step_base[active_before]
            )
            for index in torch.nonzero(done & active_before, as_tuple=False).flatten().tolist():
                loc = (last_xy[index] - origin_xy[index]).tolist()
                failure_local_xy[index] = [round(float(loc[0]), 3), round(float(loc[1]), 3)]
                for body_name, force in zip(reset_contact_names, reset_forces[index].tolist()):
                    if force > 0.1:
                        failure_contact_forces[index][body_name] = max(
                            round(float(force), 2),
                            failure_contact_forces[index].get(body_name, 0.0),
                        )
            for reason, mask in env._termination_reason_masks.items():
                for index in torch.nonzero(
                    mask & done & active_before, as_tuple=False
                ).flatten().tolist():
                    failure_causes[index].add(reason)

            if step >= warmup_steps:
                speed_error_sum[active_before] += (speed[active_before] - args.speed).abs()
                speed_sum[active_before] += speed[active_before]
                yaw_error_sum[active_before] += yaw_rate[active_before].abs()
                metric_steps[active_before] += 1

    displacement_vec = last_xy - start_xy
    forward_travel_m = (displacement_vec * heading_vec).sum(dim=-1)
    denom = metric_steps.clamp_min(1).to(torch.float32)
    vx_mae = speed_error_sum / denom
    vx_mean = speed_sum / denom
    wz_mae = yaw_error_sum / denom

    by_case = defaultdict(list)
    for env_id, case_idx in enumerate(case_indices):
        case = TRANSITION_CASES[case_idx]
        end_loc = (last_xy[env_id] - origin_xy[env_id]).tolist()
        r_end, c_end, bx_end, by_end = random_terrain_map.world_to_block(
            float(np.clip(end_loc[0], -24.0, 24.0)),
            float(np.clip(end_loc[1], -18.0, 18.0)),
        )
        end_block = r_end * 3 + c_end
        reached_target = (end_block == case["target_block"]) and (not bool(failed[env_id]))
        by_case[case["name"]].append(
            {
                "env_id": env_id,
                "group": case["group"],
                "source_block": case["source_block"],
                "target_block": case["target_block"],
                "end_block": int(end_block),
                "reached_target_block": bool(reached_target),
                "failed": bool(failed[env_id]),
                "failure_causes": sorted(failure_causes[env_id]),
                "failure_local_xy": failure_local_xy[env_id],
                "failure_reset_contact_forces_n": failure_contact_forces[env_id],
                "survived_s": float(steps_survived[env_id]) * float(env.step_dt),
                "forward_travel_m": float(forward_travel_m[env_id]),
                "vx_mean_m_s": float(vx_mean[env_id]),
                "vx_mae_m_s": float(vx_mae[env_id]),
                "wz_mae_rad_s": float(wz_mae[env_id]),
                "min_gimbal_clearance_m": float(min_gimbal_clearance[env_id]),
                "min_base_clearance_m": float(min_base_clearance[env_id]),
            }
        )

    summary_by_case = {}
    for case in TRANSITION_CASES:
        name = case["name"]
        rows = by_case[name]
        summary_by_case[name] = {
            "group": case["group"],
            "description": case["description"],
            "source_block": case["source_block"],
            "target_block": case["target_block"],
            "episodes": len(rows),
            "failures": sum(1 for r in rows if r["failed"]),
            "reached_target_count": sum(1 for r in rows if r["reached_target_block"]),
            "survival_rate": sum(0 if r["failed"] else 1 for r in rows) / len(rows),
            "forward_travel_mean_m": float(np.mean([r["forward_travel_m"] for r in rows])),
            "vx_mean_m_s": float(np.mean([r["vx_mean_m_s"] for r in rows])),
            "vx_mae_m_s": float(np.mean([r["vx_mae_m_s"] for r in rows])),
            "wz_mae_rad_s": float(np.mean([r["wz_mae_rad_s"] for r in rows])),
            "min_gimbal_clearance_m": float(np.min([r["min_gimbal_clearance_m"] for r in rows])),
            "min_base_clearance_m": float(np.min([r["min_base_clearance_m"] for r in rows])),
            "failure_causes": sorted({c for r in rows for c in r["failure_causes"]}),
            "failure_contact_bodies": sorted(
                {b for r in rows for b in r["failure_reset_contact_forces_n"]}
            ),
            "episodes_detail": rows,
        }

    valid_cases = [
        v for v in summary_by_case.values() if v["group"] == "valid_transition"
    ]
    crest_cases = [
        v for v in summary_by_case.values() if v["group"] == "crest_wall_probe"
    ]
    report.update(
        status="PASS" if all(v["failures"] == 0 for v in valid_cases) else "FAIL",
        wall_seconds=time.monotonic() - started,
        physics_dt=float(cfg.sim.dt),
        policy_dt=float(env.step_dt),
        valid_transitions_summary={
            "cases": len(valid_cases),
            "total_episodes": sum(v["episodes"] for v in valid_cases),
            "total_failures": sum(v["failures"] for v in valid_cases),
            "pass_cases": sum(1 for v in valid_cases if v["failures"] == 0),
            "vx_mae_mean_m_s": float(np.mean([v["vx_mae_m_s"] for v in valid_cases])),
            "forward_travel_mean_m": float(
                np.mean([v["forward_travel_mean_m"] for v in valid_cases])
            ),
            "min_gimbal_clearance_m": float(
                np.min([v["min_gimbal_clearance_m"] for v in valid_cases])
            ),
            "min_base_clearance_m": float(
                np.min([v["min_base_clearance_m"] for v in valid_cases])
            ),
        },
        crest_wall_probes_summary={
            "cases": len(crest_cases),
            "total_episodes": sum(v["episodes"] for v in crest_cases),
            "total_failures": sum(v["failures"] for v in crest_cases),
            "pass_cases": sum(1 for v in crest_cases if v["failures"] == 0),
            "vx_mae_mean_m_s": float(np.mean([v["vx_mae_m_s"] for v in crest_cases])),
            "min_gimbal_clearance_m": float(
                np.min([v["min_gimbal_clearance_m"] for v in crest_cases])
            ),
            "min_base_clearance_m": float(
                np.min([v["min_base_clearance_m"] for v in crest_cases])
            ),
        },
        by_case=summary_by_case,
    )
    dump_yaml(str(args.output / "env.yaml"), cfg)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["valid_transitions_summary"], indent=2))
    print(json.dumps(report["crest_wall_probes_summary"], indent=2))
except Exception as exc:
    report.update(
        status="ERROR",
        error=f"{type(exc).__name__}: {exc}",
        traceback=traceback.format_exc(),
        wall_seconds=time.monotonic() - started,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    raise
finally:
    if app is not None:
        app.close()
