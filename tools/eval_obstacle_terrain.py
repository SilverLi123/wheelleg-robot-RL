"""Evaluate a frozen policy on 6-lane step/ledge obstacle terrains (400 Hz probe).

Tests zero-shot and fine-tuned stair/platform traversal on six isolated lanes
built from ``obstacle_terrain.ObstacleSpec``:
1. ``stairs_down_30mm``  (4 x 30 mm steps down, 0.35 m tread, 0.12 m drop)
2. ``stairs_down_60mm``  (4 x 60 mm steps down, 0.35 m tread, 0.24 m drop)
3. ``stairs_down_90mm``  (4 x 90 mm steps down, 0.35 m tread, 0.36 m drop)
4. ``platform_up_60mm``  (single 60 mm vertical step up)
5. ``platform_up_120mm`` (single 120 mm vertical step up)
6. ``platform_up_180mm`` (single 180 mm vertical step up)
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

LANE_X_START_M = -5.0
SPAWN_X_IN_LANE_M = 1.8

CANONICAL_SPECS_RAW = (
    ("stairs_down_30mm", "stairs_down", -14.0, dict(kind="stairs_down", height_m=0.030, width_m=3.0, approach_m=3.0, landing_m=5.0, step_count=4, tread_m=0.35, base_z_m=-0.20)),
    ("stairs_down_60mm", "stairs_down", -9.0, dict(kind="stairs_down", height_m=0.060, width_m=3.0, approach_m=3.0, landing_m=5.0, step_count=4, tread_m=0.35, base_z_m=-0.20)),
    ("stairs_down_90mm", "stairs_down", -4.0, dict(kind="stairs_down", height_m=0.090, width_m=3.0, approach_m=3.0, landing_m=5.0, step_count=4, tread_m=0.35, base_z_m=-0.20)),
    ("platform_up_60mm", "platform_up", 1.0, dict(kind="platform_up", height_m=0.060, width_m=3.0, approach_m=3.0, landing_m=6.5, base_z_m=-0.20)),
    ("platform_up_120mm", "platform_up", 6.0, dict(kind="platform_up", height_m=0.120, width_m=3.0, approach_m=3.0, landing_m=6.5, base_z_m=-0.20)),
    ("platform_up_180mm", "platform_up", 11.0, dict(kind="platform_up", height_m=0.180, width_m=3.0, approach_m=3.0, landing_m=6.5, base_z_m=-0.20)),
)

EXTENDED_SPECS_RAW = (
    ("stairs_down_30mm", "stairs_down", -19.0, dict(kind="stairs_down", height_m=0.030, width_m=3.0, approach_m=3.0, landing_m=5.0, step_count=4, tread_m=0.35, base_z_m=-0.20)),
    ("stairs_down_60mm", "stairs_down", -14.0, dict(kind="stairs_down", height_m=0.060, width_m=3.0, approach_m=3.0, landing_m=5.0, step_count=4, tread_m=0.35, base_z_m=-0.20)),
    ("stairs_down_90mm", "stairs_down", -9.0, dict(kind="stairs_down", height_m=0.090, width_m=3.0, approach_m=3.0, landing_m=5.0, step_count=4, tread_m=0.35, base_z_m=-0.20)),
    ("platform_up_25mm", "platform_up", -4.0, dict(kind="platform_up", height_m=0.025, width_m=3.0, approach_m=3.0, landing_m=6.5, base_z_m=-0.20)),
    ("platform_up_40mm", "platform_up", 1.0, dict(kind="platform_up", height_m=0.040, width_m=3.0, approach_m=3.0, landing_m=6.5, base_z_m=-0.20)),
    ("platform_up_60mm", "platform_up", 6.0, dict(kind="platform_up", height_m=0.060, width_m=3.0, approach_m=3.0, landing_m=6.5, base_z_m=-0.20)),
    ("platform_up_120mm", "platform_up", 11.0, dict(kind="platform_up", height_m=0.120, width_m=3.0, approach_m=3.0, landing_m=6.5, base_z_m=-0.20)),
    ("platform_up_180mm", "platform_up", 16.0, dict(kind="platform_up", height_m=0.180, width_m=3.0, approach_m=3.0, landing_m=6.5, base_z_m=-0.20)),
)

SUITE_CONFIGS = {
    "canonical": {"size": (16.0, 30.0), "num_rows": 3, "specs": CANONICAL_SPECS_RAW},
    "extended": {"size": (16.0, 42.0), "num_rows": 4, "specs": EXTENDED_SPECS_RAW},
}


def _build_obstacle_cases(suite="canonical"):
    from agent_tasks.direct.wheelbipe.wheelbipe_own.obstacle_terrain import ObstacleSpec

    return tuple(
        {
            "name": name,
            "group": group,
            "y_start_m": y_start,
            "spec": ObstacleSpec(**spec_kwargs),
        }
        for name, group, y_start, spec_kwargs in SUITE_CONFIGS[suite]["specs"]
    )


def _make_combined_obstacle_terrain(suite):
    def _combined_obstacle_terrain(_difficulty, _cfg):
        import trimesh
        from agent_tasks.direct.wheelbipe.wheelbipe_own.obstacle_terrain import mesh_data

        meshes = []
        for case in _build_obstacle_cases(suite):
            vertices, faces = mesh_data(case["spec"])
            translated = np.asarray(vertices, dtype=np.float64)
            translated[:, 0] += LANE_X_START_M
            translated[:, 1] += case["y_start_m"]
            meshes.append(
                trimesh.Trimesh(
                    vertices=translated,
                    faces=np.asarray(faces, dtype=np.int64),
                    process=False,
                )
            )
        combined = trimesh.util.concatenate(meshes)
        return [combined], np.zeros(3, dtype=np.float64)

    return _combined_obstacle_terrain


_GROUND_Z_CACHE = {}


def _obstacle_ground_z_torch(_env, local_xy, suite="canonical"):
    import torch

    device = local_xy.device
    dtype = local_xy.dtype
    key = (suite, str(device), str(dtype))
    cached = _GROUND_Z_CACHE.get(key)
    if cached is None:
        cases = _build_obstacle_cases(suite)
        cached = (
            torch.tensor(
                [c["y_start_m"] + c["spec"].width_m / 2.0 for c in cases],
                device=device,
                dtype=dtype,
            ),
            torch.tensor(
                [c["spec"].height_m for c in cases],
                device=device,
                dtype=dtype,
            ),
            torch.tensor(
                [c["spec"].kind == "platform_up" for c in cases],
                device=device,
                dtype=torch.bool,
            ),
            torch.tensor(
                [3.0 + i * 0.35 for i in range(4)],
                device=device,
                dtype=dtype,
            ),
        )
        _GROUND_Z_CACHE[key] = cached
    y_centers, lane_heights, is_platform, step_edges = cached

    lx = local_xy[..., 0]
    ly = local_xy[..., 1]
    x_m = lx - LANE_X_START_M
    lane_idx = torch.argmin((ly.unsqueeze(-1) - y_centers).abs(), dim=-1)

    h = lane_heights[lane_idx]
    plat = is_platform[lane_idx]

    # Exact match to ObstacleSpec.ground_height():
    # platform_up: 0.0 on approach (< 3.0), height_m on landing (>= 3.0)
    plat_z = torch.where(x_m < 3.0, torch.zeros_like(h), h)
    # stairs_down: remaining steps * height_m
    remaining = (x_m.unsqueeze(-1) < step_edges).sum(dim=-1).to(dtype)
    stairs_z = remaining * h
    return torch.where(plat, plat_z, stairs_z)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", default="random_terrain_relcritic_20261113")
parser.add_argument("--suite", choices=("canonical", "extended"), default="canonical")
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--seed", type=int, default=71903)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--speed", type=float, default=0.65)
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
os.chdir(args.output)
started = time.monotonic()
report = {
    "status": "INITIALIZING",
    "profile": args.profile,
    "suite": args.suite,
    "seed": args.seed,
    "seconds": args.seconds,
    "speed_m_s": args.speed,
    "repeats": args.repeats,
    "checkpoint": str(args.checkpoint),
    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
}
app = None
original_random_heights = None

try:
    app = AppLauncher(args, fast_shutdown=True).app
    suite_cfg = SUITE_CONFIGS[args.suite]
    OBSTACLE_CASES = _build_obstacle_cases(args.suite)
    import torch
    from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg
    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
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
    cfg.scene.num_envs = len(OBSTACLE_CASES) * args.repeats

    prepare_checkpoint(cfg, None, str(args.checkpoint))

    cfg.terrain = cfg.terrain.copy()
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=20261113,
        size=suite_cfg["size"],
        border_width=0.0,
        num_rows=suite_cfg["num_rows"],
        num_cols=6,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        color_scheme="height",
        sub_terrains={
            "obstacle_lanes": MeshPlaneTerrainCfg(
                proportion=1.0,
                function=_make_combined_obstacle_terrain(args.suite),
            )
        },
    )
    cfg.scene.terrain = cfg.terrain.copy()
    spawn_xy_yaw = [
        (
            LANE_X_START_M + SPAWN_X_IN_LANE_M,
            case["y_start_m"] + case["spec"].width_m / 2.0,
            0.0,
        )
        for _ in range(args.repeats)
        for case in OBSTACLE_CASES
    ]
    case_indices = [
        idx for _ in range(args.repeats) for idx in range(len(OBSTACLE_CASES))
    ]
    cfg.experiment_profile_settings = dict(cfg.experiment_profile_settings)
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

    original_random_heights = random_terrain_map._heights
    random_terrain_map._heights = lambda _env, xy: _obstacle_ground_z_torch(
        _env, xy, suite=args.suite
    )

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
    latest_step_metrics = [None]

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
            relative_z = T(_obstacle_ground_z_torch(env, local_xy, suite=args.suite))
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
        roll_s, pitch_s, _ = euler_xyz_from_quat(T(env.robot.data.root_quat_w))
        roll_s = wrap_to_pi(roll_s)
        pitch_s = wrap_to_pi(pitch_s)
        vz_b_s = T(env.robot.data.root_lin_vel_b)[:, 2]
        mech_p_s = (
            T(env.robot.data.applied_torque)[:, env._actuate_idx]
            * T(env.robot.data.joint_vel)[:, env._actuate_idx]
        ).abs().sum(dim=-1)
        latest_step_metrics[0] = (
            roll_s.clone(),
            pitch_s.clone(),
            vz_b_s.clone(),
            mech_p_s.clone(),
        )
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
    max_survived_x_m = (start_xy[:, 0] - origin_xy[:, 0]) - LANE_X_START_M
    peak_initial_tire_penetration_m = float(
        getattr(env, "_random_terrain_peak_penetration_m", 0.0)
    )

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
    max_abs_pitch_rad = torch.zeros(count, device=env.device)
    max_abs_roll_rad = torch.zeros(count, device=env.device)
    roll_sum_rad = torch.zeros(count, device=env.device)
    max_upward_vz_b = torch.zeros(count, device=env.device)
    mech_power_sum_w = torch.zeros(count, device=env.device)
    peak_mech_power_w = torch.zeros(count, device=env.device)

    with torch.inference_mode():
        for step in range(steps):
            action = actor(obs["policy"])
            latest_reset_contact_forces[0] = None
            latest_gimbal_clearance[0] = None
            latest_base_clearance[0] = None
            latest_step_metrics[0] = None
            obs, reward, done, timeout, _ = env.step(action)
            active_before = active.clone()
            active &= ~done & ~timeout

            speed = T(env.robot.data.root_lin_vel_b)[:, 0]
            yaw_rate = T(env.robot.data.root_ang_vel_b)[:, 2]
            root_xy = T(env.robot.data.root_pos_w)[:, :2]
            last_xy[active_before] = root_xy[active_before]
            cur_x_m = (root_xy[:, 0] - origin_xy[:, 0]) - LANE_X_START_M
            survived_this_step = active_before & ~done
            max_survived_x_m[survived_this_step] = torch.maximum(
                max_survived_x_m[survived_this_step], cur_x_m[survived_this_step]
            )
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
                step_roll, step_pitch, step_vz_b, step_mech_p = latest_step_metrics[0]
                max_abs_pitch_rad[active_before] = torch.maximum(
                    max_abs_pitch_rad[active_before], step_pitch[active_before].abs()
                )
                max_abs_roll_rad[active_before] = torch.maximum(
                    max_abs_roll_rad[active_before], step_roll[active_before].abs()
                )
                roll_sum_rad[active_before] += step_roll[active_before]
                max_upward_vz_b[active_before] = torch.maximum(
                    max_upward_vz_b[active_before], step_vz_b[active_before]
                )
                mech_power_sum_w[active_before] += step_mech_p[active_before]
                peak_mech_power_w[active_before] = torch.maximum(
                    peak_mech_power_w[active_before], step_mech_p[active_before]
                )

    forward_travel_m = last_xy[:, 0] - start_xy[:, 0]
    denom = metric_steps.clamp_min(1).to(torch.float32)
    vx_mae = speed_error_sum / denom
    vx_mean = speed_sum / denom
    wz_mae = yaw_error_sum / denom
    max_abs_pitch_deg = torch.rad2deg(max_abs_pitch_rad)
    max_abs_roll_deg = torch.rad2deg(max_abs_roll_rad)
    mean_roll_deg = torch.rad2deg(roll_sum_rad / denom)
    mean_mech_power_w = mech_power_sum_w / denom

    by_case = defaultdict(list)
    for env_id, case_idx in enumerate(case_indices):
        case = OBSTACLE_CASES[case_idx]
        spec = case["spec"]
        final_x_m = float((last_xy[env_id, 0] - origin_xy[env_id, 0]).item()) - LANE_X_START_M
        max_x = float(max_survived_x_m[env_id].item())
        ep_failed = bool(failed[env_id].item())
        if spec.kind == "stairs_down":
            steps_cleared = sum(
                1 for i in range(spec.step_count) if max_x >= spec.approach_m + i * spec.tread_m
            )
            reached_landing = (not ep_failed) and (
                final_x_m >= spec.approach_m + spec.step_count * spec.tread_m
            )
        else:
            steps_cleared = 1 if (not ep_failed and max_x >= spec.approach_m + 0.30) else 0
            reached_landing = (not ep_failed) and (final_x_m >= spec.approach_m + 1.0)
        by_case[case["name"]].append(
            {
                "env_id": env_id,
                "group": case["group"],
                "height_m": spec.height_m,
                "failed": ep_failed,
                "reached_landing": bool(reached_landing),
                "steps_cleared": int(steps_cleared),
                "max_survived_x_m": round(max_x, 3),
                "final_x_m": round(final_x_m, 3),
                "failure_causes": sorted(failure_causes[env_id]),
                "failure_local_xy": failure_local_xy[env_id],
                "failure_reset_contact_forces_n": failure_contact_forces[env_id],
                "survived_s": float(steps_survived[env_id]) * float(env.step_dt),
                "forward_travel_m": float(forward_travel_m[env_id]),
                "vx_mean_m_s": float(vx_mean[env_id]),
                "vx_mae_m_s": float(vx_mae[env_id]),
                "wz_mae_rad_s": float(wz_mae[env_id]),
                "max_abs_pitch_deg": float(max_abs_pitch_deg[env_id]),
                "max_abs_roll_deg": float(max_abs_roll_deg[env_id]),
                "mean_roll_deg": float(mean_roll_deg[env_id]),
                "max_upward_vz_b_m_s": float(max_upward_vz_b[env_id]),
                "mean_mech_power_w": float(mean_mech_power_w[env_id]),
                "peak_mech_power_w": float(peak_mech_power_w[env_id]),
                "min_gimbal_clearance_m": float(min_gimbal_clearance[env_id]),
                "min_base_clearance_m": float(min_base_clearance[env_id]),
            }
        )

    summary_by_case = {}
    for case in OBSTACLE_CASES:
        name = case["name"]
        rows = by_case[name]
        summary_by_case[name] = {
            "group": case["group"],
            "height_m": case["spec"].height_m,
            "episodes": len(rows),
            "failures": sum(1 for r in rows if r["failed"]),
            "reached_landing_count": sum(1 for r in rows if r["reached_landing"]),
            "max_steps_cleared": max(r["steps_cleared"] for r in rows),
            "survival_rate": sum(0 if r["failed"] else 1 for r in rows) / len(rows),
            "forward_travel_mean_m": float(np.mean([r["forward_travel_m"] for r in rows])),
            "vx_mean_m_s": float(np.mean([r["vx_mean_m_s"] for r in rows])),
            "vx_mae_m_s": float(np.mean([r["vx_mae_m_s"] for r in rows])),
            "wz_mae_rad_s": float(np.mean([r["wz_mae_rad_s"] for r in rows])),
            "max_abs_pitch_deg": float(np.max([r["max_abs_pitch_deg"] for r in rows])),
            "max_abs_roll_deg": float(np.max([r["max_abs_roll_deg"] for r in rows])),
            "mean_roll_deg": float(np.mean([r["mean_roll_deg"] for r in rows])),
            "max_upward_vz_b_m_s": float(np.max([r["max_upward_vz_b_m_s"] for r in rows])),
            "mean_mech_power_w": float(np.mean([r["mean_mech_power_w"] for r in rows])),
            "peak_mech_power_w": float(np.max([r["peak_mech_power_w"] for r in rows])),
            "min_gimbal_clearance_m": float(np.min([r["min_gimbal_clearance_m"] for r in rows])),
            "min_base_clearance_m": float(np.min([r["min_base_clearance_m"] for r in rows])),
            "failure_causes": sorted({c for r in rows for c in r["failure_causes"]}),
            "failure_contact_bodies": sorted(
                {b for r in rows for b in r["failure_reset_contact_forces_n"]}
            ),
            "episodes_detail": rows,
        }

    stairs_cases = [v for v in summary_by_case.values() if v["group"] == "stairs_down"]
    platform_cases = [v for v in summary_by_case.values() if v["group"] == "platform_up"]
    all_cases = list(summary_by_case.values())
    report.update(
        status="PASS" if all(v["failures"] == 0 for v in all_cases) else "FAIL",
        wall_seconds=time.monotonic() - started,
        physics_dt=float(cfg.sim.dt),
        policy_dt=float(env.step_dt),
        peak_initial_tire_penetration_m=peak_initial_tire_penetration_m,
        summary={
            "cases": len(all_cases),
            "total_episodes": sum(v["episodes"] for v in all_cases),
            "total_failures": sum(v["failures"] for v in all_cases),
            "total_reached_landing": sum(v["reached_landing_count"] for v in all_cases),
            "stairs_pass_cases": sum(1 for v in stairs_cases if v["failures"] == 0),
            "stairs_total_episodes": sum(v["episodes"] for v in stairs_cases),
            "stairs_failures": sum(v["failures"] for v in stairs_cases),
            "platform_pass_cases": sum(1 for v in platform_cases if v["failures"] == 0),
            "platform_total_episodes": sum(v["episodes"] for v in platform_cases),
            "platform_failures": sum(v["failures"] for v in platform_cases),
            "vx_mae_mean_m_s": float(np.mean([v["vx_mae_m_s"] for v in all_cases])),
            "wz_mae_mean_rad_s": float(np.mean([v["wz_mae_rad_s"] for v in all_cases])),
            "stairs_max_abs_pitch_deg": float(np.max([v["max_abs_pitch_deg"] for v in stairs_cases])),
            "stairs_max_abs_roll_deg": float(np.max([v["max_abs_roll_deg"] for v in stairs_cases])),
            "stairs_mean_roll_abs_deg": float(np.mean([abs(v["mean_roll_deg"]) for v in stairs_cases])),
            "stairs_max_upward_vz_b_m_s": float(np.max([v["max_upward_vz_b_m_s"] for v in stairs_cases])),
            "mean_mech_power_w": float(np.mean([v["mean_mech_power_w"] for v in all_cases])),
            "peak_mech_power_w": float(np.max([v["peak_mech_power_w"] for v in all_cases])),
            "min_gimbal_clearance_m": float(np.min([v["min_gimbal_clearance_m"] for v in all_cases])),
            "min_base_clearance_m": float(np.min([v["min_base_clearance_m"] for v in all_cases])),
        },
        by_case=summary_by_case,
    )
    dump_yaml(str(args.output / "env.yaml"), cfg)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
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
    if original_random_heights is not None:
        random_terrain_map._heights = original_random_heights
    if app is not None:
        app.close()
