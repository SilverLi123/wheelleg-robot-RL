"""One fixed, connected map containing all nine mixed-slope surfaces.

The map is a 3 by 3 arrangement of the flat, 6/10/14/18 percent uphill
and 6/10/14/18 percent downhill patches.  The patch order is shuffled from
a fixed seed.  Every patch starts and ends at zero height, so adjacent
patches meet without a vertical step even though their order is random.

Training environments receive copies of this same map.  Their robots are
isolated by Isaac Lab's environment collision filters; the reset helper below
samples a patch and a physical position inside every environment's copy.
"""

from __future__ import annotations

import math
import json

import numpy as np


TERRAIN_TYPES = (
    "flat",
    "up06",
    "up10",
    "up14",
    "up18",
    "down06",
    "down10",
    "down14",
    "down18",
)
GRADES = {
    "flat": 0.00,
    "up06": 0.06,
    "up10": 0.10,
    "up14": 0.14,
    "up18": 0.18,
    "down06": -0.06,
    "down10": -0.10,
    "down14": -0.14,
    "down18": -0.18,
}

# Patch sizes keep the terrain triangles readable while giving a wheel-legged
# robot enough longitudinal run-up and slope distance.
BLOCK_LENGTH_M = 16.0
BLOCK_WIDTH_M = 12.0
MAP_ROWS = 3
MAP_COLS = 3
MAP_SIZE_M = (BLOCK_LENGTH_M * MAP_COLS, BLOCK_WIDTH_M * MAP_ROWS)

# The complete map is repeated over the terrain generator grid.  A grid has
# 60 cells for 4096 environments, matching the existing mixed-slope setup:
# environments may share terrain geometry, but their robot collision groups do
# not interact.
TERRAIN_ROWS = 5
TERRAIN_COLS = 12
LAYOUT_SEED = 20261112

# A longitudinal ridge starts flat, climbs for six metres, keeps a two-metre
# crest or trough, descends for six metres and ends flat.  The zero-height
# edges are what make arbitrary patch order safe.
RAMP_START_M = -7.0
RAMP_TOP_START_M = -1.0
RAMP_TOP_END_M = 1.0
RAMP_END_M = 7.0
SPAWN_X_MARGIN_M = 0.8
SPAWN_Y_MARGIN_M = 0.8


def make_layout(seed: int = LAYOUT_SEED) -> tuple[tuple[str, ...], ...]:
    """Return a reproducible 3x3 order in which every terrain appears once."""
    types = np.asarray(TERRAIN_TYPES, dtype=object).copy()
    np.random.default_rng(int(seed)).shuffle(types)
    return tuple(
        tuple(str(value) for value in types[row * MAP_COLS:(row + 1) * MAP_COLS])
        for row in range(MAP_ROWS)
    )


def block_position(block_id: int) -> tuple[int, int]:
    """Convert a row-major 0..8 block id to ``(row, column)``."""
    if not 0 <= int(block_id) < len(TERRAIN_TYPES):
        raise ValueError(f"block_id must be in [0, 8], got {block_id}")
    return divmod(int(block_id), MAP_COLS)


def terrain_at_block(
    block_id: int,
    layout: tuple[tuple[str, ...], ...] | None = None,
) -> str:
    """Return the terrain name at a row-major block id."""
    row, column = block_position(block_id)
    selected = layout or make_layout()
    return selected[row][column]


def patch_height(local_x_m: float, terrain: str) -> float:
    """Return patch-relative height for terrain at ``local_x_m``.

    ``local_x_m`` is in one patch's local frame.  The map centre is zero.
    Uphill is a hill-shaped ridge; downhill is its inverse valley.  Both use
    the requested signed grade while maintaining zero-height patch edges.
    """
    if terrain not in GRADES:
        raise ValueError(f"Unknown random-terrain patch: {terrain}")
    x = float(local_x_m)
    grade = GRADES[terrain]
    if terrain == "flat" or x <= RAMP_START_M or x >= RAMP_END_M:
        return 0.0
    if x < RAMP_TOP_START_M:
        magnitude = abs(grade) * (x - RAMP_START_M)
    elif x <= RAMP_TOP_END_M:
        magnitude = abs(grade) * (RAMP_TOP_START_M - RAMP_START_M)
    else:
        magnitude = abs(grade) * (RAMP_END_M - x)
    return math.copysign(magnitude, grade)


def world_to_block(
    local_x_m: float,
    local_y_m: float,
) -> tuple[int, int, float, float]:
    """Map a terrain-relative coordinate to ``row, col, block_x, block_y``."""
    half_width = MAP_SIZE_M[0] / 2.0
    half_height = MAP_SIZE_M[1] / 2.0
    x = float(local_x_m)
    y = float(local_y_m)
    if not (-half_width <= x <= half_width and -half_height <= y <= half_height):
        raise ValueError(f"Coordinate ({x}, {y}) is outside the random terrain map")
    column = min(int((x + half_width) // BLOCK_LENGTH_M), MAP_COLS - 1)
    row = min(int((y + half_height) // BLOCK_WIDTH_M), MAP_ROWS - 1)
    block_x = x - ((column - (MAP_COLS - 1) / 2.0) * BLOCK_LENGTH_M)
    block_y = y - ((row - (MAP_ROWS - 1) / 2.0) * BLOCK_WIDTH_M)
    return row, column, block_x, block_y


def height_at_local(
    local_x_m: float,
    local_y_m: float,
    layout: tuple[tuple[str, ...], ...] | None = None,
) -> float:
    """Return terrain height relative to each environment's origin."""
    row, column, block_x, _ = world_to_block(local_x_m, local_y_m)
    selected = layout or make_layout()
    return patch_height(block_x, selected[row][column])


def block_bounds_local(block_id: int) -> tuple[float, float, float, float]:
    """Return ``x_min, x_max, y_min, y_max`` for one block's local frame."""
    row, column = block_position(block_id)
    x_center = (column - (MAP_COLS - 1) / 2.0) * BLOCK_LENGTH_M
    y_center = (row - (MAP_ROWS - 1) / 2.0) * BLOCK_WIDTH_M
    return (
        x_center - BLOCK_LENGTH_M / 2.0,
        x_center + BLOCK_LENGTH_M / 2.0,
        y_center - BLOCK_WIDTH_M / 2.0,
        y_center + BLOCK_WIDTH_M / 2.0,
    )


def _vertices_and_faces(
    layout: tuple[tuple[str, ...], ...],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Triangulate every patch while sharing zero height at map boundaries."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    breakpoints = (
        -BLOCK_LENGTH_M / 2.0,
        RAMP_START_M,
        RAMP_TOP_START_M,
        RAMP_TOP_END_M,
        RAMP_END_M,
        BLOCK_LENGTH_M / 2.0,
    )
    for row in range(MAP_ROWS):
        for column in range(MAP_COLS):
            terrain = layout[row][column]
            x_center = (column - (MAP_COLS - 1) / 2.0) * BLOCK_LENGTH_M
            y_center = (row - (MAP_ROWS - 1) / 2.0) * BLOCK_WIDTH_M
            y_values = (
                y_center - BLOCK_WIDTH_M / 2.0,
                y_center + BLOCK_WIDTH_M / 2.0,
            )
            offset = len(vertices)
            for local_x in breakpoints:
                z = patch_height(local_x, terrain)
                vertices.extend(
                    (x_center + local_x, y, z)
                    for y in y_values
                )
            for index in range(len(breakpoints) - 1):
                left = offset + index * 2
                right = left + 2
                faces.append((left, right, left + 1))
                faces.append((left + 1, right, right + 1))
    return vertices, faces


def _terrain_function(
    layout: tuple[tuple[str, ...], ...],
):
    def generate(_difficulty, cfg):
        import trimesh

        vertices, faces = _vertices_and_faces(layout)
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        # Vertices are centred on the environment origin.  Returning a zero
        # origin keeps later reset coordinates in the same frame.
        return [mesh], np.zeros(3, dtype=np.float64)

    generate.__name__ = "random_terrain_map_mesh"
    return generate


def configure(cfg) -> None:
    """Install the fixed randomized map for the new training profile."""
    settings = cfg.experiment_profile_settings
    if not settings.get("random_terrain_map"):
        return
    if cfg.experiment_profile_name != "random_terrain_20261112":
        raise ValueError(
            "random_terrain_map requires profile random_terrain_20261112, "
            f"got {cfg.experiment_profile_name!r}"
        )

    from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg

    layout_seed = int(settings.get("random_terrain_layout_seed", LAYOUT_SEED))
    layout = make_layout(layout_seed)
    cfg.terrain = cfg.terrain.copy()
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=layout_seed,
        size=MAP_SIZE_M,
        border_width=2.0,
        num_rows=TERRAIN_ROWS,
        num_cols=TERRAIN_COLS,
        curriculum=False,
        horizontal_scale=0.1,
        vertical_scale=0.001,
        use_cache=False,
        difficulty_range=(1.0, 1.0),
        sub_terrains={
            "random_map": MeshPlaneTerrainCfg(
                proportion=1.0,
                function=_terrain_function(layout),
            )
        },
    )
    cfg.scene.terrain = cfg.terrain.copy()


def initialize(env) -> None:
    """Create reset diagnostics for one constructed training environment."""
    if not env.cfg.experiment_profile_settings.get("random_terrain_map"):
        return
    import torch

    settings = env.cfg.experiment_profile_settings
    env._random_terrain_layout = make_layout(
        int(settings.get("random_terrain_layout_seed", LAYOUT_SEED))
    )
    env._random_terrain_block_grades = torch.tensor(
        [[GRADES[name] for name in row] for row in env._random_terrain_layout],
        dtype=torch.float32,
        device=env.device,
    )
    from agent_world.assets.wheelbipe_own import ASSET_DIR

    reset_bank = json.loads((ASSET_DIR / "reset_bank.json").read_text())
    env._random_terrain_wheel_endpoints = {
        side: torch.tensor(
            reset_bank["wheel_endpoints"][side],
            dtype=torch.float32,
            device=env.device,
        )
        for side in ("left", "right")
    }
    env._random_terrain_spawn_counts = torch.zeros(
        len(TERRAIN_TYPES), dtype=torch.long, device=env.device
    )
    env._random_terrain_reset_count = 0
    env._random_terrain_peak_penetration_m = 0.0
    env._random_terrain_force_block = None
    forced = settings.get("evaluation_block_ids")
    if forced is not None:
        if len(forced) != env.num_envs:
            raise ValueError(
                "evaluation_block_ids must contain one block id per environment"
            )
        env._random_terrain_force_block = torch.tensor(
            forced, dtype=torch.long, device=env.device
        )
    env._random_terrain_force_yaw = bool(settings.get("evaluation_force_yaw", False))
    env._random_terrain_evaluation_spawn = bool(
        settings.get("evaluation_block_ids") is not None
    )


def _heights(env, local_xy):
    """Analytic heights for reset; geometry and query use the same function."""
    import torch

    local_x = local_xy[..., 0]
    local_y = local_xy[..., 1]
    columns = torch.floor(
        (local_x + MAP_SIZE_M[0] / 2.0) / BLOCK_LENGTH_M
    ).long().clamp(0, MAP_COLS - 1)
    rows = torch.floor(
        (local_y + MAP_SIZE_M[1] / 2.0) / BLOCK_WIDTH_M
    ).long().clamp(0, MAP_ROWS - 1)
    block_x = local_x - (
        columns.to(torch.float32) - (MAP_COLS - 1) / 2.0
    ) * BLOCK_LENGTH_M
    grade = env._random_terrain_block_grades[rows, columns]
    magnitude = torch.where(
        block_x < RAMP_TOP_START_M,
        grade.abs() * (block_x - RAMP_START_M),
        torch.where(
            block_x <= RAMP_TOP_END_M,
            grade.abs() * (RAMP_TOP_START_M - RAMP_START_M),
            grade.abs() * (RAMP_END_M - block_x),
        ),
    ).clamp_min(0.0)
    signed = torch.where(grade < 0.0, -magnitude, magnitude)
    return torch.where(
        (grade == 0.0)
        | (block_x <= RAMP_START_M)
        | (block_x >= RAMP_END_M),
        torch.zeros_like(signed),
        signed,
    )


def _sample_spawn(env, ids):
    """Sample a patch and a valid local position for each reset environment."""
    import torch

    n = len(ids)
    if env._random_terrain_force_block is not None:
        blocks = env._random_terrain_force_block[ids]
    else:
        blocks = torch.randint(
            len(TERRAIN_TYPES), (n,), device=env.device
        )

    rows = torch.div(blocks, MAP_COLS, rounding_mode="floor")
    columns = blocks.remainder(MAP_COLS)
    x_centers = (columns.to(torch.float32) - (MAP_COLS - 1) / 2.0) * BLOCK_LENGTH_M
    y_centers = (rows.to(torch.float32) - (MAP_ROWS - 1) / 2.0) * BLOCK_WIDTH_M

    if env._random_terrain_evaluation_spawn:
        # Evaluation starts on the flat run-up and faces world +X.  This makes
        # every patch's fixed seed comparable across policies.
        local_x = x_centers - BLOCK_LENGTH_M / 2.0 + 1.0
        local_y = y_centers
        yaw = torch.zeros(n, device=env.device)
    else:
        local_x = x_centers + torch.empty(n, device=env.device).uniform_(
            -BLOCK_LENGTH_M / 2.0 + SPAWN_X_MARGIN_M,
            BLOCK_LENGTH_M / 2.0 - SPAWN_X_MARGIN_M,
        )
        local_y = y_centers + torch.empty(n, device=env.device).uniform_(
            -BLOCK_WIDTH_M / 2.0 + SPAWN_Y_MARGIN_M,
            BLOCK_WIDTH_M / 2.0 - SPAWN_Y_MARGIN_M,
        )
        yaw = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
        if env._random_terrain_force_yaw:
            yaw.zero_()
    local_xy = torch.stack((local_x, local_y), dim=-1)
    return blocks, local_xy, yaw


def reset(env, env_ids) -> None:
    """Reset robots at random map positions while preserving the old policy I/O."""
    import torch
    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
    from agent_world.sim620_compat import torch_data as T

    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    n = len(ids)
    if not n:
        return

    native_q = T(env._native_robot.data.default_joint_pos)[ids].clone()
    endpoints = []
    for side in ("left", "right"):
        bank = env._reset_bank[side]
        sample = torch.randint(len(bank), (n,), device=env.device)
        names = [name for name in env._bank_names if name.startswith(side + "_")]
        source = [env._bank_names.index(name) for name in names]
        destination = [env.robot.joint_names.index(name) for name in names]
        native_q[:, destination] = bank[sample][:, source]
        endpoints.append(env._random_terrain_wheel_endpoints[side][sample])
    endpoints_tensor = torch.stack(endpoints, dim=1)

    origin = env.terrain.env_origins[ids]
    blocks, local_xy, yaw = _sample_spawn(env, ids)
    xy = origin[:, :2] + local_xy

    # Estimate the local ground normal from central differences.  The robot is
    # tilted to match that plane before its wheels are placed.
    offsets = local_xy.new_tensor(((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25)))
    ground = _heights(env, local_xy[:, None, :] + offsets)
    gx = (ground[:, 0] - ground[:, 1]) / 0.5
    gy = (ground[:, 2] - ground[:, 3]) / 0.5
    roll = torch.atan(gy * yaw.cos() - gx * yaw.sin())
    pitch = -torch.atan(gx * yaw.cos() + gy * yaw.sin())
    quat = quat_from_euler_xyz(roll, pitch, yaw)
    rotated_endpoints = quat_apply(
        quat[:, None].expand(-1, 2, -1), endpoints_tensor
    )

    footprint = local_xy.new_tensor(
        ((0.0, 0.0), (0.03, 0.0), (-0.03, 0.0), (0.0, 0.02), (0.0, -0.02))
    )
    tire_local = local_xy[:, None, None, :] + rotated_endpoints[:, :, None, :2] + footprint
    tire_ground = _heights(env, tire_local)
    clearance = 0.004 + torch.rand(n, device=env.device) * 0.003
    z_local = (
        tire_ground.max(dim=-1).values - rotated_endpoints[:, :, 2] + 0.0625
    ).max(dim=-1).values + clearance

    policy_q = native_q * env.robot.sign + env.robot.offset
    zeros = torch.zeros_like(policy_q)
    env.robot.write_joint_state_to_sim(policy_q, zeros, env_ids=ids)
    env.robot.set_joint_position_target(policy_q, env_ids=ids)
    env.robot.set_joint_velocity_target(zeros, env_ids=ids)
    env.robot.set_joint_effort_target(zeros, env_ids=ids)

    root = T(env._native_robot.data.default_root_state)[ids].clone()
    root[:, :2] = xy
    root[:, 2] = origin[:, 2] + z_local
    root[:, 3:7] = quat
    root[:, 7:] = 0.0
    env.robot.write_root_state_to_sim(root, ids)
    env._spring_outside[ids] = False
    env.spring_force_rand[ids] = 0.0
    env._own_reset_count += n
    env._sync_policy_state()

    if getattr(env, "_dr_ready", False):
        env._dr_clock[ids] = 0.0
    if hasattr(env, "_slew_leg"):
        env._slew_leg[ids] = env.robot.data.joint_pos[ids][:, env._legs_act_idx]
        env._slew_wheel[ids] = zeros[:, env._wheel_idx]

    env._random_terrain_spawn_counts += torch.bincount(
        blocks, minlength=len(TERRAIN_TYPES)
    )
    env._random_terrain_reset_count += n
    penetration = (
        tire_ground.max(dim=-1).values
        - (z_local[:, None] + rotated_endpoints[:, :, 2] - 0.0625)
    ).max()
    env._random_terrain_peak_penetration_m = max(
        env._random_terrain_peak_penetration_m,
        float(penetration),
    )
    if float(penetration) > 1e-5:
        raise RuntimeError("A tire starts below the random terrain map")
def health(env) -> dict:
    """Return fields included in the existing periodic training audit."""
    forced = (
        env._random_terrain_force_block.detach().cpu().tolist()
        if env._random_terrain_force_block is not None
        else None
    )
    return {
        "layout_seed": int(
            env.cfg.experiment_profile_settings.get(
                "random_terrain_layout_seed", LAYOUT_SEED
            )
        ),
        "layout": [list(row) for row in env._random_terrain_layout],
        "spawn_counts": env._random_terrain_spawn_counts.tolist(),
        "resets": env._random_terrain_reset_count,
        "forced_evaluation_blocks": forced,
        "peak_initial_tire_penetration_m": env._random_terrain_peak_penetration_m,
    }
