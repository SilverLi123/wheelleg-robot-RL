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
RANDOM_TERRAIN_PROFILES = (
    "random_terrain_20261112",
    "random_terrain_forward_20261112",
    "random_terrain_straight_20261112",
    "random_terrain_slope_pitch_20261112",
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
PATCH_Y_EDGE_TAPER_M = 1.5
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


def patch_height(
    local_x_m: float,
    terrain: str,
    local_y_m: float = 0.0,
) -> float:
    """Return patch-relative height for terrain at ``local_x_m``.

    ``local_x_m`` is in one patch's local frame.  The map centre is zero.
    Uphill is a hill-shaped ridge; downhill is its inverse valley.  Both use
    the requested signed grade. The profile tapers to zero over the final
    1.5 m at each `y` edge, so neighboring rows meet without a step.
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
    y = float(local_y_m)
    y_taper = max(
        0.0,
        min(1.0, (BLOCK_WIDTH_M / 2.0 - abs(y)) / PATCH_Y_EDGE_TAPER_M),
    )
    return math.copysign(magnitude * y_taper, grade)


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
    row, column, block_x, block_y = world_to_block(local_x_m, local_y_m)
    selected = layout or make_layout()
    return patch_height(block_x, selected[row][column], block_y)


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


def collision_box_corners(
    local_min: tuple[float, float, float] | list[float] | np.ndarray,
    local_max: tuple[float, float, float] | list[float] | np.ndarray,
) -> np.ndarray:
    """Return the eight corners of an axis-aligned collision bounding box.

    The values are expressed in the rigid body's local frame. Keeping this
    conversion separate makes the clearance calculation testable without
    starting Isaac Sim.
    """
    minimum = np.asarray(local_min, dtype=np.float64)
    maximum = np.asarray(local_max, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("collision bounds must each contain exactly three values")
    if np.any(minimum > maximum):
        raise ValueError("collision-bound minimum must not exceed maximum")
    return np.asarray(
        [
            (x, y, z)
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )


def quat_rotate_wxyz_numpy(points: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate row-vector points by one Isaac-style ``(w, x, y, z)`` quaternion."""
    vectors = np.asarray(points, dtype=np.float64)
    q = np.asarray(quat, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if q.shape != (4,):
        raise ValueError("quaternion must have shape (4,), ordered wxyz")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = q / norm
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return vectors @ rotation.T


def quats_rotate_wxyz_numpy(
    points: np.ndarray,
    quats: np.ndarray,
) -> np.ndarray:
    """Rotate one point set by each quaternion in a batch."""
    vectors = np.asarray(points, dtype=np.float64)
    q = np.asarray(quats, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("quaternions must have shape (N, 4), ordered wxyz")
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("a batch quaternion has zero norm")
    w, x, y, z = (q / norms).T
    rotation = np.empty((len(q), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - w * z)
    rotation[:, 0, 2] = 2.0 * (x * z + w * y)
    rotation[:, 1, 0] = 2.0 * (x * y + w * z)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - w * x)
    rotation[:, 2, 0] = 2.0 * (x * z - w * y)
    rotation[:, 2, 1] = 2.0 * (y * z + w * x)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return np.einsum("ij,njk->nik", vectors, rotation, optimize=True)


def collision_points_minimum_clearance(
    *,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    local_points: np.ndarray,
    height_at_local_xy,
) -> np.ndarray:
    """Return minimum clearance for supplied body-local mesh points."""
    positions = np.asarray(body_pos_w, dtype=np.float64)
    quats = np.asarray(body_quat_w, dtype=np.float64)
    points = np.asarray(local_points, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("body_pos_w must have shape (N, 3)")
    if quats.shape != (len(positions), 4):
        raise ValueError("body_quat_w must have shape (N, 4), ordered wxyz")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("local_points must have shape (M, 3), with M > 0")
    world_points = quats_rotate_wxyz_numpy(points, quats)
    world_points += positions[:, None, :]
    ground_z = np.asarray(
        height_at_local_xy(
            world_points[:, :, :2].reshape(-1, 2)
        ),
        dtype=np.float64,
    )
    if ground_z.shape != (len(positions) * len(points),):
        raise ValueError(
            "height callback must return one value per transformed mesh point"
        )
    return np.min(
        world_points[:, :, 2] - ground_z.reshape(len(positions), len(points)),
        axis=1,
    )


def collision_boxes_minimum_clearance(
    *,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    local_min: tuple[float, float, float] | list[float] | np.ndarray,
    local_max: tuple[float, float, float] | list[float] | np.ndarray,
    height_at_local_xy,
) -> np.ndarray:
    """Return one minimum corner clearance for every body in a batch."""
    return collision_points_minimum_clearance(
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        local_points=collision_box_corners(local_min, local_max),
        height_at_local_xy=height_at_local_xy,
    )


def collision_box_minimum_clearance(
    *,
    body_pos_w: tuple[float, float, float] | list[float] | np.ndarray,
    body_quat_w: tuple[float, float, float, float] | list[float] | np.ndarray,
    local_min: tuple[float, float, float] | list[float] | np.ndarray,
    local_max: tuple[float, float, float] | list[float] | np.ndarray,
    height_at_local_xy,
) -> float:
    """Measure world-Z clearance at every corner of a rotated body box.

    ``height_at_local_xy`` receives an ``(8, 2)`` array in the terrain-local
    frame and returns eight ground heights. Using the lowest corner prevents a
    body-origin estimate from hiding an overhanging gimbal collision mesh.
    """
    position = np.asarray(body_pos_w, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("body_pos_w must contain exactly three values")
    return float(
        collision_boxes_minimum_clearance(
            body_pos_w=position[None, :],
            body_quat_w=np.asarray(body_quat_w, dtype=np.float64)[None, :],
            local_min=local_min,
            local_max=local_max,
            height_at_local_xy=height_at_local_xy,
        )[0]
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
            taper_start = BLOCK_WIDTH_M / 2.0 - PATCH_Y_EDGE_TAPER_M
            y_values = (
                -BLOCK_WIDTH_M / 2.0,
                -taper_start,
                taper_start,
                BLOCK_WIDTH_M / 2.0,
            )
            offset = len(vertices)
            for local_x in breakpoints:
                for local_y in y_values:
                    vertices.append(
                        (
                            x_center + local_x,
                            y_center + local_y,
                            patch_height(local_x, terrain, local_y),
                        )
                    )
            for index in range(len(breakpoints) - 1):
                for y_index in range(len(y_values) - 1):
                    left = offset + index * len(y_values) + y_index
                    right = left + len(y_values)
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
    if cfg.experiment_profile_name not in RANDOM_TERRAIN_PROFILES:
        raise ValueError(
            "random_terrain_map requires a declared random-terrain profile, "
            f"got {cfg.experiment_profile_name!r}"
        )

    from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg

    layout_seed = int(settings.get("random_terrain_layout_seed", LAYOUT_SEED))
    layout = make_layout(layout_seed)
    color_scheme = str(settings.get("terrain_color_scheme", "height"))
    if color_scheme != "height":
        raise ValueError(
            "The connected random map requires terrain_color_scheme='height' "
            "so uphill and downhill remain visually distinguishable"
        )
    cfg.terrain = cfg.terrain.copy()
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=layout_seed,
        size=MAP_SIZE_M,
        # Spawn, height queries and route waypoints use the full map frame.
        # A non-zero generator border shrinks each sloped map inward and inserts
        # a flat strip at zero height inside those coordinates. The zero-height
        # map edges already join repeated copies, so no extra border is needed.
        border_width=0.0,
        num_rows=TERRAIN_ROWS,
        num_cols=TERRAIN_COLS,
        curriculum=False,
        horizontal_scale=0.1,
        vertical_scale=0.001,
        use_cache=False,
        # Isaac Lab applies the Turbo/inferno height palette to the combined
        # terrain mesh.  High crests, neutral crossings and low valleys then
        # read clearly in GUI previews and recorded videos.
        color_scheme=color_scheme,
        difficulty_range=(1.0, 1.0),
        sub_terrains={
            "random_map": MeshPlaneTerrainCfg(
                proportion=1.0,
                function=_terrain_function(layout),
            )
        },
    )
    cfg.scene.terrain = cfg.terrain.copy()


def configure_preview(cfg) -> None:
    """Render one map copy while leaving training and physics settings intact."""
    if not cfg.experiment_profile_settings.get("random_terrain_map"):
        return
    configure(cfg)
    generator = cfg.terrain.terrain_generator.copy()
    generator.num_rows = 1
    generator.num_cols = 1
    generator.curriculum = False
    cfg.terrain.terrain_generator = generator
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
    env._random_terrain_run_up_spawn_count = 0
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
    yaw_span_deg = settings.get("random_terrain_training_yaw_span_deg", None)
    env._random_terrain_training_yaw_span_rad = (
        None
        if yaw_span_deg is None
        else math.radians(float(yaw_span_deg))
    )
    if env._random_terrain_training_yaw_span_rad is not None and not (
        0.0 <= env._random_terrain_training_yaw_span_rad <= math.pi
    ):
        raise ValueError(
            "random_terrain_training_yaw_span_deg must be in [0, 180]"
        )
    start_fraction = float(settings.get("random_terrain_start_fraction", 0.0))
    if not 0.0 <= start_fraction <= 1.0:
        raise ValueError("random_terrain_start_fraction must be in [0, 1]")
    env._random_terrain_start_fraction = start_fraction


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
    block_y = local_y - (
        rows.to(torch.float32) - (MAP_ROWS - 1) / 2.0
    ) * BLOCK_WIDTH_M
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
    signed = torch.where(
        (grade == 0.0)
        | (block_x <= RAMP_START_M)
        | (block_x >= RAMP_END_M),
        torch.zeros_like(signed),
        signed,
    )
    y_taper = (
        (BLOCK_WIDTH_M / 2.0 - block_y.abs()) / PATCH_Y_EDGE_TAPER_M
    ).clamp(0.0, 1.0)
    return signed * y_taper


def slope_pitch_targets(env):
    """Return current pitch, terrain-following pitch target, and slope norm.

    The sign convention matches ``_sample_spawn``: a positive world-X grade
    produces a negative body pitch in the reset quaternion.  The target is the
    slope along the robot's current heading, so the same code works when the
    policy changes yaw.
    """
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat
    from agent_world.sim620_compat import torch_data as T

    root_pos = T(env._native_robot.data.root_pos_w)
    root_quat = T(env._native_robot.data.root_quat_w)
    origin = T(env.terrain.env_origins)
    local_xy = root_pos[:, :2] - origin[:, :2]
    offsets = local_xy.new_tensor(
        ((0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25))
    )
    ground = _heights(env, local_xy[:, None, :] + offsets)
    gradient_x = (ground[:, 0] - ground[:, 1]) / 0.5
    gradient_y = (ground[:, 2] - ground[:, 3]) / 0.5
    _, current_pitch, yaw = euler_xyz_from_quat(root_quat)
    target_pitch = -torch.atan(
        gradient_x * yaw.cos() + gradient_y * yaw.sin()
    )
    slope_norm = torch.sqrt(gradient_x.square() + gradient_y.square())
    return current_pitch, target_pitch, slope_norm


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
        yaw_span = env._random_terrain_training_yaw_span_rad
        if yaw_span is None:
            # Preserve the original omnidirectional reset distribution for
            # experiments that do not opt into a narrower heading.
            yaw = torch.empty(n, device=env.device).uniform_(
                -math.pi, math.pi
            )
        else:
            # The slopes run along world X. Keep the position random while
            # training enough longitudinal downhill samples for body clearance.
            yaw = torch.empty(n, device=env.device).uniform_(
                -yaw_span, yaw_span
            )
        start_fraction = env._random_terrain_start_fraction
        if start_fraction > 0.0:
            at_run_up = (
                torch.rand(n, device=env.device) < start_fraction
            )
            if at_run_up.any():
                # Reproduce the deterministic evaluation condition for a
                # fraction of training resets: flat run-up, patch centreline,
                # facing uphill/downhill along world +X.
                local_x[at_run_up] = (
                    x_centers[at_run_up] - BLOCK_LENGTH_M / 2.0 + 1.0
                )
                local_y[at_run_up] = y_centers[at_run_up]
                local_yaw = torch.zeros_like(yaw[at_run_up])
                if yaw_span is not None and yaw_span > 0.0:
                    local_yaw = torch.empty_like(local_yaw).uniform_(
                        -yaw_span, yaw_span
                    )
                yaw[at_run_up] = local_yaw
                env._random_terrain_run_up_spawn_count += int(
                    at_run_up.sum().item()
                )
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
        "run_up_spawns": env._random_terrain_run_up_spawn_count,
        "resets": env._random_terrain_reset_count,
        "forced_evaluation_blocks": forced,
        "peak_initial_tire_penetration_m": env._random_terrain_peak_penetration_m,
    }
