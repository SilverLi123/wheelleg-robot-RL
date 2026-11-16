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
    "random_terrain_20261113",
    "random_terrain_forward_20261112",
    "random_terrain_straight_20261112",
    "random_terrain_slope_pitch_20261112",
    "random_terrain_speed065_20261113",
    "random_terrain_reward12_20261113",
    "random_terrain_sigma003_20261113",
    "random_terrain_matcheddiv_20261113",
    "random_terrain_reward12sigma003_20261113",
    "random_terrain_transition_20261113",
    "random_terrain_relcritic_20261113",
    "random_terrain_yaw045_20261113",
    "random_terrain_stairs_20261113",
    "random_terrain_opt_posture_20261114",
    "random_terrain_opt_speed_20261114",
    "random_terrain_opt_full_rm_20261114",
    "random_terrain_stage2_agile_20261114",
    "random_terrain_stage2_steep_agile_20261114",
    "random_terrain_stage3_curriculum_20261115",
    "random_terrain_stage3_dirH2a_20261115",
    "random_terrain_stage3_dirH2b_yaw_20261115",
    "random_terrain_stage3_dirH3_20261115",
    "random_terrain_stage3_dirH4_20261115",
    "random_terrain_stage3_dirH5_20261115",
    "random_terrain_stage3_dirH6_20261115",
    "random_terrain_stage3_dirH7_20261115",
    "random_terrain_stage3_dirH8_20261115",
    "random_terrain_stage4_dirI1_20261116",
    "random_terrain_stage4_dirI2_20261116",
    "random_terrain_stage4_dirI3_20261116",
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

# Hybrid stairs+platforms+slopes map extends Y from [-18, +18] to [-24, +24]
# and appends four closed obstacle lanes along y_center = +21.0 m
# (y in [19.5, 22.5]): two stairs_down lanes (35/70 mm per step) and two
# wheel-climbable platform_up steps (25/40 mm).
HYBRID_MAP_SIZE_M = (BLOCK_LENGTH_M * MAP_COLS, 48.0)
STAIRS_LANE_KINDS = ("stairs_down", "stairs_down", "platform_up", "platform_up")
STAIRS_LANE_STEP_HEIGHTS_M = (0.035, 0.070, 0.025, 0.040)
STAIRS_LANE_X_STARTS_M = (-22.0, -10.0, 2.0, 14.0)
STAIRS_LANE_Y_CENTER_M = 21.0
STAIRS_LANE_WIDTH_M = 3.0
STAIRS_LANE_APPROACH_M = 3.0
STAIRS_LANE_LANDING_M = 3.6
STAIRS_LANE_PLATFORM_LANDING_M = 5.0
STAIRS_LANE_STEP_COUNT = 4
STAIRS_LANE_TREAD_M = 0.35
STAIRS_LANE_BAND_Y_MIN_M = 18.5

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

# The mesh vertices and every height query use these exact breakpoints. Between
# them, the collision surface is triangular; it is not the continuous product
# of the longitudinal profile and the lateral taper.
PATCH_X_BREAKPOINTS_M = (
    -BLOCK_LENGTH_M / 2.0,
    RAMP_START_M,
    RAMP_TOP_START_M,
    RAMP_TOP_END_M,
    RAMP_END_M,
    BLOCK_LENGTH_M / 2.0,
)
PATCH_Y_BREAKPOINTS_M = (
    -BLOCK_WIDTH_M / 2.0,
    -(BLOCK_WIDTH_M / 2.0 - PATCH_Y_EDGE_TAPER_M),
    BLOCK_WIDTH_M / 2.0 - PATCH_Y_EDGE_TAPER_M,
    BLOCK_WIDTH_M / 2.0,
)


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
    """Return the actual triangular collision-surface height in one patch.

    ``local_x_m`` is in one patch's local frame.  The map centre is zero.
    Uphill is a hill-shaped ridge; downhill is its inverse valley.  Both use
    the requested signed grade. The profile tapers to zero over the final
    1.5 m at each `y` edge. Inside each mesh cell, linear interpolation across
    one of the two actual triangles is authoritative; using the product of an
    X profile and a Y taper would disagree with physics between vertices.
    """
    if terrain not in GRADES:
        raise ValueError(f"Unknown random-terrain patch: {terrain}")
    grade = GRADES[terrain]
    if terrain == "flat":
        return 0.0
    x = float(local_x_m)
    y = float(local_y_m)
    x_index = min(
        max(
            int(np.searchsorted(PATCH_X_BREAKPOINTS_M, x, side="right")) - 1,
            0,
        ),
        len(PATCH_X_BREAKPOINTS_M) - 2,
    )
    y_index = min(
        max(
            int(np.searchsorted(PATCH_Y_BREAKPOINTS_M, y, side="right")) - 1,
            0,
        ),
        len(PATCH_Y_BREAKPOINTS_M) - 2,
    )
    x0, x1 = PATCH_X_BREAKPOINTS_M[x_index:x_index + 2]
    y0, y1 = PATCH_Y_BREAKPOINTS_M[y_index:y_index + 2]
    x_fraction = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
    y_fraction = min(1.0, max(0.0, (y - y0) / (y1 - y0)))

    z00 = _profile_vertex_height(x0, grade, y0)
    z10 = _profile_vertex_height(x1, grade, y0)
    z01 = _profile_vertex_height(x0, grade, y1)
    z11 = _profile_vertex_height(x1, grade, y1)
    if x_fraction + y_fraction <= 1.0:
        # Lower-left triangle: (x0, y0), (x1, y0), (x0, y1).
        return (
            z00 * (1.0 - x_fraction - y_fraction)
            + z10 * x_fraction
            + z01 * y_fraction
        )
    # Upper-right triangle: (x0, y1), (x1, y0), (x1, y1).
    return (
        z10 * (1.0 - y_fraction)
        + z01 * (1.0 - x_fraction)
        + z11 * (x_fraction + y_fraction - 1.0)
    )


def _profile_vertex_height(
    local_x_m: float,
    grade: float,
    local_y_m: float = 0.0,
) -> float:
    """Return the height used at one exact mesh vertex."""
    x = float(local_x_m)
    if grade == 0.0 or x <= RAMP_START_M or x >= RAMP_END_M:
        return 0.0
    if x < RAMP_TOP_START_M:
        magnitude = abs(grade) * (x - RAMP_START_M)
    elif x <= RAMP_TOP_END_M:
        magnitude = abs(grade) * (RAMP_TOP_START_M - RAMP_START_M)
    else:
        magnitude = abs(grade) * (RAMP_END_M - x)
    y_taper = max(
        0.0,
        min(
            1.0,
            (BLOCK_WIDTH_M / 2.0 - abs(float(local_y_m)))
            / PATCH_Y_EDGE_TAPER_M,
        ),
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
    return np.einsum("ij,nkj->nik", vectors, rotation, optimize=True)


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


def _load_obstacle_terrain():
    try:
        from . import obstacle_terrain

        return obstacle_terrain
    except ImportError:
        import importlib.util
        import sys
        from pathlib import Path

        path = Path(__file__).with_name("obstacle_terrain.py")
        name = "_wheelbipe_obstacle_terrain_standalone"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load obstacle_terrain from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


def _stair_lane_specs():
    obstacle = _load_obstacle_terrain()
    y_start = STAIRS_LANE_Y_CENTER_M - STAIRS_LANE_WIDTH_M / 2.0
    return tuple(
        (
            x_start,
            y_start,
            obstacle.ObstacleSpec(
                kind=kind,
                height_m=step_height_m,
                width_m=STAIRS_LANE_WIDTH_M,
                approach_m=STAIRS_LANE_APPROACH_M,
                landing_m=(
                    STAIRS_LANE_PLATFORM_LANDING_M
                    if kind == "platform_up"
                    else STAIRS_LANE_LANDING_M
                ),
                step_count=STAIRS_LANE_STEP_COUNT,
                tread_m=STAIRS_LANE_TREAD_M,
                base_z_m=-0.20,
            ),
        )
        for kind, x_start, step_height_m in zip(
            STAIRS_LANE_KINDS,
            STAIRS_LANE_X_STARTS_M,
            STAIRS_LANE_STEP_HEIGHTS_M,
        )
    )


def _vertices_and_faces(
    layout: tuple[tuple[str, ...], ...],
    include_stairs_lanes: bool = False,
    grade_scale: float = 1.0,
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Triangulate every patch while sharing zero height at map boundaries."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    scale = float(grade_scale)
    for row in range(MAP_ROWS):
        for column in range(MAP_COLS):
            terrain = layout[row][column]
            x_center = (column - (MAP_COLS - 1) / 2.0) * BLOCK_LENGTH_M
            y_center = (row - (MAP_ROWS - 1) / 2.0) * BLOCK_WIDTH_M
            offset = len(vertices)
            for local_x in PATCH_X_BREAKPOINTS_M:
                for local_y in PATCH_Y_BREAKPOINTS_M:
                    vertices.append(
                        (
                            x_center + local_x,
                            y_center + local_y,
                            _profile_vertex_height(
                                local_x,
                                GRADES[terrain] * scale,
                                local_y,
                            ),
                        )
                    )
            for index in range(len(PATCH_X_BREAKPOINTS_M) - 1):
                for y_index in range(len(PATCH_Y_BREAKPOINTS_M) - 1):
                    left = (
                        offset
                        + index * len(PATCH_Y_BREAKPOINTS_M)
                        + y_index
                    )
                    right = left + len(PATCH_Y_BREAKPOINTS_M)
                    faces.append((left, right, left + 1))
                    faces.append((left + 1, right, right + 1))
    if include_stairs_lanes:
        obstacle = _load_obstacle_terrain()
        for x_start, y_start, spec in _stair_lane_specs():
            lane_verts, lane_faces = obstacle.mesh_data(spec)
            offset = len(vertices)
            for vx, vy, vz in lane_verts:
                vertices.append((x_start + vx, y_start + vy, vz))
            for f0, f1, f2 in lane_faces:
                faces.append((offset + f0, offset + f1, offset + f2))
    return vertices, faces


def _terrain_function(
    layout: tuple[tuple[str, ...], ...],
    include_stairs_lanes: bool = False,
    grade_scale: float = 1.0,
):
    def generate(_difficulty, cfg):
        import trimesh

        vertices, faces = _vertices_and_faces(
            layout,
            include_stairs_lanes=include_stairs_lanes,
            grade_scale=grade_scale,
        )
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
    include_stairs_lanes = bool(settings.get("random_terrain_stairs_lanes", False))
    grade_scale = float(settings.get("random_terrain_grade_scale", 1.0))
    if grade_scale <= 0.0:
        raise ValueError(f"random_terrain_grade_scale must be positive, got {grade_scale}")
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
        size=HYBRID_MAP_SIZE_M if include_stairs_lanes else MAP_SIZE_M,
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
                function=_terrain_function(
                    layout,
                    include_stairs_lanes=include_stairs_lanes,
                    grade_scale=grade_scale,
                ),
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
    grade_scale = float(settings.get("random_terrain_grade_scale", 1.0))
    if grade_scale <= 0.0:
        raise ValueError(f"random_terrain_grade_scale must be positive, got {grade_scale}")
    env._random_terrain_grade_scale = grade_scale
    env._random_terrain_block_grades = torch.tensor(
        [[GRADES[name] * grade_scale for name in row] for row in env._random_terrain_layout],
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
    env._random_terrain_transition_spawn_count = 0
    env._random_terrain_stairs_spawn_count = 0
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
    env._random_terrain_evaluation_xy_yaw = None
    forced_xy_yaw = settings.get("evaluation_spawn_xy_yaw")
    if forced_xy_yaw is not None:
        if len(forced_xy_yaw) != env.num_envs:
            raise ValueError(
                "evaluation_spawn_xy_yaw must contain one (x, y, yaw) tuple per environment"
            )
        env._random_terrain_evaluation_xy_yaw = torch.tensor(
            forced_xy_yaw, dtype=torch.float32, device=env.device
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
    transition_fraction = float(
        settings.get("random_terrain_transition_fraction", 0.0)
    )
    if not 0.0 <= transition_fraction <= 1.0:
        raise ValueError("random_terrain_transition_fraction must be in [0, 1]")
    env._random_terrain_transition_fraction = transition_fraction
    transition_yaw_rad = float(
        settings.get("random_terrain_transition_yaw_rad", 0.30)
    )
    if not 0.0 <= transition_yaw_rad <= math.pi:
        raise ValueError("random_terrain_transition_yaw_rad must be in [0, pi]")
    env._random_terrain_transition_yaw_rad = transition_yaw_rad
    stairs_lanes = bool(settings.get("random_terrain_stairs_lanes", False))
    stairs_fraction = float(settings.get("random_terrain_stairs_fraction", 0.0))
    if not 0.0 <= stairs_fraction <= 1.0:
        raise ValueError("random_terrain_stairs_fraction must be in [0, 1]")
    if stairs_fraction > 0.0 and not stairs_lanes:
        raise ValueError(
            "random_terrain_stairs_fraction requires random_terrain_stairs_lanes=True"
        )
    env._random_terrain_stairs_lanes = stairs_lanes
    env._random_terrain_stairs_fraction = stairs_fraction
    stairs_yaw_rad = float(settings.get("random_terrain_stairs_yaw_rad", 0.20))
    if not 0.0 <= stairs_yaw_rad <= math.pi:
        raise ValueError("random_terrain_stairs_yaw_rad must be in [0, pi]")
    env._random_terrain_stairs_yaw_rad = stairs_yaw_rad
    lane_probs = settings.get("random_terrain_stairs_lane_probs", None)
    if lane_probs is not None:
        if len(lane_probs) != len(STAIRS_LANE_KINDS):
            raise ValueError(
                "random_terrain_stairs_lane_probs must match STAIRS_LANE_KINDS length"
            )
        probs_t = torch.tensor(lane_probs, dtype=torch.float32, device=env.device)
        if not torch.all(probs_t >= 0.0) or float(probs_t.sum()) <= 0.0:
            raise ValueError(
                "random_terrain_stairs_lane_probs must be non-negative with positive sum"
            )
        env._random_terrain_stairs_lane_probs = probs_t / probs_t.sum()
    else:
        env._random_terrain_stairs_lane_probs = None
    x_range = settings.get("random_terrain_stairs_x_range", (1.0, 2.5))
    if len(x_range) != 2 or float(x_range[0]) >= float(x_range[1]):
        raise ValueError("random_terrain_stairs_x_range must be a valid (low, high) pair")
    env._random_terrain_stairs_x_range = (float(x_range[0]), float(x_range[1]))
    env._random_terrain_steep_block_mask = torch.tensor(
        [abs(GRADES[name]) >= 0.14 for row in env._random_terrain_layout for name in row],
        dtype=torch.bool,
        device=env.device,
    )
    env._random_terrain_block_probs = None
    env._random_terrain_steep_block_prob = None


def curriculum_progress_fraction(iteration: int, warmup_iters: int, ramp_iters: int) -> float:
    """Return normalized Stage 3 curriculum progress in ``[0.0, 1.0]``."""
    if iteration <= warmup_iters:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return min(max((float(iteration) - float(warmup_iters)) / float(ramp_iters), 0.0), 1.0)


def interpolate_stage_curriculum(curriculum_cfg: dict, iteration: int) -> dict:
    """Interpolate Stage 3 speed, yaw-rate, standing, and steep-slope bounds."""
    progress = curriculum_progress_fraction(
        iteration,
        int(curriculum_cfg.get("warmup_iters", 0)),
        int(curriculum_cfg.get("ramp_iters", 1)),
    )

    def lerp_pair(start_key: str, end_key: str) -> tuple[float, float]:
        a0, b0 = curriculum_cfg[start_key]
        a1, b1 = curriculum_cfg[end_key]
        return (
            float(a0) + (float(a1) - float(a0)) * progress,
            float(b0) + (float(b1) - float(b0)) * progress,
        )

    stand0 = float(curriculum_cfg.get("standing_fraction_start", 0.20))
    stand1 = float(curriculum_cfg.get("standing_fraction_end", stand0))
    steep0 = float(curriculum_cfg.get("steep_block_prob_start", 0.25))
    steep1 = float(curriculum_cfg.get("steep_block_prob_end", steep0))
    return {
        "iteration": int(iteration),
        "progress": progress,
        "lin_vel_x": lerp_pair("lin_vel_x_start", "lin_vel_x_end"),
        "ang_vel_z": lerp_pair("ang_vel_z_start", "ang_vel_z_end"),
        "standing_fraction": stand0 + (stand1 - stand0) * progress,
        "steep_block_prob": steep0 + (steep1 - steep0) * progress,
    }


def compute_block_sampling_probs(steep_mask, steep_block_prob: float):
    """Return per-block sampling probabilities split cleanly across steep/non-steep blocks."""
    if not 0.0 <= float(steep_block_prob) <= 1.0:
        raise ValueError(f"steep_block_prob must be in [0, 1], got {steep_block_prob}")
    steep_count = sum(1 for flag in steep_mask if flag)
    normal_count = len(steep_mask) - steep_count
    if steep_count == 0 or normal_count == 0:
        return [1.0 / len(steep_mask)] * len(steep_mask)
    w_steep = float(steep_block_prob) / float(steep_count)
    w_normal = (1.0 - float(steep_block_prob)) / float(normal_count)
    return [w_steep if flag else w_normal for flag in steep_mask]


def set_steep_block_probability(env, steep_block_prob: float) -> None:
    """Update per-block sampling probabilities on ``env`` during curriculum training."""
    import torch

    mask = env._random_terrain_steep_block_mask.tolist()
    probs = compute_block_sampling_probs(mask, steep_block_prob)
    env._random_terrain_block_probs = torch.tensor(probs, dtype=torch.float32, device=env.device)
    env._random_terrain_steep_block_prob = float(steep_block_prob)


def clamp_command_centripetal_accel(commands, max_lat_accel_m_s2: float, env_ids=None):
    """Clamp yaw command ``commands[..., 2]`` so ``|vx * wz| <= max_lat_accel_m_s2``."""
    limit = float(max_lat_accel_m_s2)
    if limit <= 0.0:
        return commands
    if hasattr(commands, "clone"):
        idx = slice(None) if env_ids is None else env_ids
        vx = commands[idx, 0]
        wz = commands[idx, 2]
        wz_cap = limit / vx.abs().clamp_min(1e-3)
        commands[idx, 2] = wz.clamp(-wz_cap, wz_cap)
        return commands
    indices = range(len(commands)) if env_ids is None else env_ids
    for idx in indices:
        vx = float(commands[idx][0])
        wz = float(commands[idx][2])
        wz_cap = limit / max(abs(vx), 1e-3)
        if wz > wz_cap:
            commands[idx][2] = wz_cap
        elif wz < -wz_cap:
            commands[idx][2] = -wz_cap
    return commands


def _stairs_lane_heights(env, local_x, slope_height):
    import torch

    device = local_x.device
    dtype = slope_height.dtype
    step_edges = getattr(env, "_random_terrain_stairs_step_edges", None)
    if step_edges is None or step_edges.device != device:
        step_edges = local_x.new_tensor(
            tuple(
                STAIRS_LANE_APPROACH_M + i * STAIRS_LANE_TREAD_M
                for i in range(STAIRS_LANE_STEP_COUNT)
            )
        )
        lane_length_m = (
            STAIRS_LANE_APPROACH_M
            + STAIRS_LANE_STEP_COUNT * STAIRS_LANE_TREAD_M
            + STAIRS_LANE_LANDING_M
        )
        lane_centers = local_x.new_tensor(
            tuple(x + lane_length_m / 2.0 for x in STAIRS_LANE_X_STARTS_M)
        )
        lane_x_starts = local_x.new_tensor(STAIRS_LANE_X_STARTS_M)
        lane_step_h = local_x.new_tensor(STAIRS_LANE_STEP_HEIGHTS_M)
        lane_is_platform = torch.tensor(
            tuple(kind == "platform_up" for kind in STAIRS_LANE_KINDS),
            device=device,
            dtype=torch.bool,
        )
        env._random_terrain_stairs_step_edges = step_edges
        env._random_terrain_stairs_lane_centers = lane_centers
        env._random_terrain_stairs_lane_x_starts = lane_x_starts
        env._random_terrain_stairs_lane_step_h = lane_step_h
        env._random_terrain_stairs_lane_is_platform = lane_is_platform
    else:
        lane_centers = env._random_terrain_stairs_lane_centers
        lane_x_starts = env._random_terrain_stairs_lane_x_starts
        lane_step_h = env._random_terrain_stairs_lane_step_h
        lane_is_platform = env._random_terrain_stairs_lane_is_platform

    lane_idx = torch.argmin((local_x.unsqueeze(-1) - lane_centers).abs(), dim=-1)
    x_m = local_x - lane_x_starts[lane_idx]
    h = lane_step_h[lane_idx]
    remaining = (x_m.unsqueeze(-1) < step_edges).sum(dim=-1).to(dtype)
    stairs_z = remaining * h
    plat_z = torch.where(x_m < STAIRS_LANE_APPROACH_M, torch.zeros_like(h), h)
    return torch.where(lane_is_platform[lane_idx], plat_z, stairs_z)


def _heights(env, local_xy):
    """Return the triangle-interpolated mesh height for reset and rewards."""
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
    x_breaks = getattr(env, "_random_terrain_x_breakpoints", None)
    if x_breaks is None or x_breaks.device != block_x.device:
        x_breaks = block_x.new_tensor(PATCH_X_BREAKPOINTS_M)
        env._random_terrain_x_breakpoints = x_breaks
    y_breaks = getattr(env, "_random_terrain_y_breakpoints", None)
    if y_breaks is None or y_breaks.device != block_y.device:
        y_breaks = block_y.new_tensor(PATCH_Y_BREAKPOINTS_M)
        env._random_terrain_y_breakpoints = y_breaks
    x_index = (
        torch.searchsorted(x_breaks, block_x, right=True) - 1
    ).clamp(0, len(PATCH_X_BREAKPOINTS_M) - 2)
    y_index = (
        torch.searchsorted(y_breaks, block_y, right=True) - 1
    ).clamp(0, len(PATCH_Y_BREAKPOINTS_M) - 2)
    x0 = x_breaks[x_index]
    x1 = x_breaks[x_index + 1]
    y0 = y_breaks[y_index]
    y1 = y_breaks[y_index + 1]
    x_fraction = (
        (block_x - x0) / (x1 - x0)
    ).clamp(0.0, 1.0)
    y_fraction = (
        (block_y - y0) / (y1 - y0)
    ).clamp(0.0, 1.0)

    def vertex_height(vertex_x, vertex_y):
        magnitude = torch.where(
            vertex_x < RAMP_TOP_START_M,
            grade.abs() * (vertex_x - RAMP_START_M),
            torch.where(
                vertex_x <= RAMP_TOP_END_M,
                grade.abs() * (RAMP_TOP_START_M - RAMP_START_M),
                grade.abs() * (RAMP_END_M - vertex_x),
            ),
        ).clamp_min(0.0)
        signed = torch.where(grade < 0.0, -magnitude, magnitude)
        signed = torch.where(
            (grade == 0.0)
            | (vertex_x <= RAMP_START_M)
            | (vertex_x >= RAMP_END_M),
            torch.zeros_like(signed),
            signed,
        )
        y_taper = (
            (BLOCK_WIDTH_M / 2.0 - vertex_y.abs())
            / PATCH_Y_EDGE_TAPER_M
        ).clamp(0.0, 1.0)
        return signed * y_taper

    z00 = vertex_height(x0, y0)
    z10 = vertex_height(x1, y0)
    z01 = vertex_height(x0, y1)
    z11 = vertex_height(x1, y1)
    lower_left = (
        z00 * (1.0 - x_fraction - y_fraction)
        + z10 * x_fraction
        + z01 * y_fraction
    )
    upper_right = (
        z10 * (1.0 - y_fraction)
        + z01 * (1.0 - x_fraction)
        + z11 * (x_fraction + y_fraction - 1.0)
    )
    slope_height = torch.where(
        x_fraction + y_fraction <= 1.0,
        lower_left,
        upper_right,
    )
    if not getattr(env, "_random_terrain_stairs_lanes", False):
        return slope_height
    in_stairs = local_y > STAIRS_LANE_BAND_Y_MIN_M
    if not in_stairs.any():
        return slope_height
    stairs_height = _stairs_lane_heights(env, local_x, slope_height)
    return torch.where(in_stairs, stairs_height, slope_height)


def slope_orientation_targets(env):
    """Return current pitch/roll, terrain-aligned targets, and slope norm.

    Both ``target_pitch`` and ``target_roll`` match ``reset``'s ground-normal
    convention so on flat ground ``target_roll == 0`` while across sloped terrain
    the robot aligns perpendicular to the local surface at any heading.
    """
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
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
    current_roll, current_pitch, yaw = euler_xyz_from_quat(root_quat)
    current_roll = wrap_to_pi(current_roll)
    current_pitch = wrap_to_pi(current_pitch)
    target_pitch = -torch.atan(
        gradient_x * yaw.cos() + gradient_y * yaw.sin()
    )
    target_roll = torch.atan(
        gradient_y * yaw.cos() - gradient_x * yaw.sin()
    )
    slope_norm = torch.sqrt(gradient_x.square() + gradient_y.square())
    return current_pitch, target_pitch, current_roll, target_roll, slope_norm


def slope_pitch_targets(env):
    """Return current pitch, terrain-following pitch target, and slope norm."""
    current_pitch, target_pitch, _, _, slope_norm = slope_orientation_targets(env)
    return current_pitch, target_pitch, slope_norm


def _sample_spawn(env, ids):
    """Sample a patch and a valid local position for each reset environment."""
    import torch

    n = len(ids)
    if getattr(env, "_random_terrain_evaluation_xy_yaw", None) is not None:
        pose = env._random_terrain_evaluation_xy_yaw[ids]
        local_x = pose[:, 0]
        local_y = pose[:, 1]
        yaw = pose[:, 2]
        columns = torch.floor(
            (local_x + MAP_SIZE_M[0] / 2.0) / BLOCK_LENGTH_M
        ).long().clamp(0, MAP_COLS - 1)
        rows = torch.floor(
            (local_y + MAP_SIZE_M[1] / 2.0) / BLOCK_WIDTH_M
        ).long().clamp(0, MAP_ROWS - 1)
        blocks = rows * MAP_COLS + columns
        return blocks, torch.stack((local_x, local_y), dim=-1), yaw

    if env._random_terrain_force_block is not None:
        blocks = env._random_terrain_force_block[ids]
    elif getattr(env, "_random_terrain_block_probs", None) is not None:
        blocks = torch.multinomial(env._random_terrain_block_probs, n, replacement=True)
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
        transition_fraction = getattr(
            env, "_random_terrain_transition_fraction", 0.0
        )
        if transition_fraction > 0.0:
            at_transition = (
                torch.rand(n, device=env.device) < transition_fraction
            )
            if at_transition.any():
                t_count = int(at_transition.sum().item())
                t_cols = columns[at_transition]
                t_xc = x_centers[at_transition]
                t_yc = y_centers[at_transition]
                # Half spawn in the ramp-entry / crest approach zone facing +X;
                # half spawn on the exit ramp facing across the inter-block
                # seam into the adjacent block (or -X for rightmost column 2).
                mode_exit = torch.rand(t_count, device=env.device) < 0.5
                go_neg_x = mode_exit & (t_cols == MAP_COLS - 1)
                go_pos_x_exit = mode_exit & ~go_neg_x
                bx = torch.empty(t_count, device=env.device).uniform_(-7.4, -5.2)
                if go_pos_x_exit.any():
                    bx[go_pos_x_exit] = torch.empty(
                        int(go_pos_x_exit.sum().item()), device=env.device
                    ).uniform_(3.8, 6.4)
                if go_neg_x.any():
                    bx[go_neg_x] = torch.empty(
                        int(go_neg_x.sum().item()), device=env.device
                    ).uniform_(-6.4, -3.8)
                by = torch.empty(t_count, device=env.device).uniform_(-3.5, 3.5)
                t_yaw_rad = getattr(env, "_random_terrain_transition_yaw_rad", 0.30)
                tyaw = torch.empty(t_count, device=env.device).uniform_(
                    -t_yaw_rad, t_yaw_rad
                )
                tyaw = torch.where(go_neg_x, tyaw + math.pi, tyaw)
                tyaw = torch.atan2(torch.sin(tyaw), torch.cos(tyaw))
                local_x[at_transition] = t_xc + bx
                local_y[at_transition] = t_yc + by
                yaw[at_transition] = tyaw
                env._random_terrain_transition_spawn_count += t_count
        stairs_fraction = getattr(env, "_random_terrain_stairs_fraction", 0.0)
        if stairs_fraction > 0.0 and getattr(env, "_random_terrain_stairs_lanes", False):
            at_stairs = torch.rand(n, device=env.device) < stairs_fraction
            if at_stairs.any():
                s_count = int(at_stairs.sum().item())
                lane_x_starts = local_x.new_tensor(STAIRS_LANE_X_STARTS_M)
                lane_probs = getattr(env, "_random_terrain_stairs_lane_probs", None)
                if lane_probs is not None:
                    lane_idx = torch.multinomial(
                        lane_probs, s_count, replacement=True
                    )
                else:
                    lane_idx = torch.randint(
                        len(STAIRS_LANE_X_STARTS_M),
                        (s_count,),
                        device=env.device,
                    )
                x_low, x_high = getattr(
                    env, "_random_terrain_stairs_x_range", (1.0, 2.5)
                )
                x_in_lane = torch.empty(s_count, device=env.device).uniform_(
                    x_low, x_high
                )
                local_x[at_stairs] = lane_x_starts[lane_idx] + x_in_lane
                local_y[at_stairs] = torch.empty(s_count, device=env.device).uniform_(
                    STAIRS_LANE_Y_CENTER_M - 0.6,
                    STAIRS_LANE_Y_CENTER_M + 0.6,
                )
                stairs_yaw_rad = float(
                    getattr(env, "_random_terrain_stairs_yaw_rad", 0.20)
                )
                yaw[at_stairs] = torch.empty(s_count, device=env.device).uniform_(
                    -stairs_yaw_rad, stairs_yaw_rad
                )
                blocks[at_stairs] = 0
                env._random_terrain_stairs_spawn_count += s_count
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
        "transition_spawns": getattr(
            env, "_random_terrain_transition_spawn_count", 0
        ),
        "stairs_spawns": getattr(
            env, "_random_terrain_stairs_spawn_count", 0
        ),
        "resets": env._random_terrain_reset_count,
        "forced_evaluation_blocks": forced,
        "peak_initial_tire_penetration_m": env._random_terrain_peak_penetration_m,
    }
