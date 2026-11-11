"""Straight up/down ramp family for mixed-slope adaptation.

The robot spawns on a short flat run-up in every tile.  Slopes begin after
``SLOPE_OFFSET_M`` metres, so the policy has enough room to settle after reset.
Only the terrain geometry changes here; robot, observation, action, actuator,
physics and domain-randomization settings remain owned by the existing task.
"""

from __future__ import annotations

import os

import numpy as np


# Robot origin measured from the left edge of a generated tile.
START_X_M = 2.5
# Slope begins two metres ahead of the robot, leaving a flat settling area.
SLOPE_OFFSET_M = 2.0
# The generator places the terrain origin at the tile's flat run-up.
DOWN_SLOPE_CLEARANCE_M = 0.015

# Full-difficulty target grades.  The curriculum scales these from 60% to
# 100%, so early rows are gentler copies of the same surface family.
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
TILES = tuple(GRADES)
TRAIN_TILES = TILES
EVAL_TILES = TILES
TILE_LENGTH_M = 20.0
TILE_WIDTH_M = 10.0
TRAIN_ROWS = 5
TRAIN_COLUMNS = 12
TRAIN_PROPORTIONS = {
    "flat": 0.28,
    "up06": 0.09,
    "up10": 0.09,
    "up14": 0.09,
    "up18": 0.09,
    "down06": 0.09,
    "down10": 0.09,
    "down14": 0.09,
    "down18": 0.09,
}


def effective_grade(tile: str, difficulty: float) -> float:
    """Return the full grade for an evaluation tile or curriculum-scaled grade."""
    if tile not in GRADES:
        raise ValueError(f"Unknown mixed-slope tile: {tile}")
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"difficulty must be in [0, 1], got {difficulty}")
    return GRADES[tile] * (0.60 + 0.40 * difficulty)


def surface_height(tile: str, x_m: float, difficulty: float = 1.0) -> float:
    """Height relative to the flat run-up at horizontal position ``x_m``."""
    slope_start = START_X_M + SLOPE_OFFSET_M
    return effective_grade(tile, difficulty) * max(0.0, x_m - slope_start)


def _vertices_and_faces(tile: str, difficulty: float, cfg):
    """Return the six-vertex ramp strip and its two triangles."""
    length, width = cfg.size
    slope_start = START_X_M + SLOPE_OFFSET_M
    # Three rows preserve an exact flat-to-slope transition instead of a wavy
    # height-field approximation.  Both y edges use identical heights, so the
    # surface has no unintended lateral roll.
    rows = (
        (0.0, surface_height(tile, 0.0, difficulty)),
        (slope_start, surface_height(tile, slope_start, difficulty)),
        (length, surface_height(tile, length, difficulty)),
    )
    vertices = [(x, y, z) for x, z in rows for y in (0.0, width)]
    faces = [(2 * i, 2 * i + 2, 2 * i + 1) for i in range(len(rows) - 1)]
    faces += [(2 * i + 1, 2 * i + 2, 2 * i + 3) for i in range(len(rows) - 1)]
    return vertices, faces


def _origin(tile: str, width: float) -> np.ndarray:
    clearance = DOWN_SLOPE_CLEARANCE_M if tile.startswith("down") else 0.0
    return np.array([START_X_M, width / 2.0, clearance], dtype=np.float64)


def _mesh(tile: str, difficulty: float, cfg):
    """Build one longitudinally sloped quad strip with a shared lateral height."""
    import trimesh

    length, width = cfg.size
    vertices, faces = _vertices_and_faces(tile, difficulty, cfg)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return [mesh], _origin(tile, width)


def _terrain_function(tile: str):
    def generate(difficulty, cfg):
        return _mesh(tile, float(difficulty), cfg)

    generate.__name__ = f"{tile}_ramp_mesh"
    return generate


def tile_functions(tiles=TILES) -> dict:
    """Return one terrain-callable per requested tile."""
    return {tile: _terrain_function(tile) for tile in tiles}


def configure(cfg) -> None:
    """Install the mixed-slope generator when the selected profile requests it."""
    profile = cfg.experiment_profile_settings
    if not profile.get("mixed_slopes"):
        return
    supported_profiles = {
        "mixed_slopes_20261111",
        "mixed_slopes_forward_20261111",
    }
    if cfg.experiment_profile_name not in supported_profiles:
        raise ValueError(
            "mixed_slopes requires an explicit mixed-slope profile: "
            f"{sorted(supported_profiles)}"
        )

    from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg

    pieces = {
        tile: MeshPlaneTerrainCfg(
            proportion=TRAIN_PROPORTIONS[tile],
            function=tile_functions((tile,))[tile],
        )
        for tile in TRAIN_TILES
    }
    cfg.terrain = cfg.terrain.copy()
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=int(profile.get("terrain_seed", 20261111)),
        size=(TILE_LENGTH_M, TILE_WIDTH_M),
        border_width=2.0,
        num_rows=TRAIN_ROWS,
        num_cols=TRAIN_COLUMNS,
        curriculum=True,
        horizontal_scale=0.1,
        vertical_scale=0.001,
        use_cache=False,
        difficulty_range=(0.0, 1.0),
        sub_terrains=pieces,
    )
    cfg.scene.terrain = cfg.terrain.copy()


def configure_evaluation(cfg, tiles=EVAL_TILES) -> None:
    """Replace a configured generator with one full-difficulty tile per type."""
    if tuple(tiles) != EVAL_TILES:
        raise ValueError(f"Evaluation tiles must be {EVAL_TILES}, got {tiles}")

    from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg

    pieces = {
        tile: MeshPlaneTerrainCfg(
            proportion=1.0 / len(EVAL_TILES),
            function=tile_functions(EVAL_TILES)[tile],
        )
        for tile in EVAL_TILES
    }
    cfg.terrain = cfg.terrain.copy()
    cfg.terrain.terrain_type = "generator"
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=20261112,
        size=(TILE_LENGTH_M, TILE_WIDTH_M),
        border_width=1.0,
        num_rows=1,
        num_cols=len(EVAL_TILES),
        curriculum=True,
        horizontal_scale=0.1,
        vertical_scale=0.001,
        use_cache=False,
        difficulty_range=(1.0, 1.0),
        sub_terrains=pieces,
    )
    cfg.scene.terrain = cfg.terrain.copy()


def profile_requested() -> bool:
    """Small diagnostic used by launch scripts without importing Isaac Lab."""
    return bool(int(os.environ.get("WHEELBIPE_MIXED_SLOPES_CHECK", "0")))
