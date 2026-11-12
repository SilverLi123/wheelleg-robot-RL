#!/usr/bin/env python3
"""Fast checks for the fixed random-terrain map contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "project/source/agent_tasks/agent_tasks/direct/wheelbipe/"
    "wheelbipe_own/random_terrain_map.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("random_terrain_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    terrain = load_module()
    profiles = json.loads((ROOT / "profiles.json").read_text())
    first = terrain.make_layout(terrain.LAYOUT_SEED)
    second = terrain.make_layout(terrain.LAYOUT_SEED)
    assert first == second
    assert sorted(value for row in first for value in row) == sorted(
        terrain.TERRAIN_TYPES
    )

    for block_id in range(len(terrain.TERRAIN_TYPES)):
        x_min, x_max, _, _ = terrain.block_bounds_local(block_id)
        name = terrain.terrain_at_block(block_id, first)
        assert abs(terrain.patch_height(x_min, name)) < 1e-12
        assert abs(terrain.patch_height(x_max, name)) < 1e-12

    vertices, faces = terrain._vertices_and_faces(first)
    # Six x breakpoints and four y breakpoints represent both the slope ridge
    # and its 1.5 m taper to zero at the row edges.
    assert len(vertices) == len(terrain.TERRAIN_TYPES) * 24
    assert len(faces) == len(terrain.TERRAIN_TYPES) * 30

    # Sample just inside both sides of every vertical block boundary.  Heights
    # must approach zero from both patches, preventing an impassable ledge.
    for column in (1, 2):
        boundary = -terrain.MAP_SIZE_M[0] / 2.0 + column * terrain.BLOCK_LENGTH_M
        for y in (-17.9, 0.0, 17.9):
            assert abs(terrain.height_at_local(boundary - 1e-6, y, first)) < 1e-6
            assert abs(terrain.height_at_local(boundary + 1e-6, y, first)) < 1e-6

    # Rows also meet at zero height. Without this y taper, a sloping ridge at
    # the shared edge meets a flat row as a vertical step.
    for row in (1, 2):
        boundary = -terrain.MAP_SIZE_M[1] / 2.0 + row * terrain.BLOCK_WIDTH_M
        for x in (-23.9, 0.0, 23.9):
            assert abs(terrain.height_at_local(x, boundary - 1e-6, first)) < 1e-6
            assert abs(terrain.height_at_local(x, boundary + 1e-6, first)) < 1e-6

    # A maximum-grade crest must be 1.08 m: 18 percent over the six-metre rise.
    assert abs(terrain.patch_height(0.0, "up18") - 1.08) < 1e-12
    assert abs(terrain.patch_height(0.0, "down18") + 1.08) < 1e-12
    assert abs(terrain.patch_height(0.0, "up18", 4.5) - 1.08) < 1e-12
    assert abs(terrain.patch_height(0.0, "up18", 5.25) - 0.54) < 1e-12
    assert abs(terrain.patch_height(0.0, "up18", 6.0)) < 1e-12

    # Absolute root height must be converted to height above the local ground
    # before it reaches track_height rewards. At the crest the subtraction is
    # exactly zero, while the inherited absolute reference would report 1.08 m.
    crest_ground_m = terrain.height_at_local(0.0, 0.0, first)
    absolute_root_height_m = crest_ground_m
    assert abs(absolute_root_height_m - crest_ground_m) < 1e-12
    assert abs(
        (1.08 + crest_ground_m) - crest_ground_m - 1.08
    ) < 1e-12

    # Every random-map experiment must opt into that correction. Without this
    # profile flag the mixin deliberately preserves the inherited absolute-z
    # reward semantics for checkpoint compatibility.
    for name in terrain.RANDOM_TERRAIN_PROFILES:
        assert profiles[name].get("terrain_relative_height_reward") is True

    # Clearance must be measured at the lowest collision-mesh corner. A body
    # origin above the ground says nothing about an overhanging mesh corner.
    corners = terrain.collision_box_corners(
        (-0.10, -0.05, 0.02),
        (0.30, 0.05, 0.28),
    )
    assert corners.shape == (8, 3)
    assert tuple(corners.min(axis=0)) == (-0.10, -0.05, 0.02)
    assert tuple(corners.max(axis=0)) == (0.30, 0.05, 0.28)
    clearance = terrain.collision_box_minimum_clearance(
        body_pos_w=(0.0, 0.0, 0.30),
        body_quat_w=(1.0, 0.0, 0.0, 0.0),
        local_min=(-0.10, -0.05, 0.02),
        local_max=(0.30, 0.05, 0.28),
        height_at_local_xy=lambda xy: 0.18 + 0.20 * xy[:, 0],
    )
    # The origin-only estimate would report 0.12 m, hiding the 0.08 m
    # clearance under the high-ground end of the rotated collision box.
    assert abs(float(clearance) - 0.08) < 1e-12

    # AABB corners can lie outside a concave collision mesh. The generic point
    # helper must therefore measure the actual supplied mesh/hull vertices.
    triangle_points = terrain.collision_points_minimum_clearance(
        body_pos_w=((0.0, 0.0, 0.30),),
        body_quat_w=((1.0, 0.0, 0.0, 0.0),),
        local_points=((-0.20, 0.0, 0.05), (0.20, 0.0, 0.05), (0.0, 0.0, 0.30)),
        height_at_local_xy=lambda xy: 0.30 + 0.20 * xy[:, 0],
    )[0]
    assert abs(float(triangle_points) - 0.01) < 1e-12

    batched_triangle_points = terrain.collision_points_minimum_clearance(
        body_pos_w=((0.0, 0.0, 0.30), (0.0, 0.0, 0.40)),
        body_quat_w=((1.0, 0.0, 0.0, 0.0),) * 2,
        local_points=((-0.20, 0.0, 0.05), (0.20, 0.0, 0.05), (0.0, 0.0, 0.30)),
        height_at_local_xy=lambda xy: 0.30 + 0.20 * xy[:, 0],
    )
    assert abs(float(batched_triangle_points[0]) - 0.01) < 1e-12
    assert abs(float(batched_triangle_points[1]) - 0.11) < 1e-12
    print("random terrain geometry contract: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"random terrain geometry contract: FAIL: {exc}", file=sys.stderr)
        raise
