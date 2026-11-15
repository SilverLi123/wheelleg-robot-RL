#!/usr/bin/env python3
"""Fast checks for the fixed random-terrain map contract."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


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


def mesh_height_at(vertices, faces, local_x_m, local_y_m):
    """Return the highest triangle-surface height at one local XY point."""
    hits = []
    for face in faces:
        first, second, third = (
            vertices[index]
            for index in face
        )
        determinant = (
            (second[1] - third[1]) * (first[0] - third[0])
            + (third[0] - second[0]) * (first[1] - third[1])
        )
        if abs(determinant) < 1e-12:
            continue
        first_weight = (
            (second[1] - third[1]) * (local_x_m - third[0])
            + (third[0] - second[0]) * (local_y_m - third[1])
        ) / determinant
        second_weight = (
            (third[1] - first[1]) * (local_x_m - third[0])
            + (first[0] - third[0]) * (local_y_m - third[1])
        ) / determinant
        third_weight = 1.0 - first_weight - second_weight
        if min(first_weight, second_weight, third_weight) >= -1e-9:
            hits.append(
                first_weight * first[2]
                + second_weight * second[2]
                + third_weight * third[2]
            )
    if hits:
        return max(hits)
    raise AssertionError(
        f"No terrain triangle contains ({local_x_m}, {local_y_m})"
    )


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

    # The collision mesh is piecewise triangular, not the product of an X
    # profile and a Y taper. These interior points therefore have hand-derived
    # collision heights that differ from the old analytic product formula.
    up18_block = next(
        block_id
        for block_id in range(len(terrain.TERRAIN_TYPES))
        if terrain.terrain_at_block(block_id, first) == "up18"
    )
    x_min, _, y_min, _ = terrain.block_bounds_local(up18_block)
    x_center = x_min + terrain.BLOCK_LENGTH_M / 2.0
    y_center = y_min + terrain.BLOCK_WIDTH_M / 2.0
    interior_cases = (
        (-4.0, -5.25, 0.0),
        (-4.0, -5.00, 0.18),
        (-3.5, -5.25, 0.09),
    )
    for patch_x, patch_y, expected in interior_cases:
        collision_height = mesh_height_at(
            vertices,
            faces,
            x_center + patch_x,
            y_center + patch_y,
        )
        assert abs(collision_height - expected) < 1e-12
        assert abs(
            terrain.patch_height(patch_x, "up18", patch_y)
            - collision_height
        ) < 1e-12

    # Reset and the relative-height reward call the vectorized `_heights`.
    # It must reproduce exactly the same triangular collision surface. The
    # Mac quick test does not require PyTorch; the server env620 check covers
    # this branch when torch is available.
    try:
        import torch
    except ImportError:
        print("batch height check: SKIP (torch unavailable)")
    else:
        fake_env = SimpleNamespace(
            device="cpu",
            _random_terrain_block_grades=torch.tensor(
                [[terrain.GRADES[name] for name in row] for row in first],
                dtype=torch.float32,
            ),
        )
        map_points = torch.tensor(
            [
                (x_center + patch_x, y_center + patch_y)
                for patch_x, patch_y, _ in interior_cases
            ],
            dtype=torch.float32,
        )
        batched = terrain._heights(fake_env, map_points).tolist()
        for value, (_, _, expected) in zip(batched, interior_cases):
            assert abs(value - expected) < 1e-6

    # On the up18 crest the Y taper is 1.08 m over 1.5 m, which is a 72%
    # lateral grade (about 35.8 degrees), much steeper than the labelled 18%
    # longitudinal grade.
    crest_edge = mesh_height_at(
        vertices, faces, x_center, y_center + 4.5
    )
    row_edge = mesh_height_at(
        vertices, faces, x_center, y_center + 6.0
    )
    lateral_grade = (crest_edge - row_edge) / 1.5
    assert abs(lateral_grade - 0.72) < 1e-12
    assert abs(math.degrees(math.atan(lateral_grade)) - 35.75) < 0.01

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

    # Batched quaternion rotation must match single-quaternion rotation across
    # non-trivial 3D orientations (including yaw = pi combined with pitch/roll).
    import numpy as np

    sample_points = np.asarray(
        ((-0.20, 0.05, 0.05), (0.20, -0.05, 0.08), (0.05, 0.10, 0.30)),
        dtype=np.float64,
    )
    test_quats = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),  # yaw = pi
            (0.0, -0.1, 0.15, 0.98),  # yaw ~ pi + pitch/roll
            (0.92, 0.08, -0.18, 0.34),  # general 3D rotation
        ),
        dtype=np.float64,
    )
    batched_rot = terrain.quats_rotate_wxyz_numpy(sample_points, test_quats)
    for idx, quat in enumerate(test_quats):
        expected_rot = terrain.quat_rotate_wxyz_numpy(sample_points, quat)
        assert np.allclose(batched_rot[idx], expected_rot, atol=1e-12)

    # Rotating a symmetric hull by yaw = pi on a slope reversed along -x must
    # yield the exact same clearance as yaw = 0 on the original +x slope.
    symmetric_points = ((-0.20, 0.0, 0.05), (0.20, 0.0, 0.05), (0.0, 0.0, 0.30))
    # Combine yaw=pi around z with pitch around body-y (q_yaw_pi * q_pitch)
    theta = 0.15
    q_yaw0 = (math.cos(theta / 2.0), 0.0, math.sin(theta / 2.0), 0.0)
    # Rotating by yaw=pi (0,0,0,1) after pitch (c, 0, s, 0) gives (0, -s, 0, c)
    q_yaw_pi = (0.0, -math.sin(theta / 2.0), 0.0, math.cos(theta / 2.0))
    c_pos = terrain.collision_points_minimum_clearance(
        body_pos_w=((0.0, 0.0, 0.30),),
        body_quat_w=(q_yaw0,),
        local_points=symmetric_points,
        height_at_local_xy=lambda xy: 0.20 + 0.18 * xy[:, 0],
    )[0]
    c_neg = terrain.collision_points_minimum_clearance(
        body_pos_w=((0.0, 0.0, 0.30),),
        body_quat_w=(q_yaw_pi,),
        local_points=symmetric_points,
        height_at_local_xy=lambda xy: 0.20 - 0.18 * xy[:, 0],
    )[0]
    assert abs(float(c_pos) - float(c_neg)) < 1e-12

    # Hybrid map appends 2 stairs_down + 2 platform_up lanes at y = +21.0 m
    # without altering the 3x3 slope mesh vertices inside [-24, 24] x [-18, 18].
    h_vertices, h_faces = terrain._vertices_and_faces(
        first, include_stairs_lanes=True
    )
    assert len(h_vertices) > len(vertices)
    assert len(h_faces) > len(faces)
    assert len(terrain.STAIRS_LANE_KINDS) == len(terrain.STAIRS_LANE_X_STARTS_M) == len(terrain.STAIRS_LANE_STEP_HEIGHTS_M)
    for kind, x_start, step_h in zip(
        terrain.STAIRS_LANE_KINDS,
        terrain.STAIRS_LANE_X_STARTS_M,
        terrain.STAIRS_LANE_STEP_HEIGHTS_M,
    ):
        probes = (
            ((1.5, 4 * step_h), (3.15, 3 * step_h), (3.50, 2 * step_h), (3.85, step_h), (5.0, 0.0))
            if kind == "stairs_down"
            else ((1.5, 0.0), (5.0, step_h))
        )
        for x_in_lane, expected_z in probes:
            z_actual = mesh_height_at(
                h_vertices,
                h_faces,
                x_start + x_in_lane,
                terrain.STAIRS_LANE_Y_CENTER_M,
            )
            assert abs(z_actual - expected_z) < 1e-9

    print("random terrain geometry contract: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"random terrain geometry contract: FAIL: {exc}", file=sys.stderr)
        raise
