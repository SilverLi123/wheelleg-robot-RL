#!/usr/bin/env python3
"""Fast checks for the fixed random-terrain map contract."""

from __future__ import annotations

import importlib.util
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
    assert len(vertices) == len(terrain.TERRAIN_TYPES) * 12
    assert len(faces) == len(terrain.TERRAIN_TYPES) * 10

    # Sample just inside both sides of every vertical block boundary.  Heights
    # must approach zero from both patches, preventing an impassable ledge.
    for column in (1, 2):
        boundary = -terrain.MAP_SIZE_M[0] / 2.0 + column * terrain.BLOCK_LENGTH_M
        for y in (-17.9, 0.0, 17.9):
            assert abs(terrain.height_at_local(boundary - 1e-6, y, first)) < 1e-6
            assert abs(terrain.height_at_local(boundary + 1e-6, y, first)) < 1e-6

    # A maximum-grade crest must be 1.08 m: 18 percent over the six-metre rise.
    assert abs(terrain.patch_height(0.0, "up18") - 1.08) < 1e-12
    assert abs(terrain.patch_height(0.0, "down18") + 1.08) < 1e-12
    print("random terrain geometry contract: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"random terrain geometry contract: FAIL: {exc}", file=sys.stderr)
        raise
