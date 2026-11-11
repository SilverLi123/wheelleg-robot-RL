#!/usr/bin/env python3
"""Fast geometry test for the mixed-slope mesh contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "project/source/agent_tasks/agent_tasks/direct/wheelbipe/"
    "wheelbipe_own/mixed_slopes.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("mixed_slopes_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    mixed = load_module()

    class Config:
        size = (mixed.TILE_LENGTH_M, mixed.TILE_WIDTH_M)

    cfg = Config()
    assert abs(sum(mixed.TRAIN_PROPORTIONS.values()) - 1.0) < 1e-12

    for tile in mixed.EVAL_TILES:
        vertices, faces = mixed._vertices_and_faces(tile, 1.0, cfg)
        assert len(vertices) == 6
        assert len(faces) == 4
        assert sum(y == 0.0 for _, y, _ in vertices) == 3
        assert sum(y == mixed.TILE_WIDTH_M for _, y, _ in vertices) == 3
        # Both lateral edges must have identical elevation: no unintended roll.
        for left, right in ((0, 1), (2, 3), (4, 5)):
            assert vertices[left][0] == vertices[right][0]
            assert vertices[left][2] == vertices[right][2]

        slope_start = mixed.START_X_M + mixed.SLOPE_OFFSET_M
        assert abs(vertices[2][2] - mixed.surface_height(tile, slope_start)) < 1e-12
        assert (
            abs(vertices[4][2] - mixed.surface_height(tile, mixed.TILE_LENGTH_M))
            < 1e-12
        )
        expected_clearance = (
            mixed.DOWN_SLOPE_CLEARANCE_M if tile.startswith("down") else 0.0
        )
        origin = mixed._origin(tile, mixed.TILE_WIDTH_M)
        assert abs(origin[0] - mixed.START_X_M) < 1e-12
        assert abs(origin[1] - mixed.TILE_WIDTH_M / 2.0) < 1e-12
        assert abs(origin[2] - expected_clearance) < 1e-12

    try:
        import trimesh  # noqa: F401
    except ModuleNotFoundError:
        print("trimesh runtime import: SKIP (not installed on this interpreter)")
    else:
        meshes, origin = mixed._mesh("up18", 1.0, cfg)
        assert len(meshes) == 1 and len(meshes[0].faces) == 4
        assert abs(origin[2]) < 1e-12
        print("trimesh runtime import: PASS")

    print("mixed_slopes geometry contract: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"mixed_slopes geometry contract: FAIL: {exc}", file=sys.stderr)
        raise
