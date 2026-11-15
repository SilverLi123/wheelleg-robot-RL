#!/usr/bin/env python3
"""Independent triangle-level checks for the terrain-only stair/platform meshes."""

from __future__ import annotations

from collections import Counter, defaultdict
import importlib.util
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (ROOT / "project/source/agent_tasks/agent_tasks/direct/wheelbipe/"
               "wheelbipe_own/obstacle_terrain.py")
spec = importlib.util.spec_from_file_location("obstacle_terrain_test", MODULE_PATH)
terrain = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = terrain
spec.loader.exec_module(terrain)


def cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def ray_heights(vertices, faces, x, y):
    """Independent barycentric intersection with a vertical line through x/y."""
    hits = []
    for face in faces:
        a, b, c = (vertices[index] for index in face)
        u, v = subtract(b, a), subtract(c, a)
        det = u[0] * v[1] - u[1] * v[0]
        if abs(det) < 1e-12:
            continue
        dx, dy = x - a[0], y - a[1]
        r = (dx * v[1] - dy * v[0]) / det
        s = (u[0] * dy - u[1] * dx) / det
        if r >= -1e-10 and s >= -1e-10 and r + s <= 1.0 + 1e-10:
            hits.append(a[2] + r * u[2] + s * v[2])
    return hits


class ObstacleGeometryTest(unittest.TestCase):
    def cases(self):
        return ([terrain.ObstacleSpec("stairs_down", h) for h in (0.03, 0.06, 0.09)]
                + [terrain.ObstacleSpec("platform_up", h) for h in (0.06, 0.12, 0.18)])

    def test_closed_outward_mesh_with_no_duplicate_faces(self):
        for cfg in self.cases():
            with self.subTest(kind=cfg.kind, height=cfg.height_m):
                vertices, faces = terrain.mesh_data(cfg)
                edges = Counter()
                directions = Counter()
                unique_faces = set()
                volume = 0.0
                for face in faces:
                    self.assertEqual(len(set(face)), 3)
                    key = tuple(sorted(face))
                    self.assertNotIn(key, unique_faces)
                    unique_faces.add(key)
                    a, b, c = (vertices[index] for index in face)
                    normal = cross(subtract(b, a), subtract(c, a))
                    self.assertGreater(sum(value * value for value in normal), 1e-18)
                    volume += sum(x * y for x, y in zip(a, cross(b, c))) / 6.0
                    for i, j in zip(face, (face[1], face[2], face[0])):
                        edges[tuple(sorted((i, j)))] += 1
                        directions[(i, j)] += 1
                self.assertEqual(set(edges.values()), {2})
                for i, j in edges:
                    self.assertEqual(directions[i, j], 1)
                    self.assertEqual(directions[j, i], 1)
                self.assertGreater(volume, 0.0)
                self.assertEqual(len(vertices) - len(edges) + len(faces), 2)

    def test_rays_match_treads_and_platform_without_hidden_floor(self):
        for cfg in self.cases():
            vertices, faces = terrain.mesh_data(cfg)
            if cfg.kind == "stairs_down":
                probes = [(1.0, cfg.step_count * cfg.height_m)]
                probes += [(cfg.approach_m + (i + 0.5) * cfg.tread_m,
                            (cfg.step_count - 1 - i) * cfg.height_m)
                           for i in range(cfg.step_count)]
                probes += [(cfg.length_m - 0.5, 0.0)]
            else:
                probes = [(1.0, 0.0), (cfg.approach_m - 0.01, 0.0),
                          (cfg.approach_m + 0.01, cfg.height_m),
                          (cfg.length_m - 0.5, cfg.height_m)]
            for x, expected in probes:
                for y in (0.1, cfg.width_m / 2, cfg.width_m - 0.1):
                    hits = ray_heights(vertices, faces, x, y)
                    self.assertTrue(hits)
                    self.assertAlmostEqual(max(hits), expected)
                    # Closed solid has just its top and buried bottom here.
                    self.assertTrue(all(math.isclose(z, expected, abs_tol=1e-9)
                                        or math.isclose(z, cfg.base_z_m, abs_tol=1e-9)
                                        for z in hits))
                    self.assertAlmostEqual(cfg.ground_height(x, y), expected)
            self.assertFalse(ray_heights(vertices, faces, -0.1, 1.0))
            self.assertFalse(ray_heights(vertices, faces, 1.0, cfg.width_m + 0.1))

    def test_exact_vertical_faces_and_no_accidental_ramps(self):
        for cfg in self.cases():
            vertices, faces = terrain.mesh_data(cfg)
            risers = defaultdict(float)
            for face in faces:
                a, b, c = (vertices[index] for index in face)
                normal = cross(subtract(b, a), subtract(c, a))
                if abs(normal[2]) > 1e-9:
                    self.assertAlmostEqual(normal[0], 0.0)
                    self.assertAlmostEqual(normal[1], 0.0)
                if 0.0 < a[0] < cfg.length_m and a[0] == b[0] == c[0]:
                    expected_sign = 1.0 if cfg.kind == "stairs_down" else -1.0
                    self.assertGreater(normal[0] * expected_sign, 0.0)
                    risers[a[0]] += abs(normal[0]) / 2.0
            expected_count = cfg.step_count if cfg.kind == "stairs_down" else 1
            self.assertEqual(len(risers), expected_count)
            for area in risers.values():
                self.assertAlmostEqual(area, cfg.width_m * cfg.height_m)

    def test_lab_centering_and_ground_markers(self):
        for cfg in self.cases():
            vertices, faces = terrain.mesh_data(cfg)
            centered = [(x - cfg.length_m / 2, y - cfg.width_m / 2, z)
                        for x, y, z in vertices]
            self.assertAlmostEqual(min(v[0] for v in centered), -cfg.length_m / 2)
            self.assertAlmostEqual(max(v[0] for v in centered), cfg.length_m / 2)
            markers = terrain.manifest(cfg)["isaac_single_tile_markers"]
            for x, y, z in markers.values():
                self.assertAlmostEqual(max(ray_heights(centered, faces, x, y)), z)
            self.assertLess(markers["spawn_ground_m"][0],
                            cfg.approach_m - cfg.length_m / 2)
            self.assertGreater(markers["landing_ground_m"][0],
                               cfg.approach_m - cfg.length_m / 2)

    def test_single_step_and_custom_dimensions(self):
        cfg = terrain.ObstacleSpec("stairs_down", 0.04, step_count=1,
                                   width_m=1.3, approach_m=2.2, landing_m=1.4,
                                   tread_m=0.4)
        vertices, faces = terrain.mesh_data(cfg)
        self.assertAlmostEqual(cfg.length_m, 4.0)
        self.assertAlmostEqual(max(ray_heights(vertices, faces, 1.0, 0.7)), 0.04)
        self.assertAlmostEqual(max(ray_heights(vertices, faces, 2.4, 0.7)), 0.0)

    def test_invalid_parameters_and_outside_queries_fail(self):
        for kwargs in ({"height_m": 0.0}, {"height_m": float("nan")},
                       {"width_m": -1.0}, {"step_count": 0}, {"step_count": True},
                       {"kind": "ramp"}, {"base_z_m": 0.0}):
            values = {"kind": "stairs_down", "height_m": 0.06}
            values.update(kwargs)
            with self.assertRaises(ValueError):
                terrain.ObstacleSpec(**values)
        cfg = terrain.ObstacleSpec("stairs_down", 0.06)
        for xy in ((-0.1, 0.5), (0.5, -0.1), (cfg.length_m + 0.1, 0.5)):
            with self.assertRaises(ValueError):
                cfg.ground_height(*xy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
