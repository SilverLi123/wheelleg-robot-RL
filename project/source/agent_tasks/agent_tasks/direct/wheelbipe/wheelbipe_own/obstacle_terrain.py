"""Terrain-only preparations for descending stairs and jumping onto a platform.

Geometry uses Isaac Lab's sub-terrain convention: x in [0, length], y in
[0, width], z in metres. No task, robot, reward or training configuration is
changed on import. Importing this module needs only the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObstacleSpec:
    kind: str
    height_m: float
    width_m: float = 2.0
    approach_m: float = 3.0
    landing_m: float = 3.0
    step_count: int = 4
    tread_m: float = 0.35
    base_z_m: float = -0.20

    def __post_init__(self):
        if self.kind not in ("stairs_down", "platform_up"):
            raise ValueError("kind must be stairs_down or platform_up")
        for name in ("height_m", "width_m", "approach_m", "landing_m", "tread_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(self.step_count, bool) or not isinstance(self.step_count, int)
                or self.step_count < 1):
            raise ValueError("step_count must be a positive integer")
        if not math.isfinite(self.base_z_m) or self.base_z_m >= 0.0:
            raise ValueError("base_z_m must be finite and below the lower landing")

    @property
    def length_m(self):
        stairs = self.step_count * self.tread_m if self.kind == "stairs_down" else 0.0
        return self.approach_m + stairs + self.landing_m

    @property
    def total_height_m(self):
        return self.step_count * self.height_m if self.kind == "stairs_down" else self.height_m

    def profile(self):
        """Return the exact x/z surface, including vertical risers (no ramps)."""
        if self.kind == "platform_up":
            return [(0.0, 0.0), (self.approach_m, 0.0),
                    (self.approach_m, self.height_m), (self.length_m, self.height_m)]
        points = [(0.0, self.total_height_m), (self.approach_m, self.total_height_m)]
        for index in range(self.step_count):
            x = self.approach_m + index * self.tread_m
            z = (self.step_count - index - 1) * self.height_m
            points.extend(((x, z), (x + self.tread_m, z)))
        points.append((self.length_m, 0.0))
        return points

    def ground_height(self, x_m, y_m):
        """Right-hand top height at a riser; outside the lane is not ground.

        A height query cannot describe a vertical face. Collision uses the full
        triangle mesh; spawn and landing markers deliberately avoid riser edges.
        """
        if not (0.0 <= x_m <= self.length_m and 0.0 <= y_m <= self.width_m):
            raise ValueError("point is outside this obstacle lane")
        if self.kind == "platform_up":
            return 0.0 if x_m < self.approach_m else self.height_m
        if x_m < self.approach_m:
            return self.total_height_m
        remaining = sum(x_m < self.approach_m + i * self.tread_m
                        for i in range(self.step_count))
        return remaining * self.height_m

    def markers(self):
        """Ground positions, not robot root poses; a reset bank is still needed."""
        mid_y = self.width_m / 2.0
        spawn_x = self.approach_m / 2.0
        landing_x = self.length_m - self.landing_m / 2.0
        return {
            "spawn_ground_m": (spawn_x, mid_y, self.ground_height(spawn_x, mid_y)),
            "approach_ground_m": (0.8 * self.approach_m, mid_y,
                                   self.ground_height(0.8 * self.approach_m, mid_y)),
            "landing_ground_m": (landing_x, mid_y, self.ground_height(landing_x, mid_y)),
        }


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _side_polygon(spec):
    # Counter-clockwise in the x/z plane, including the buried base. Remove
    # collinear tread/landing joins before triangulating the two side walls.
    polygon = [(0.0, spec.base_z_m), (spec.length_m, spec.base_z_m)]
    polygon.extend(reversed(spec.profile()))
    cleaned = []
    for point in polygon:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
        while len(cleaned) >= 3 and abs(_cross(*cleaned[-3:])) < 1e-12:
            cleaned.pop(-2)
    return cleaned


def _triangulate(polygon):
    """Ear clipping for the simple counter-clockwise x/z side polygon."""
    remaining = list(range(len(polygon)))
    triangles = []
    while len(remaining) > 3:
        for index, current in enumerate(remaining):
            previous = remaining[index - 1]
            following = remaining[(index + 1) % len(remaining)]
            a, b, c = (polygon[i] for i in (previous, current, following))
            if _cross(a, b, c) <= 1e-12:
                continue
            contains_vertex = any(
                _cross(a, b, polygon[other]) >= -1e-12
                and _cross(b, c, polygon[other]) >= -1e-12
                and _cross(c, a, polygon[other]) >= -1e-12
                for other in remaining if other not in (previous, current, following)
            )
            if contains_vertex:
                continue
            triangles.append((previous, current, following))
            remaining.pop(index)
            break
        else:
            raise ValueError("Obstacle side polygon could not be triangulated")
    triangles.append(tuple(remaining))
    return triangles


def mesh_data(spec):
    """Return one closed, outward-facing mesh with explicit risers and side walls.

    Adjacent boxes are deliberately avoided: there are no coincident interior
    faces and no z=0 plane overlaid on the staircase or landing.
    """
    polygon = _side_polygon(spec)
    count = len(polygon)
    vertices = [(x, y, z) for y in (0.0, spec.width_m) for x, z in polygon]
    sides = _triangulate(polygon)
    faces = list(sides)  # x/z CCW faces point toward -Y at y=0.
    faces.extend((c + count, b + count, a + count) for a, b, c in sides)
    for index in range(count):
        following = (index + 1) % count
        faces.extend(((index, index + count, following),
                      (following, index + count, following + count)))
    return vertices, faces


def isaac_generator_cfg(spec):
    """Build a single-lane generator; calling this needs the existing Isaac Lab.

    Install this explicitly into a terrain-only preview. Do not register a PPO
    task or tile lanes together: their outer edges intentionally have different
    heights. The origin is the spawn GROUND position before Lab's centering.
    """
    import numpy as np
    import trimesh
    from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg

    def generate(_difficulty, cfg):
        if not all(math.isclose(a, b, abs_tol=1e-9)
                   for a, b in zip(cfg.size, (spec.length_m, spec.width_m))):
            raise ValueError("Generator size differs from the obstacle specification")
        vertices, faces = mesh_data(spec)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return [mesh], np.array(spec.markers()["spawn_ground_m"], dtype=np.float64)

    return TerrainGeneratorCfg(
        size=(spec.length_m, spec.width_m), num_rows=1, num_cols=1,
        border_width=0.0, curriculum=False, use_cache=False,
        sub_terrains={spec.kind: MeshPlaneTerrainCfg(proportion=1.0, function=generate)},
    )


def manifest(spec):
    from dataclasses import asdict

    markers = spec.markers()
    offset = (spec.length_m / 2.0, spec.width_m / 2.0, 0.0)
    return {
        "kind": spec.kind,
        "parameters": asdict(spec),
        "dimensions_m": [spec.length_m, spec.width_m, spec.total_height_m],
        "surface_profile_xz_m": spec.profile(),
        "raw_markers": markers,
        "isaac_single_tile_markers": {
            key: [value[i] - offset[i] for i in range(3)] for key, value in markers.items()
        },
        "travel_direction": "+X",
        "coordinate_convention": "callback: x=[0,length], y=[0,width]; Lab subtracts size/2",
        "robot_root_pose": None,
        "validation_scope": "geometry only; dimensions are test settings, not demonstrated robot capability",
        "training_started": False,
    }
