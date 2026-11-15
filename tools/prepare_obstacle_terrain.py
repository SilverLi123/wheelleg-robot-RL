#!/usr/bin/env python3
"""Export static stair/ledge meshes without importing Isaac Sim or starting PPO."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "project/source/agent_tasks/agent_tasks/direct/wheelbipe/"
    "wheelbipe_own/obstacle_terrain.py"
)


def load_geometry():
    spec = importlib.util.spec_from_file_location("obstacle_terrain_geometry", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tuple_text(values):
    return "(" + ", ".join(f"{value:.9g}" for value in values) + ")"


def export_case(geometry, spec, folder):
    """Export the same triangles in OBJ and static USD, centred as Lab would do."""
    folder.mkdir(parents=True, exist_ok=False)
    vertices, faces = geometry.mesh_data(spec)
    vertices = [(x - spec.length_m / 2.0, y - spec.width_m / 2.0, z)
                for x, y, z in vertices]
    obj = ["# metres; +X travel; +Z up; centred like an Isaac Lab single tile"]
    obj.extend("v " + " ".join(f"{value:.9g}" for value in vertex) for vertex in vertices)
    obj.extend("f " + " ".join(str(index + 1) for index in face) for face in faces)
    (folder / "terrain.obj").write_text("\n".join(obj) + "\n")

    data = geometry.manifest(spec)
    data["exports"] = {
        "terrain.obj": "same centred triangles as the USD collider",
        "terrain.usda": "static collider only; no robot, simulation loop or training",
    }
    data["material"] = "unspecified; assign the existing terrain material at integration"
    (folder / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    usd = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        'def Xform "World" {',
        '    def Mesh "Terrain" (',
        '        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]',
        "    ) {",
        "        bool physics:collisionEnabled = true",
        '        uniform token physics:approximation = "none"',
        '        uniform token subdivisionScheme = "none"',
        '        uniform token orientation = "rightHanded"',
        "        point3f[] points = [" + ", ".join(_tuple_text(v) for v in vertices) + "]",
        "        int[] faceVertexCounts = [" + ", ".join("3" for _ in faces) + "]",
        "        int[] faceVertexIndices = [" + ", ".join(str(i) for f in faces for i in f) + "]",
        "        color3f[] primvars:displayColor = [(0.35, 0.55, 0.65)]",
        "    }",
        "}",
    ]
    (folder / "terrain.usda").write_text("\n".join(usd) + "\n")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory; existing outputs are never overwritten")
    parser.add_argument("--kind", choices=("stairs_down", "platform_up"),
                        help="Omit to export the six initial geometry cases")
    parser.add_argument("--height-m", type=float,
                        help="Per-riser height for stairs, or platform top height, in metres")
    parser.add_argument("--step-count", type=int, default=4)
    parser.add_argument("--tread-m", type=float, default=0.35)
    args = parser.parse_args()
    if (args.kind is None) != (args.height_m is None):
        parser.error("--kind and --height-m must be provided together")
    geometry = load_geometry()
    cases = ([(args.kind, args.height_m)] if args.kind else
             [("stairs_down", h) for h in (0.03, 0.06, 0.09)]
             + [("platform_up", h) for h in (0.06, 0.12, 0.18)])
    try:
        specs = [geometry.ObstacleSpec(kind=kind, height_m=height,
                                      step_count=args.step_count, tread_m=args.tread_m)
                 for kind, height in cases]
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error("--output must be a new directory; choose a new name")
    args.output.mkdir(parents=True)
    index = []
    for index_number, spec in enumerate(specs, 1):
        name = f"{index_number:02d}_{spec.kind}_{spec.height_m * 1000:g}mm"
        data = export_case(geometry, spec, args.output / name)
        index.append({"directory": name, "kind": spec.kind,
                      "height_m": spec.height_m, "dimensions_m": data["dimensions_m"]})
        print(f"{name}: {data['dimensions_m']} m; static geometry only")
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
