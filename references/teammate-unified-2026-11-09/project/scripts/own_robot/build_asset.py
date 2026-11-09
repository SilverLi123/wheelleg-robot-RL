"""Extract a portable robot asset; no Kit/PhysX runtime is imported.

Run in a separate USD tooling environment with usd-core==25.5.1 and numpy/scipy.
The source is never modified. Unknown PhysX schema attributes are retained by Sdf.
"""
import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "source/agent_world/agent_world/assets/usd_files/wheelbipe_own"
EXPECTED = "5b4b42207401c8f207178bfb3e7f9996f5f4dc5c81f38c819dff2246b2d25176"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame(pos, quat):
    m = Gf.Matrix4d(1)
    m.SetRotate(Gf.Quatd(quat))
    m.SetTranslateOnly(Gf.Vec3d(pos))
    # Gf uses row vectors; offline numpy FK uses column vectors.
    return [[float(m[j][i]) for j in range(4)] for i in range(4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    args = ap.parse_args()
    if sha(args.source) != EXPECTED:
        raise ValueError("Source changed since mechanical audit; re-audit before extracting")
    DEST.mkdir(parents=True, exist_ok=True)
    src = Usd.Stage.Open(str(args.source))
    flat = src.Flatten()
    # Sdf namespace edits move specs; explicitly remap all target/connection paths.
    edits = Sdf.BatchNamespaceEdit()
    edits.Add("/World/robot", "/Robot")
    if not flat.Apply(edits):
        raise RuntimeError("USD namespace edit failed")
    st = Usd.Stage.Open(flat)
    for p in list(st.GetPseudoRoot().GetChildren()):
        if p.GetPath() != Sdf.Path("/Robot"):
            st.RemovePrim(p.GetPath())
    st.SetDefaultPrim(st.GetPrimAtPath("/Robot"))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    # base_link is the floating root, not the source scene's world-space pose.
    for p in [st.GetPrimAtPath("/Robot")]:
        UsdGeom.Xformable(p).ClearXformOpOrder()
    if not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in st.Traverse()):
        raise RuntimeError("Missing articulation root")
    model = {"source_sha256": EXPECTED, "coordinate_equation": "q_policy=s*q_native+b",
             "zero_definition": "USD joint frames aligned, b=0; not saved pose zero",
             "bodies": {}, "joints": {}, "loops": []}
    for p in st.Traverse():
        # Every path-valued relationship/connection must stay within the asset.
        for rel in p.GetRelationships():
            rel.SetTargets([t.ReplacePrefix(Sdf.Path("/World/robot"), Sdf.Path("/Robot"))
                            for t in rel.GetTargets()])
            for target in rel.GetTargets():
                if not target.HasPrefix(Sdf.Path("/Robot")):
                    raise RuntimeError(f"External relationship: {rel.GetPath()} -> {target}")
        for attr in p.GetAttributes():
            if attr.GetConnections():
                attr.SetConnections([t.ReplacePrefix(Sdf.Path("/World/robot"), Sdf.Path("/Robot"))
                                     for t in attr.GetConnections()])
            for target in attr.GetConnections():
                if not target.HasPrefix(Sdf.Path("/Robot")):
                    raise RuntimeError(f"External connection: {attr.GetPath()} -> {target}")
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            model["bodies"][p.GetName()] = {"mass": UsdPhysics.MassAPI(p).GetMassAttr().Get(),
                "path": str(p.GetPath())}
        if not p.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(p)
        row = {"name": p.GetName(), "path": str(p.GetPath()), "type": p.GetTypeName(),
               "parent": j.GetBody0Rel().GetTargets()[0].name,
               "child": j.GetBody1Rel().GetTargets()[0].name,
               "frame0": frame(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
               "frame1": frame(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())}
        if p.IsA(UsdPhysics.SphericalJoint):
            assert j.GetJointEnabledAttr().Get() and j.GetExcludeFromArticulationAttr().Get()
            model["loops"].append(row)
            continue
        linear = p.IsA(UsdPhysics.PrismaticJoint)
        row.update(axis=p.GetAttribute("physics:axis").Get(), sign=1 if linear else -1, offset=0.0,
                   unit="m" if linear else "rad")
        for key in ("lower", "upper"):
            value = p.GetAttribute(f"physics:{key}Limit").Get()
            row[key] = float(value if linear else math.radians(value)) if math.isfinite(value) else None
        state = "linear" if linear else "angular"
        q = p.GetAttribute(f"state:{state}:physics:position").Get() or 0.0
        row["saved_q"] = float(q if linear else math.radians(q))
        # Explicit IdealPD owns all drives. Clear native K/D and stale targets.
        for a in p.GetAttributes():
            if a.GetName().startswith("drive:") and a.GetName().split(":")[-1] in (
                    "stiffness", "damping", "targetPosition", "targetVelocity"):
                a.Set(0.0)
        model["joints"][p.GetName()] = row
    assert len(model["bodies"]) == 34 and len(model["joints"]) == 33 and len(model["loops"]) == 6
    assert all(model["joints"][f"{s}_spring1_joint"]["parent"] == f"{s}_rear1_link" for s in ("left", "right"))
    # Bake a closed nominal configuration from the actual joint frames. This
    # removes residual error in the source's saved simulation snapshot without
    # changing any anchor, mass, inertia, geometry or limit.
    (DEST / "mechanism.json").write_text(json.dumps(model, indent=2) + "\n")
    from kinematics import Mechanism
    mechanism = Mechanism()
    nominal = dict(mechanism.saved)
    for side in ("left", "right"):
        solved, _ = mechanism.solve(side, nominal[side + "_front1_joint"], nominal[side + "_rear1_joint"])
        nominal.update({n: v for n, v in solved.items() if n.startswith(side + "_")})
    for name, pose in mechanism.fk(nominal).items():
        p = st.GetPrimAtPath(model["bodies"][name]["path"])
        xf = UsdGeom.Xformable(p)
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix="trainingNominal").Set(Gf.Matrix4d(*pose.T.ravel().tolist()))
    for name, j in model["joints"].items():
        kind = "linear" if j["unit"] == "m" else "angular"
        p = st.GetPrimAtPath(j["path"])
        p.GetAttribute(f"state:{kind}:physics:position").Set(
            nominal[name] if kind == "linear" else math.degrees(nominal[name]))
        p.GetAttribute(f"state:{kind}:physics:velocity").Set(0.0)
    model["nominal_native_q"] = nominal
    st.GetRootLayer().customLayerData = {
        "purpose": "Robot-only training asset; explicit Isaac Lab IdealPD actuation required",
        "sourceSHA256": EXPECTED,
        "sourceMetadata": dict(st.GetRootLayer().customLayerData),
        "trainingGasSpring": {"runtimePythonRequired": True,
            "nativeStiffness": 0.0, "nativeDamping": 0.0,
            "explicitActuatorDamping": 50.0, "effortLimit_N": 1000.0,
            "forceLaw": "500 - (150/0.042)*clamp(q,0,0.042); damping applied once by IdealPD",
            "workingExtension_m": 0.042},
    }
    out = DEST / "robot.usdc"
    st.GetRootLayer().defaultPrim = "Robot"
    st.GetRootLayer().Export(str(out))
    # Check relocation without relying on desktop-side resources or search paths.
    with tempfile.TemporaryDirectory() as td:
        relocated = Path(td) / out.name
        relocated.write_bytes(out.read_bytes())
        probe = Usd.Stage.Open(str(relocated))
        assert probe and str(probe.GetDefaultPrim().GetPath()) == "/Robot"
        layers, assets, missing = UsdUtils.ComputeAllDependencies(str(relocated))
        # OmniPBR is a simulator built-in, resolved later by Kit's MDL resolver.
        external_missing = [str(x) for x in missing if str(x) != "OmniPBR.mdl"]
        if external_missing:
            raise RuntimeError(f"Missing portable dependencies: {external_missing}")
    model["asset_sha256"] = sha(out)
    (DEST / "mechanism.json").write_text(json.dumps(model, indent=2) + "\n")
    (DEST / "asset_manifest.json").write_text(json.dumps({
        "source_sha256": EXPECTED, "robot_sha256": sha(out), "rigid_bodies": 34,
        "ordinary_joints": 33, "spherical_loops": 6,
        "mass_kg": sum(b["mass"] for b in model["bodies"].values()),
        "geometry": "unchanged; no collision simplification", "native_drives": "all K/D/targets zero",
        "builtin_dependencies": ["OmniPBR.mdl"], "relocated_stage_open": True,
        "dynamic_validation": "PENDING Sim5.1/Lab2.3 GPU"}, indent=2) + "\n")
    print(out, sha(out))


if __name__ == "__main__":
    main()
