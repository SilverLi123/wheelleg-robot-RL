"""Target-only isolated axial fixture for 0/21/42mm and dynamic damping.

This does not write an unclosed pose into the full robot. It reuses the actual
Own spring actuator configuration on a 10kg slider, with no gravity/contact.
"""
import argparse
import copy
import json
import traceback
from pathlib import Path

from preflight import require_target
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
versions = require_target()
app = AppLauncher(args).app

import torch
from pxr import UsdPhysics, Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from agent_world.assets.wheelbipe_own import WheelbipeOwn_CFG


def main():
    if not (args.device or "cuda:0").startswith("cuda"):
        raise ValueError("Fixture acceptance requires GPU physics")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.0005, device=args.device or "cuda:0",
                                                           gravity=(0.0, 0.0, 0.0)))
    stage = sim.stage
    stage.DefinePrim("/Fixture", "Xform")
    for name, mass in (("base", 1.0), ("slider", 10.0)):
        body = stage.DefinePrim("/Fixture/" + name, "Xform")
        UsdPhysics.RigidBodyAPI.Apply(body)
        m = UsdPhysics.MassAPI.Apply(body)
        m.CreateMassAttr(mass)
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
        m.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Fixture/fixed_base")
    fixed.CreateBody1Rel().SetTargets(["/Fixture/base"])
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Fixture/spring2_joint")
    joint.CreateBody0Rel().SetTargets(["/Fixture/base"])
    joint.CreateBody1Rel().SetTargets(["/Fixture/slider"])
    joint.CreateAxisAttr("X")
    # Fixture travel permits both velocity signs at q=0 without an end-stop impulse.
    # These are NOT changes to the robot's mechanical/working limits.
    joint.CreateLowerLimitAttr(-1.0)
    joint.CreateUpperLimitAttr(1.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)
    spring_cfg = copy.deepcopy(WheelbipeOwn_CFG.actuators["spring"])
    spring_cfg.joint_names_expr = ["spring2_joint"]
    # Account for the actuator armature when checking observed acceleration.
    robot = Articulation(ArticulationCfg(prim_path="/Fixture", spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={"spring": spring_cfg}))
    sim.reset()
    robot.update(sim.get_physics_dt())
    k = robot.root_physx_view.get_dof_stiffnesses()
    d = robot.root_physx_view.get_dof_dampings()
    if torch.count_nonzero(k) or torch.count_nonzero(d):
        raise RuntimeError("Residual native spring drive in fixture")
    rows = []
    for q in (0.0, 0.021, 0.042):
        for dq in (0.0, -0.1, 0.1):
            position = torch.tensor([[q]], device=robot.device)
            velocity = torch.tensor([[dq]], device=robot.device)
            robot.reset()
            robot.write_joint_state_to_sim(position, velocity)
            robot.set_joint_position_target(position)
            robot.set_joint_velocity_target(torch.zeros_like(velocity))
            static = 500.0 - (150.0 / 0.042) * q
            robot.set_joint_effort_target(torch.full_like(position, static))
            robot.write_data_to_sim()
            expected = static - 50.0 * dq
            applied = float(robot.data.applied_torque[0, 0])
            sim.step(render=False)
            robot.update(sim.get_physics_dt())
            observed_accel = (float(robot.data.joint_vel[0, 0]) - dq) / sim.get_physics_dt()
            effective_mass = 10.0 + float(robot.actuators["spring"].armature[0, 0])
            observed_force = observed_accel * effective_mass
            rows.append({"q_m": q, "dq_m_s": dq, "F_static_N": static,
                "expected_net_N": expected, "applied_effort_N": applied,
                "force_from_acceleration_N": observed_force})
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"environment": versions, "kind": "isolated GPU axial fixture, not full linkage motion",
        "native_K": k.tolist(), "native_D": d.tolist(), "slider_mass_kg": 10.0,
        "dt": sim.get_physics_dt(), "rows": rows,
        "max_applied_error_N": max(abs(r["applied_effort_N"] - r["expected_net_N"]) for r in rows),
        "max_observed_error_N": max(abs(r["force_from_acceleration_N"] - r["expected_net_N"]) for r in rows)}
    report["status"] = "PASS" if report["max_applied_error_N"] <= 0.1 and report["max_observed_error_N"] <= 10.0 else "FAIL"
    (args.output / "spring_fixture.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["max_applied_error_N"] > 0.1 or report["max_observed_error_N"] > 10.0:
        raise RuntimeError("Spring fixture force/damping check failed")


if __name__ == "__main__":
    try:
        with torch.inference_mode():
            main()
    except BaseException:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        # No Replicator writer is used by this physics-only fixture.
        app.close(wait_for_replicator=False)
