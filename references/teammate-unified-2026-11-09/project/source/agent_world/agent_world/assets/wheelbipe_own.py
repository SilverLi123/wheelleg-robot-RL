"""Own robot, explicit native drives; motor limits remain provisional simulation values."""
import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

ASSET_DIR = Path(__file__).parent / "usd_files/wheelbipe_own"
MECHANISM = json.loads((ASSET_DIR / "mechanism.json").read_text())
REACHABILITY = json.loads((ASSET_DIR / "reachability.json").read_text())
JOINT_NAMES = tuple(MECHANISM["joints"])
BODY_NAMES = tuple(MECHANISM["bodies"])
LEG_NAMES = tuple(s + "_" + part + "_joint" for s in ("left", "right") for part in ("front1", "rear1"))
WHEEL_NAMES = ("left_wheel_joint", "right_wheel_joint")
SPRING_NAMES = ("left_spring2_joint", "right_spring2_joint")
GUIDE_NAMES = tuple(n for n in JOINT_NAMES if "guide" in n)
PASSIVE_NAMES = tuple(n for n in JOINT_NAMES if n not in (*LEG_NAMES, *WHEEL_NAMES, *SPRING_NAMES, "gimbal_yaw_joint"))
GUIDE_BODIES = tuple(n for n in BODY_NAMES if "guide" in n)
UNDESIRED_BODIES = tuple(n for n in BODY_NAMES if n not in ("left_wheel_link", "right_wheel_link"))
RESET_BODIES = ("base_link", "gimbal_yaw_link", *GUIDE_BODIES)
GROUPS = {"legs_act": LEG_NAMES, "legs_inact": PASSIVE_NAMES, "wheel": WHEEL_NAMES,
          "spring": SPRING_NAMES, "gimbal_yaw": ("gimbal_yaw_joint",)}
assert len(GUIDE_NAMES) == len(GUIDE_BODIES) == 14
assert len(PASSIVE_NAMES) == 24
assert len([n for names in GROUPS.values() for n in names]) == len(set(JOINT_NAMES)) == 33
assert set(n for names in GROUPS.values() for n in names) == set(JOINT_NAMES)

WheelbipeOwn_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "robot.usdc"), activate_contact_sensors=True, copy_from_source=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0,
            angular_damping=0.0, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False,
            enabled_self_collisions=False, solver_position_iteration_count=128,
            solver_velocity_iteration_count=8)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.265),
        joint_pos=MECHANISM["nominal_native_q"], joint_vel={".*": 0.0}),
    actuators={
        # These gains/limits are reference simulation starting points, NOT hardware specifications.
        "legs_act": IdealPDActuatorCfg(joint_names_expr=list(LEG_NAMES), stiffness=60.0, damping=2.0,
            effort_limit=40.0, velocity_limit=17.0, armature=1.95e-4 * 81),
        "legs_inact": IdealPDActuatorCfg(joint_names_expr=list(PASSIVE_NAMES), stiffness=0.0, damping=0.01,
            effort_limit=50.0, velocity_limit=300.0, armature=0.0001),
        "wheel": IdealPDActuatorCfg(joint_names_expr=list(WHEEL_NAMES), stiffness=0.0, damping=0.2,
            effort_limit=5.0, velocity_limit=60.0, armature=0.0),
        "spring": IdealPDActuatorCfg(joint_names_expr=list(SPRING_NAMES), stiffness=0.0, damping=50.0,
            effort_limit=1000.0, velocity_limit=50.0, armature=0.0001),
        # Real yaw is passive with modest damping. There is no pitch joint or artificial lock.
        "gimbal_yaw": IdealPDActuatorCfg(joint_names_expr=["gimbal_yaw_joint"], stiffness=0.0,
            damping=0.5, effort_limit=2.0, velocity_limit=30.0, armature=0.0001),
    },
)
