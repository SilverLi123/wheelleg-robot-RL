"""Conservative deterministic bring-up; V14 Flat-v0 rewards are inherited intact."""
import copy

from isaaclab.utils import configclass
from agent_world.assets.wheelbipe_own import (
    WheelbipeOwn_CFG, LEG_NAMES, PASSIVE_NAMES, WHEEL_NAMES, SPRING_NAMES, REACHABILITY,
)
from agent_tasks.direct.wheelbipe.wheelbipe_V14.env_cfg import WheelbipeV14FlatEnvCfg


@configclass
class WheelbipeOwnFlatEnvCfg(WheelbipeV14FlatEnvCfg):
    robot_cfg = WheelbipeOwn_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()
    legs_act_name = list(LEG_NAMES)
    legs_inact_name = list(PASSIVE_NAMES)
    wheel_name = list(WHEEL_NAMES)
    spring_name = list(SPRING_NAMES)
    use_gimbal = False  # V14 active gimbal controller disabled; native yaw remains passive.
    gimbal_yaw_name = ["gimbal_yaw_joint"]
    gimbal_pitch_name = []
    events = None
    curriculum = None
    use_obs_delay = False
    use_act_delay = False
    use_self_obs_noise = False
    use_self_act_noise = False
    self_obs_noise_cfg = None
    self_act_noise_cfg = None
    use_predefined_leg_random_start = False
    own_random_reset = False
    spring_settings = dict(mode="own_explicit", constant_force=580.0, random_force=[0.0, 0.0], damping=False)
    # Sim5.1 GPU: 200Hz/128/8 passed 60s continuous closure; 12/6 failed.
    # Keep policy at 50Hz (decimation4) and the validated 128/8 asset solver.
    own_physics_hz = 200
    default_height_cmd = 0.255
    height_range = [0.245, 0.265]
    height_measure_wheel_radius = 0.0625
    wheel_radius = 0.0625
    leg_action_scale = 0.15
    wheel_vel_action_scale = 10.0
    max_wheel_vel = 60.0
    max_wheel_torque = 5.0
    lower_joint_limit = -3.14
    upper_joint_limit = 3.14
    base_height_bound = 0.16
    # Front and rear pivots have different lateral locations. The alignment
    # implementation below uses the front pivot of each side explicitly.
    wheel_motor_z_axis_align_ref_y_offset = 0.2012
    leg_length_range = [0.15, 0.26]
    leg_angle_range = [-0.20, 0.32]
    # Reference analytical IK must never be used for this mechanism.
    links_length = None
    alpha_offset = None
    play_wheel_material_debug = False
    play_ang_vel_z_debug_vis = False
    play_terrain_debug_vis = False
    own_spring_tolerance_m = 0.001
    own_strict_finite = True
    rebound_vel_z_threshold = 0.08

    def __post_init__(self):
        super().__post_init__()
        self.rewards = copy.deepcopy(self.rewards)
        self.rewards["upward_rebound_vel_z"] = 0.0
        self.rewards["roll_angle_l2"] = 0.0
        self.events = None
        self.curriculum = None
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        # Copying each environment is deliberate until closed-loop replication is validated.
        self.scene.replicate_physics = False
        if self.own_physics_hz % 50:
            raise ValueError("own_physics_hz must preserve the 50 Hz policy")
        from isaaclab_physx.physics import PhysxCfg
        # Lab 3 changed this default to True. Match the actual 5.1 scene's
        # False setting so TGS integrates the original controller's forces alike.
        self.sim.physics = PhysxCfg(enable_external_forces_every_iteration=False)
        self.sim.use_newton_actuators = False  # preserve original explicit IdealPD execution
        # Lab 2.3 set every explicit actuator's native effort cap to 1e9.
        # Lab 3 otherwise inherits a mix of USD-authored caps and float-max.
        # IdealPD still enforces the unchanged per-actuator 40/50/5/1000/2 caps.
        for actuator in self.robot_cfg.actuators.values():
            actuator.joint_effort_limit = 1.0e9
        self.sim.dt = 1.0 / self.own_physics_hz
        self.decimation = self.own_physics_hz // 50
        self.sim.render_interval = self.decimation
        self.predefined_reset_ground = {"enabled": False}
        self.commands = copy.deepcopy(self.commands)
        self.commands.special_modes = {}
        self.commands.heading_command = False
        self.commands.rel_heading_envs = 0.0
        self.commands.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (-0.5, 0.5)
        self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)
        self.airborne_state_machine_cfg["enter"]["wheel_radius"] = 0.0625
        self.height_command_special_modes_cfg = {"enabled": False}
        self.observation_space = self.num_single_obs = 35
        self.state_space = self.num_single_privileged_obs = 78
        assert len(self.robot_cfg.actuators) == 5 and "gimbal_pitch" not in self.robot_cfg.actuators
        # Lab 3 replaces clone_environments() with scene-owned clone planning.
        # Declare the same assets and sensors; keep native physics replication off
        # so every closed-loop USD mechanism is copied in full.
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        self.scene.robot = self.robot_cfg.copy()
        self.scene.terrain = self.terrain.copy()
        self.scene.contact_sensor = self.contact_sensor.copy()
        if not self.use_leg_length_as_height and not self.use_absolute_height:
            self.scene.height_scanner = self.height_scanner.copy()
        for name in ("right_wheel_height_scanner", "left_wheel_height_scanner"):
            sensor = getattr(self, name, None)
            if sensor is not None:
                setattr(self.scene, name, sensor.copy())
        if getattr(self, "enable_scan_dot", False) and self.dot_scanner is not None:
            self.scene.dot_scanner = self.dot_scanner.copy()
        self.scene.light = AssetBaseCfg(prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))


@configclass
class WheelbipeOwnFlatPlayEnvCfg(WheelbipeOwnFlatEnvCfg):
    play = True
    play_keep_done_reset = True  # Replay uses the same fall/contact termination as training.
