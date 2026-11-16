"""Isolated, explicitly named ablations for low-speed policy improvement."""
from pathlib import Path
import copy
import json
import os
import torch
from isaaclab.utils import configclass
from .env_cfg import WheelbipeOwnFlatEnvCfg
from .env import WheelbipeOwnEnv
from .unified_environment import UnifiedLocomotionMixin, configure_unified
from .domain_randomization import DomainRandomizationMixin, configure_domain_randomization
from agent_world.assets.wheelbipe_own import MECHANISM
from agent_world.sim620_compat import torch_data as T
from .runner_cfg import WheelbipeOwnPPORunnerCfg
from . import mixed_slopes, random_terrain_map
from .terrain_curriculum import terrain_curriculum_decisions

PROFILE_FILE = Path(__file__).resolve().parents[7] / 'profiles.json'

def selected_profile():
    profiles = json.loads(PROFILE_FILE.read_text())
    name = os.environ.get('WHEELBIPE_EXPERIMENT', '')
    if name not in profiles:
        raise ValueError(f'WHEELBIPE_EXPERIMENT must select one of {list(profiles)}; got {name!r}')
    return name, copy.deepcopy(profiles[name])

@configclass
class WheelbipeOwnExperimentCfg(WheelbipeOwnFlatEnvCfg):
    experiment_profile_name: str = ''
    experiment_profile_settings: dict = {}
    experiment_leg_filter: bool = False
    experiment_cylinder_wheels: bool = False

    def __post_init__(self):
        super().__post_init__()
        name, p = selected_profile()
        self.experiment_profile_name = name
        self.experiment_profile_settings = p
        self.sim.physics.enable_external_forces_every_iteration = p['external_forces_every_iteration']
        # All ablations retain the exact same mechanism, explicit spring and IdealPD gains.
        self.rewards = copy.deepcopy(self.rewards)
        self.rewards.update(p.get('reward_weights', {}))
        for key, value in p.get('reward_parameters', {}).items():
            setattr(self, key, value)
        if 'rebound_vel_z_threshold' in p:
            self.rebound_vel_z_threshold = float(p['rebound_vel_z_threshold'])
        # POLICY_REBUILD_20261107: explicit limits at actuator and solver levels.
        if 'leg_target_rate' in p:
            # Mixed-slope ablations may narrow the command distribution while
            # preserving the existing 35/78 observation and 6D action contract.
            self.commands.ranges.lin_vel_x = tuple(
                p.get('command_lin_vel_x', (-p['command_speed'], p['command_speed']))
            )
            self.commands.ranges.ang_vel_z = tuple(
                p.get('command_ang_vel_z', (-p['command_yaw'], p['command_yaw']))
            )
            self.commands.resampling_time_range = tuple(p['command_resampling'])
            self.commands.rel_standing_envs = float(
                p.get('command_standing_fraction', 0.2)
            )
            self.own_random_reset = True
            for key, limit, speed in [('legs_act', p['leg_effort_limit'], 17.0),
                                       ('wheel', p['wheel_effort_limit'], 60.0)]:
                actuator = self.robot_cfg.actuators[key]
                actuator.effort_limit = limit
                actuator.joint_effort_limit = limit
                actuator.joint_velocity_limit = speed
            self.scene.robot = self.robot_cfg.copy()
            if p.get('terrain_mix'):
                from isaaclab.terrains import (TerrainGeneratorCfg, MeshPlaneTerrainCfg,
                    HfPyramidSlopedTerrainCfg, HfInvertedPyramidSlopedTerrainCfg,
                    HfWaveTerrainCfg, HfRandomUniformTerrainCfg, HfPyramidStairsTerrainCfg)
                self.terrain = self.terrain.copy()
                self.terrain.terrain_type = 'generator'
                self.terrain.max_init_terrain_level = 0
                family = p.get('terrain_family', 'mixed')
                if family == 'rough':
                    pieces = {
                        'flat': MeshPlaneTerrainCfg(proportion=.20),
                        'wave': HfWaveTerrainCfg(proportion=.35, amplitude_range=(.002,.025), num_waves=6),
                        'rough': HfRandomUniformTerrainCfg(proportion=.35, noise_range=(-.006,.006), noise_step=.002, downsampled_scale=.2),
                        'slope': HfPyramidSlopedTerrainCfg(proportion=.10, slope_range=(.02,.12), platform_width=2.),
                    }
                else:
                    from isaaclab.terrains import HfInvertedPyramidStairsTerrainCfg
                    pieces = {
                        'flat': MeshPlaneTerrainCfg(proportion=.30),
                        'slope_up': HfPyramidSlopedTerrainCfg(proportion=.15, slope_range=(.02,.15), platform_width=2.),
                        'slope_down': HfInvertedPyramidSlopedTerrainCfg(proportion=.15, slope_range=(.02,.15), platform_width=2.),
                        'wave': HfWaveTerrainCfg(proportion=.15, amplitude_range=(.002,.020), num_waves=5),
                        'rough': HfRandomUniformTerrainCfg(proportion=.10, noise_range=(-.006,.006), noise_step=.002, downsampled_scale=.2),
                        'steps_up': HfPyramidStairsTerrainCfg(proportion=.075, step_height_range=(.005,.030), step_width=.30, platform_width=2.),
                        'steps_down': HfInvertedPyramidStairsTerrainCfg(proportion=.075, step_height_range=(.005,.030), step_width=.30, platform_width=2.),
                    }
                self.terrain.terrain_generator = TerrainGeneratorCfg(
                    seed=42, size=(8.,8.), border_width=10., num_rows=6, num_cols=10,
                    curriculum=True, horizontal_scale=.1, vertical_scale=.001,
                    use_cache=False, sub_terrains=pieces)
                self.scene.terrain = self.terrain.copy()
        self.experiment_cylinder_wheels = bool(p.get('cylinder_wheels', False))
        self.experiment_leg_filter = bool(p.get('leg_filter', False))
        self.use_action_low_pass_filter = self.experiment_leg_filter
        self.action_low_pass_prev_weight = float(p.get('filter_prev', 0.6))
        self.action_low_pass_curr_weight = 1.0 - self.action_low_pass_prev_weight
        configure_domain_randomization(self, p['domain_randomization'])
        configure_unified(self, p)
        mixed_slopes.configure(self)
        random_terrain_map.configure(self)

@configclass
class WheelbipeOwnExperimentPlayCfg(WheelbipeOwnExperimentCfg):
    play = True
    play_keep_done_reset = True

class WheelbipeOwnExperimentEnv(UnifiedLocomotionMixin, DomainRandomizationMixin, WheelbipeOwnEnv):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._legacy_terrain_episode_start = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._legacy_terrain_episode_heading = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._legacy_terrain_pending_move_up = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._legacy_terrain_pending_move_down = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._capture_legacy_terrain_start(self.robot._ALL_INDICES)
        p = cfg.experiment_profile_settings
        if 'reset_band' in p:
            for side in ('left', 'right'):
                names = [side + '_front1_joint', side + '_rear1_joint']
                indices = [self._bank_names.index(n) for n in names]
                nominal = torch.tensor([MECHANISM['nominal_native_q'][n] for n in names], device=self.device)
                keep = (self._reset_bank[side][:, indices] - nominal).abs().amax(1) <= p['reset_band']
                if not keep.any():
                    raise RuntimeError('No closure-preserving reset samples within configured band')
                self._reset_bank[side] = self._reset_bank[side][keep]
                self._bank_body_z[side] = self._bank_body_z[side][keep]
        if 'spring_damping_range' in p:
            low, high = p['spring_damping_range']
            self._native_robot.actuators['spring'].damping.uniform_(low, high)
        self._init_training_audit()
        self._init_domain_randomization()
        self._slew_leg = self.robot.data.joint_pos[:, self._legs_act_idx].clone()
        self._slew_wheel = self.robot.data.joint_vel[:, self._wheel_idx].clone()
        self._init_unified()
        random_terrain_map.initialize(self)
        self._curriculum_last_iter = -1
        self._curriculum_state = None
        self._update_experiment_curriculum(force=True)
        settings = self.cfg.experiment_profile_settings
        self._cmd_lat_accel_limit = float(
            settings.get('command_centripetal_accel_limit', 0.0)
        )
        self._cmd_high_speed_frac = float(
            settings.get('command_high_speed_fraction', 0.0)
        )
        self._cmd_high_speed_range = tuple(
            settings.get('command_high_speed_range', (1.5, 2.0))
        )
        self._cmd_high_speed_smooth_only = bool(
            settings.get('command_high_speed_smooth_only', False)
        )
        self._orig_resample_command = None
        if (
            self._cmd_lat_accel_limit > 0.0 or self._cmd_high_speed_frac > 0.0
        ) and hasattr(self, 'command_generator'):
            cmd_gen = self.command_generator
            orig_resample = cmd_gen._resample_command
            self._orig_resample_command = orig_resample

            def _capped_resample(env_ids):
                orig_resample(env_ids)
                self._apply_command_postprocess(env_ids)

            cmd_gen._resample_command = _capped_resample
            self._apply_command_postprocess()

    def _apply_command_postprocess(self, env_ids=None, allow_boost: bool = True) -> None:
        if not hasattr(self, 'command_generator'):
            return
        cmd_gen = self.command_generator
        if allow_boost and getattr(self, '_cmd_high_speed_frac', 0.0) > 0.0:
            eligible = None
            if getattr(self, '_cmd_high_speed_smooth_only', False) and hasattr(
                self, '_random_terrain_hazard_mask'
            ):
                eligible = ~self._random_terrain_hazard_mask
            random_terrain_map.boost_command_high_speed(
                cmd_gen.command,
                self._cmd_high_speed_frac,
                self._cmd_high_speed_range,
                env_ids,
                getattr(cmd_gen, 'is_standing_env', None),
                eligible=eligible,
            )
        if getattr(self, '_cmd_lat_accel_limit', 0.0) > 0.0:
            random_terrain_map.clamp_command_centripetal_accel(
                cmd_gen.command, self._cmd_lat_accel_limit, env_ids
            )

    def _legacy_terrain_enabled(self) -> bool:
        settings = self.cfg.experiment_profile_settings
        return bool(
            settings.get('terrain_mix') or settings.get('mixed_slopes')
        )

    def _capture_legacy_terrain_start(self, env_ids) -> None:
        """Record the heading and position against which progress is measured."""
        if not hasattr(self, '_legacy_terrain_episode_start'):
            return
        from isaaclab.utils.math import euler_xyz_from_quat

        ids = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        root_pos = T(self.robot.data.root_pos_w)[ids]
        _, _, yaw = euler_xyz_from_quat(T(self.robot.data.root_quat_w)[ids])
        self._legacy_terrain_episode_start[ids] = root_pos[:, :2]
        self._legacy_terrain_episode_heading[ids] = torch.stack(
            (yaw.cos(), yaw.sin()), dim=-1
        )

    def _update_experiment_curriculum(self, force: bool = False) -> None:
        """Advance profile-driven command and steep-block curriculum smoothly with iteration."""
        curriculum_cfg = self.cfg.experiment_profile_settings.get('curriculum')
        if not isinstance(curriculum_cfg, dict):
            return
        if getattr(self.cfg, 'play', False) or getattr(self, '_u_eval', False):
            return
        if not hasattr(self, 'command_generator') or not hasattr(self, '_random_terrain_steep_block_mask'):
            return
        iteration = self._get_training_iteration()
        if not force and iteration == getattr(self, '_curriculum_last_iter', -1):
            return
        state = random_terrain_map.interpolate_stage_curriculum(curriculum_cfg, iteration)
        self.command_generator.cfg.ranges.lin_vel_x = state['lin_vel_x']
        self.command_generator.cfg.ranges.ang_vel_z = state['ang_vel_z']
        self.command_generator.cfg.rel_standing_envs = state['standing_fraction']
        random_terrain_map.set_steep_block_probability(self, state['steep_block_prob'])
        self._curriculum_last_iter = iteration
        self._curriculum_state = state

    def _reset_idx(self, env_ids):
        # DirectRLEnv clears episode_length_buf before _custom_reset_random.
        # Capture age and forward displacement first so the legacy curriculum
        # cannot promote an environment merely because every reset age is zero.
        self._update_experiment_curriculum()
        if self._legacy_terrain_enabled() and hasattr(
            self, '_legacy_terrain_episode_start'
        ):
            ids = (
                self.robot._ALL_INDICES
                if env_ids is None
                else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            )
            age_s = self.episode_length_buf[ids] * self.step_dt
            displacement = (
                T(self.robot.data.root_pos_w)[ids, :2]
                - self._legacy_terrain_episode_start[ids]
            )
            forward_m = (
                displacement * self._legacy_terrain_episode_heading[ids]
            ).sum(dim=-1)
            move_up, move_down = terrain_curriculum_decisions(
                age_s,
                forward_m,
                self.reset_terminated[ids].clone(),
            )
            self._legacy_terrain_pending_move_up[ids] = move_up
            self._legacy_terrain_pending_move_down[ids] = move_down
        result = super()._reset_idx(env_ids)
        if (
            getattr(self, '_cmd_high_speed_smooth_only', False)
            and getattr(self, '_orig_resample_command', None) is not None
            and hasattr(self, '_random_terrain_hazard_mask')
        ):
            ids = (
                self.robot._ALL_INDICES
                if env_ids is None
                else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            )
            if ids.numel() > 0:
                hazard_ids = ids[self._random_terrain_hazard_mask[ids]]
                if hazard_ids.numel() > 0:
                    self._orig_resample_command(hazard_ids)
                    self._apply_command_postprocess(hazard_ids, allow_boost=False)
        return result

    def _init_training_audit(self):
        from pxr import Usd, UsdPhysics
        from isaaclab.utils.math import quat_apply
        self._audit_quat_apply = quat_apply
        layouts = []
        rel_joint_paths = []
        prefix0 = '/World/envs/env_0/Robot'
        for prim in Usd.PrimRange(self.sim.stage.GetPrimAtPath(prefix0)):
            if prim.IsA(UsdPhysics.SphericalJoint):
                rel_joint_paths.append(str(prim.GetPath())[len(prefix0):])
        assert len(rel_joint_paths) == 6
        for eid in range(self.num_envs):
            prefix = f'/World/envs/env_{eid}/Robot'
            rows = []
            for rel_path in rel_joint_paths:
                prim = self.sim.stage.GetPrimAtPath(prefix + rel_path)
                assert prim.IsValid() and prim.IsA(UsdPhysics.SphericalJoint)
                j = UsdPhysics.Joint(prim)
                assert j.GetJointEnabledAttr().Get() and j.GetExcludeFromArticulationAttr().Get()
                targets = [j.GetBody0Rel().GetTargets()[0], j.GetBody1Rel().GetTargets()[0]]
                assert all(str(t).startswith(prefix + '/') for t in targets), 'Cross-environment loop target'
                rows.append(([self._native_robot.body_names.index(t.name) for t in targets],
                             [list(j.GetLocalPos0Attr().Get()), list(j.GetLocalPos1Attr().Get())]))
            assert len(rows) == 6
            layouts.append(rows)
        assert all(x == layouts[0] for x in layouts)
        self._audit_ids = torch.tensor([r[0] for r in layouts[0]],device=self.device)
        self._audit_anchors = torch.tensor([r[1] for r in layouts[0]],device=self.device)
        self._audit_steps = 0
        self._audit_peak_m = 0.
        self._audit_bad_streak = 0
        self._audit_path = Path(os.environ.get('WHEELBIPE_DIAGNOSTICS_DIR','diagnostics')) / 'training_health.json'
        masses = self._native_robot.root_physx_view.get_masses()
        if hasattr(masses,'torch'): masses = masses.torch()
        if not isinstance(masses,torch.Tensor):
            import warp as wp
            masses = wp.to_torch(masses)
        actual = masses.sum(-1)
        assert torch.all((actual - 20.661891813760686).abs() < .0001), actual
        self._audit_mass_kg = float(actual[0])

    def _audit_training(self):
        bp = self._native_robot.data.body_pos_w[:,self._audit_ids]
        bq = self._native_robot.data.body_quat_w[:,self._audit_ids]
        anchors = self._audit_anchors.expand(self.num_envs,-1,-1,-1)
        points = bp + self._audit_quat_apply(bq,anchors)
        error = torch.linalg.vector_norm(points[:,:,0]-points[:,:,1],dim=-1)
        peak = float(error.max())
        if not torch.isfinite(error).all(): raise FloatingPointError('Nonfinite closed-loop anchors')
        self._audit_peak_m = max(self._audit_peak_m,peak)
        self._audit_bad_streak = self._audit_bad_streak + 1 if peak > .005 else 0
        if peak > .005:
            from .physics_failure_diagnostics import record_loop_spike
            record_loop_spike(self, error)
        if not getattr(self, '_u_ready', False):
            if self.cfg.experiment_profile_settings.get('per_env_closure_reset', False):
                per_env_err = error.amax(-1)
                bad_envs = int((per_env_err > .02).sum())
                if peak > .12 or bad_envs >= max(4, int(self.num_envs * .05)):
                    raise RuntimeError(
                        f'Closed-loop error exceeds per-env training guard: peak={peak}m, bad_envs={bad_envs}'
                    )
            elif peak > .02 or self._audit_bad_streak > 5:
                raise RuntimeError(f'Closed-loop error exceeds training guard: {peak}m')
        self._audit_steps += 1
        if self._audit_steps % 100 == 0:
            import time
            row = dict(profile=self.cfg.experiment_profile_name,policy_steps=self._audit_steps,
                       environments=self.num_envs,nominal_mass_kg=self._audit_mass_kg,all_loop_targets_local=True,
                       max_loop_error_m=self._audit_peak_m,latest_loop_error_m=peak,
                       spring_force_range_N=[400.,580.],spring_damping_calibrated=False,
                       spring_damping_actual_range_N_s_m=[float(self._native_robot.actuators['spring'].damping.min()),float(self._native_robot.actuators['spring'].damping.max())],
                       updated_unix=time.time())
            if getattr(self, '_dr_ready', False): row['domain_randomization'] = self._domain_randomization_health()
            if getattr(self, '_u_ready', False): row['unified'] = self._unified_health()
            if hasattr(self, '_random_terrain_layout'):
                row['random_terrain'] = random_terrain_map.health(self)
            if getattr(self, '_curriculum_state', None) is not None:
                row['curriculum'] = dict(self._curriculum_state)
            tmp = self._audit_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(row,indent=2))
            tmp.replace(self._audit_path)

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        if hasattr(self,'_audit_ids'): self._audit_training()
        p = self.cfg.experiment_profile_settings
        if p.get('leg_target_rate', 0.0) > 0.0:
            max_step = float(p['leg_target_rate']) * self.step_dt
            self._slew_leg += (self.leg_actions - self._slew_leg).clamp(-max_step, max_step)
            self.leg_actions = self._slew_leg.clone()
            # Base implementation scales wheel actions before clipping to max_wheel_vel.
            wheel_scale = self.cfg.wheel_vel_action_scale / self.wheel_action_scale
            target = (self.wheel_actions * wheel_scale).clamp(-self.max_wheel_vel, self.max_wheel_vel)
            delta = float(p['wheel_target_acceleration']) * self.step_dt
            self._slew_wheel += (target - self._slew_wheel).clamp(-delta, delta)
            self.wheel_actions = self._slew_wheel.clone()

    def _custom_reset_random(self, env_ids):
        if self.cfg.experiment_profile_settings.get('random_terrain_map'):
            return random_terrain_map.reset(self, env_ids)
        if self.cfg.experiment_profile_settings.get('unified_locomotion'):
            return self._unified_reset(env_ids)
        # Terrain progression requires observed travel, never the PPO iteration alone.
        if (
            self.cfg.experiment_profile_settings.get('terrain_mix')
            or self.cfg.experiment_profile_settings.get('mixed_slopes')
        ) and hasattr(self, '_slew_leg'):
            ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
            move_up = self._legacy_terrain_pending_move_up[ids]
            move_down = self._legacy_terrain_pending_move_down[ids]
            self.terrain.update_env_origins(env_ids, move_up, move_down)
            self._legacy_terrain_pending_move_up[ids] = False
            self._legacy_terrain_pending_move_down[ids] = False
        super()._custom_reset_random(env_ids)
        if hasattr(self, '_slew_leg'):
            self._slew_leg[env_ids] = self.robot.data.joint_pos[env_ids][:, self._legs_act_idx]
            self._slew_wheel[env_ids] = self.robot.data.joint_vel[env_ids][:, self._wheel_idx]
        if self._legacy_terrain_enabled():
            self._capture_legacy_terrain_start(env_ids)

    def _postprocess_reward_terms(self, terms):
        settings = self.cfg.experiment_profile_settings
        use_slope_pitch = bool(self.cfg.rewards.get('random_terrain_slope_pitch', 0.0))
        use_rel_roll = bool(settings.get('terrain_relative_roll_penalty', False))
        slope_targets = None
        if use_slope_pitch or use_rel_roll:
            if not settings.get('random_terrain_map'):
                raise RuntimeError(
                    'slope orientation rewards require a random-terrain profile'
                )
            slope_targets = random_terrain_map.slope_orientation_targets(self)
        if use_slope_pitch:
            current_pitch, target_pitch, _, _, slope_norm = slope_targets
            pitch_error = torch.atan2(
                torch.sin(current_pitch - target_pitch),
                torch.cos(current_pitch - target_pitch),
            )
            sigma = float(
                getattr(self.cfg, 'random_terrain_slope_pitch_sigma_rad', 0.09)
            )
            if sigma <= 0.0:
                raise ValueError(
                    'random_terrain_slope_pitch_sigma_rad must be positive'
                )
            # Flat patches return zero. The 2% ramp width makes the term reach
            # full strength immediately on every configured slope. A bounded
            # squared error avoids a vanishing reward when the initial policy
            # is far from the terrain-following angle.
            slope_active = (slope_norm / 0.02).clamp(0.0, 1.0)
            terms['random_terrain_slope_pitch'] = (
                torch.tanh((pitch_error / sigma).square()) * slope_active
            )
        if 'flight_penalty' in self.cfg.rewards:
            force = torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w[:, self._desired_contact_link_idx], dim=-1)
            terms['flight_penalty'] = (force.max(1).values < 1.0).float()
        if use_rel_roll:
            _, _, current_roll, target_roll, _ = slope_targets
            roll = torch.atan2(
                torch.sin(current_roll - target_roll),
                torch.cos(current_roll - target_roll),
            )
        else:
            from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
            roll_w, _, _ = euler_xyz_from_quat(T(self.robot.data.root_quat_w))
            roll = wrap_to_pi(roll_w)
        terms['roll_angle_l2'] = roll.square()
        vz_b = T(self.robot.data.root_lin_vel_b)[:, 2]
        rebound_thresh = float(getattr(self.cfg, 'rebound_vel_z_threshold', 0.08))
        terms['upward_rebound_vel_z'] = torch.clamp(vz_b - rebound_thresh, min=0.0).square()
        if 'track_height_square' in terms:
            terms['track_height_square'] = torch.clamp(terms['track_height_square'], max=0.25)
        if getattr(self, '_u_ready', False):
            mask = self._u_closure_mask
            terms = {key: torch.where(mask.reshape(self.num_envs, *([1] * (value.ndim - 1))),
                         torch.zeros_like(value), value) for key, value in terms.items()}
        return super()._postprocess_reward_terms(terms)

    def _setup_scene(self):
        super()._setup_scene()
        if not self.cfg.experiment_cylinder_wheels:
            return
        # Diagnostic runtime layer only: preserve source USD, body mass/inertia,
        # wheel axis, measured radius/width, joint tree and all six closed loops.
        import carb
        from pxr import UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf
        from omni.physx.bindings._physx import SETTING_COLLISION_APPROXIMATE_CYLINDERS
        carb.settings.get_settings().set_bool(SETTING_COLLISION_APPROXIMATE_CYLINDERS, False)
        assert carb.settings.get_settings().get(SETTING_COLLISION_APPROXIMATE_CYLINDERS) is False
        stage=self.sim.stage
        profile=self.cfg.experiment_profile_settings
        if profile.get('cylinder_geometry') != 'measured_flat_tread':
            raise ValueError('Only the measured rolling tread is supported; whole-part AABB is not a tire contact width')
        radius=float(profile['tread_radius_m']);width=float(profile['tread_flat_width_m'])
        centers=profile['tread_centers_y_m']
        assert abs(radius-.0625)<1.e-9 and abs(width-.015)<1.e-9
        specs=[(side,float(center)-width/2.,float(center)+width/2.) for side,center in zip(['left','right'],centers)]
        for eid in range(self.num_envs):
            for side,low,high in specs:
                body=f'/World/envs/env_{eid}/Robot/{side}_wheel_link'
                original=stage.GetPrimAtPath(body+'/collisions')
                mesh=stage.GetPrimAtPath(body+f'/collisions/{side}_wheel_link/mesh')
                assert original.IsValid() and mesh.IsValid()
                UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Set(False)
                assert UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Get() is False
                geom=UsdGeom.Cylinder.Define(stage,body+'/diagnostic_cylinder')
                geom.CreateAxisAttr('Y');geom.CreateRadiusAttr(radius);geom.CreateHeightAttr(high-low)
                geom.AddTranslateOp().Set(Gf.Vec3d(0.,(low+high)/2.,0.))
                prim=geom.GetPrim();UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                offsets=self.cfg.experiment_profile_settings.get('tread_contact_offsets_m')
                if offsets is not None:
                    collision=PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    collision.CreateContactOffsetAttr(float(offsets[0 if side=='left' else 1]))
                    collision.CreateRestOffsetAttr(0.)
                mat,_=UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial('physics')
                if mat:UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,materialPurpose='physics')
                geom.CreateVisibilityAttr('invisible')

    def _low_pass_action_filter(self, actions):
        if self.cfg.experiment_leg_filter:
            n = self._get_leg_policy_action_dim()
            legs = self.last_actions[:, :n] * self.action_low_pass_prev_weight + actions[:, :n] * self.action_low_pass_curr_weight
            # Wheel actions retain their original bandwidth. The base class records
            # filtered applied actions in history/observations and reward differences.
            return torch.cat((legs, actions[:, n:]), dim=-1)
        return super()._low_pass_action_filter(actions)

@configclass
class WheelbipeOwnExperimentRunnerCfg(WheelbipeOwnPPORunnerCfg):
    experiment_name = 'own_low_speed_exploration'
    save_interval = 40
    experiment_stage: int = 1
    def __post_init__(self):
        self.algorithm.learning_rate = 3.e-5
        self.algorithm.schedule = 'adaptive'
        self.algorithm.desired_kl = .01
        self.algorithm.entropy_coef = .001
        self.experiment_stage = int(os.environ.get('WHEELBIPE_STAGE', '1'))
        if self.experiment_stage == 2:
            self.algorithm.learning_rate = 1.e-5
            self.algorithm.schedule = 'fixed'
            self.algorithm.clip_param = .1
            self.algorithm.desired_kl = .005
            self.algorithm.max_grad_norm = .5
            self.algorithm.entropy_coef = .0005

        if self.experiment_stage == 3:
            self.experiment_name = 'wheelbipe_unified_parallel_20261108'
            self.save_interval = 10
            self.algorithm.learning_rate = 5.e-5
            self.algorithm.schedule = 'adaptive'
            self.algorithm.desired_kl = .01
            self.algorithm.entropy_coef = .005
            self.policy.init_noise_std = .35
            self.algorithm.clip_param = .15
            self.algorithm.max_grad_norm = .5

        save_interval = os.environ.get('WHEELBIPE_SAVE_INTERVAL')
        if save_interval is not None:
            self.save_interval = int(save_interval)
            if self.save_interval <= 0:
                raise ValueError('WHEELBIPE_SAVE_INTERVAL must be positive')

        learning_rate = os.environ.get('WHEELBIPE_LEARNING_RATE')
        if learning_rate is not None:
            value = float(learning_rate)
            if value <= 0.0:
                raise ValueError('WHEELBIPE_LEARNING_RATE must be positive')
            self.algorithm.learning_rate = value

        schedule = os.environ.get('WHEELBIPE_SCHEDULE')
        if schedule is not None:
            if schedule not in ('fixed', 'adaptive'):
                raise ValueError(f'WHEELBIPE_SCHEDULE must be fixed or adaptive, got {schedule!r}')
            self.algorithm.schedule = schedule

        desired_kl = os.environ.get('WHEELBIPE_DESIRED_KL')
        if desired_kl is not None:
            value = float(desired_kl)
            if value <= 0.0:
                raise ValueError('WHEELBIPE_DESIRED_KL must be positive')
            self.algorithm.desired_kl = value

        clip_param = os.environ.get('WHEELBIPE_CLIP_PARAM')
        if clip_param is not None:
            value = float(clip_param)
            if value <= 0.0:
                raise ValueError('WHEELBIPE_CLIP_PARAM must be positive')
            self.algorithm.clip_param = value

        entropy_coef = os.environ.get('WHEELBIPE_ENTROPY_COEF')
        if entropy_coef is not None:
            value = float(entropy_coef)
            if value < 0.0:
                raise ValueError('WHEELBIPE_ENTROPY_COEF must be non-negative')
            self.algorithm.entropy_coef = value
