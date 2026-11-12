"""Single-policy terrain curriculum, constraint-preserving resets and coverage audit."""
import json
import math

import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from isaaclab.utils.warp import raycast_mesh
from agent_world.assets.wheelbipe_own import ASSET_DIR, MECHANISM
from agent_world.sim620_compat import torch_data as T
from .unified_terrain import FAMILIES, COLUMN_FAMILY
from .unified_sampling import SPEED_EDGES, BalancedVelocityCommand


def configure_unified(cfg, settings):
    if not settings.get('unified_locomotion'):
        return
    from .unified_terrain import terrain_generator
    cfg.commands.class_type = BalancedVelocityCommand
    cfg.commands.rel_standing_envs = 0.  # The custom sampler owns this probability.
    cfg.terrain.terrain_generator = terrain_generator(settings.get('terrain_seed', 20261108))
    cfg.terrain.max_init_terrain_level = 0
    cfg.scene.terrain = cfg.terrain.copy()
    cfg.episode_length_s = 14.
    # At the expanded speeds a fallen mechanism must reset before a prolonged
    # floor collision destabilizes passive closed-loop joints (old delay .2 s).
    cfg.termination_duration_steps = 2


class UnifiedLocomotionMixin:
    def _get_height_reward_reference_height(self, relative_obs_height, wheel_height_w):
        settings = self.cfg.experiment_profile_settings
        if settings.get('terrain_relative_height_reward', False):
            # The inherited Own configuration uses absolute root z. On a raised
            # slope this penalizes terrain elevation as if it were leg extension.
            # Correct only the reward reference: actor observations and action
            # semantics remain compatible with the transferred checkpoint.
            if not self._use_absolute_height() or self._use_leg_length_height():
                raise ValueError('Terrain height correction requires the audited absolute-height mode')
            if getattr(self, '_u_ready', False):
                ground_z = self._u_heights(
                    T(self.robot.data.root_pos_w)[:, None, :2]
                ).squeeze(1)
            elif settings.get('random_terrain_map', False):
                # The random-map profile does not enable the unified curriculum,
                # so `_u_ready` stays false and its mesh ray query is unavailable.
                # Use the same analytic patch heights that generated the map and
                # placed the robot during reset.
                from . import random_terrain_map

                origin = T(self.terrain.env_origins)
                local_xy = T(self.robot.data.root_pos_w)[:, :2] - origin[:, :2]
                ground_z = (
                    origin[:, 2]
                    + random_terrain_map._heights(self, local_xy)
                )
            else:
                raise ValueError(
                    'terrain_relative_height_reward requires unified terrain '
                    'or random_terrain_map initialization'
                )
            relative_obs_height = relative_obs_height - ground_z
        return super()._get_height_reward_reference_height(relative_obs_height, wheel_height_w)

    def _init_unified(self):
        if not self.cfg.experiment_profile_settings.get('unified_locomotion'):
            return
        n, d = self.num_envs, self.device
        self._u_columns = torch.tensor(COLUMN_FAMILY, device=d)
        self._u_family = self._u_columns[self.terrain.terrain_types.long()].clone()
        initial_level = int(self.cfg.experiment_profile_settings.get('initial_mastery_level', 0))
        if not 0 <= initial_level <= 5:
            raise ValueError('initial_mastery_level must be in [0, 5]')
        self._u_mastery = torch.full((n,), initial_level, device=d, dtype=torch.long)
        self._u_good = torch.zeros(n, device=d, dtype=torch.long)
        self._u_age_limit = torch.empty(n, device=d).uniform_(6., 14.)
        self._u_stuck = torch.zeros(n, device=d)
        self._u_errors = torch.zeros(n, device=d)
        self._u_samples = torch.zeros(n, device=d)
        self._u_episode_start = torch.zeros(n, 2, device=d)
        self._u_command_hist = torch.zeros(5, 9, device=d, dtype=torch.long)
        self._u_velocity_hist = torch.zeros(5, 9, device=d, dtype=torch.long)
        self._u_reset_modes = torch.zeros(5, 3, device=d, dtype=torch.long)
        self._u_causes = {k: 0 for k in ('boundary', 'random_timeout', 'stuck', 'failure', 'timeout')}
        self._u_grade_counts = torch.zeros(3, device=d, dtype=torch.long)
        self._u_cmd_min, self._u_cmd_max = math.inf, -math.inf
        self._u_peak_penetration = 0.
        self._u_moving_reset_error = 0.
        self._u_last_coverage = 0
        self._u_transitions = dict.fromkeys(('resamples', 'sign_reversals', 'braking_to_zero', 'pure_yaw', 'combined_turn'), 0)
        self._u_obs_min = torch.full((35,), math.inf, device=d)
        self._u_obs_max = torch.full((35,), -math.inf, device=d)
        self._u_eval = False
        self._u_closure_mask = torch.zeros(n, device=d, dtype=torch.bool)
        self._u_closure_counts = dict(resets=0, nonfinite_resets=0, events=0, widespread_streak=0)
        self._u_closure_peak = 0.
        # Isaac Lab 3 no longer retains TerrainImporter.warp_meshes. Build the
        # query mesh from the actual USD collision mesh, with its world transform.
        import numpy as np
        from pxr import Usd, UsdGeom
        from isaaclab.utils.warp import convert_to_warp_mesh
        vertices, faces, offset = [], [], 0
        for path in self.terrain.terrain_prim_paths:
            for prim in Usd.PrimRange(self.sim.stage.GetPrimAtPath(path)):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(prim)
                points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
                assert np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3)
                matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                points = np.c_[points, np.ones(len(points))] @ matrix
                vertices.append(points[:, :3].astype(np.float32))
                faces.append(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3) + offset)
                offset += len(points)
        assert vertices, 'No actual terrain mesh found for reset queries'
        self._u_mesh = convert_to_warp_mesh(np.concatenate(vertices), np.concatenate(faces), device=self.device)
        bank = json.loads((ASSET_DIR / 'reset_bank.json').read_text())
        self._u_bank, self._u_endpoints = {}, {}
        for side in ('left', 'right'):
            q = torch.tensor(bank['banks'][side], device=d)
            points = torch.tensor(bank['wheel_endpoints'][side], device=d)
            names = [side + '_front1_joint', side + '_rear1_joint']
            ids = [self._bank_names.index(name) for name in names]
            nominal = torch.tensor([MECHANISM['nominal_native_q'][name] for name in names], device=d)
            keep = (q[:, ids] - nominal).abs().amax(1) <= self.cfg.experiment_profile_settings['reset_band']
            assert keep.any()
            self._u_bank[side], self._u_endpoints[side] = q[keep], points[keep]
        self._u_ready = True

    def _reset_idx(self, env_ids):
        if not getattr(self, '_u_ready', False):
            return super()._reset_idx(env_ids)
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return super()._reset_idx(ids)
        # Capture these BEFORE DirectRLEnv._reset_idx clears episode_length_buf.
        age = self.episode_length_buf[ids] * self.step_dt
        travel = torch.linalg.vector_norm(T(self.robot.data.root_pos_w)[ids, :2] - self._u_episode_start[ids], dim=-1)
        error = self._u_errors[ids] / self._u_samples[ids].clamp_min(1)
        failed = self.reset_terminated[ids]
        boundary = getattr(self, '_u_boundary_mask', torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))[ids]
        # A fast successful crossing can reach the tile edge before 3 seconds.
        sufficient_episode = (age >= 3.) | (boundary & (age >= 1.2) & (travel >= 3.))
        good = sufficient_episode & (travel >= 1.5) & (error < .45) & ~failed
        self._u_good[ids] = torch.where(good, self._u_good[ids] + 1, torch.zeros_like(self._u_good[ids]))
        promote = self._u_good[ids] >= 2
        self._u_mastery[ids] = (self._u_mastery[ids] + promote.long()).clamp(0, 5)
        self._u_good[ids[promote]] = 0
        early_failure = failed & (age < 2.)
        self._u_mastery[ids] = (self._u_mastery[ids] - early_failure.long()).clamp_min(0)
        for name in self._u_causes:
            if name in ('failure', 'timeout'):
                mask = self.reset_terminated if name == 'failure' else self.reset_time_outs
            else:
                mask = getattr(self, '_u_' + name + '_mask', torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
            self._u_causes[name] += int(mask[ids].sum())
        # Keep family proportions fixed per rollout; vary tile, difficulty and
        # physical start within each family. No family can disappear after failures.
        if not self._u_eval:
            family = self._u_family[ids]
            col = self.terrain.terrain_types[ids].clone()
            for f in range(5):
                members = family == f
                columns = (self._u_columns == f).nonzero().flatten()
                col[members] = columns[torch.randint(len(columns), (int(members.sum()),), device=self.device)]
            level = self._u_mastery[ids].clone()
            lottery = torch.rand(len(ids), device=self.device)
            replay = lottery < .2
            level[replay] = (torch.rand(int(replay.sum()), device=self.device) * (level[replay] + 1)).long()
            level[lottery > .9] += 1
            level.clamp_(0, 5)
            self.terrain.terrain_types[ids] = col
            self.terrain.terrain_levels[ids] = level
            self.terrain.env_origins[ids] = self.terrain.terrain_origins[level, col]
        self._u_errors[ids] = self._u_samples[ids] = self._u_stuck[ids] = 0.
        self._u_age_limit[ids] = torch.empty(len(ids), device=self.device).uniform_(6., 14.)
        super()._reset_idx(ids)

    def _u_heights(self, xy):
        starts = torch.zeros(*xy.shape[:-1], 3, device=self.device)
        starts[..., :2] = xy
        starts[..., 2] = 20.
        directions = torch.zeros_like(starts)
        directions[..., 2] = -1.
        hits = raycast_mesh(starts, directions, self._u_mesh, max_dist=40.)[0]
        if not torch.isfinite(hits).all():
            raise RuntimeError('Reset ray missed the actual collision terrain')
        return hits[..., 2]

    def _unified_reset(self, env_ids):
        if not getattr(self, '_u_ready', False):
            # During base initialization the validated Own reset remains available.
            from .env import WheelbipeOwnEnv
            return WheelbipeOwnEnv._custom_reset_random(self, env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        native_q = T(self._native_robot.data.default_joint_pos)[ids].clone()
        endpoints = []
        for side in ('left', 'right'):
            bank = self._u_bank[side]
            sample = torch.randint(len(bank), (n,), device=self.device)
            names = [name for name in self._bank_names if name.startswith(side + '_')]
            src = [self._bank_names.index(name) for name in names]
            dst = [self.robot.joint_names.index(name) for name in names]
            native_q[:, dst] = bank[sample][:, src]
            endpoints.append(self._u_endpoints[side][sample])
        endpoints = torch.stack(endpoints, 1)
        q = native_q * self.robot.sign + self.robot.offset
        origin = self.terrain.env_origins[ids]
        xy = origin[:, :2] + torch.empty(n, 2, device=self.device).uniform_(-2., 2.)
        yaw = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        # Include frequent longitudinal up/down traversals as well as oblique grades.
        axial = torch.rand(n, device=self.device) < .6
        yaw[axial] = torch.randint(2, (int(axial.sum()),), device=self.device) * math.pi + torch.empty(int(axial.sum()), device=self.device).uniform_(-.15, .15)
        offsets = xy.new_tensor([[.25, 0.], [-.25, 0.], [0., .25], [0., -.25]])
        ground = self._u_heights(xy[:, None] + offsets)
        gx, gy = (ground[:, 0] - ground[:, 1]) / .5, (ground[:, 2] - ground[:, 3]) / .5
        roll = torch.atan(gy * yaw.cos() - gx * yaw.sin())
        pitch = -torch.atan(gx * yaw.cos() + gy * yaw.sin())
        choice = torch.rand(n, device=self.device)
        mode = torch.where(choice < .45, 0, torch.where(choice < .8, 1, 2))
        override = getattr(self, '_u_reset_mode_override', None)
        if override is not None:
            mode = override[ids].clone()
        perturb = mode == 2
        jitter = torch.where(perturb, .07, .015)
        roll += torch.empty(n, device=self.device).uniform_(-1., 1.) * jitter
        pitch += torch.empty(n, device=self.device).uniform_(-1., 1.) * jitter
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        rotated = quat_apply(quat[:, None].expand(-1, 2, -1), endpoints)
        # Sample underneath and around the finite wheel footprint. Raise the root
        # enough to avoid starting any tire below the mesh, including on slopes.
        footprint = xy.new_tensor([[0., 0.], [.03, 0.], [-.03, 0.], [0., .02], [0., -.02]])
        tire_ground = self._u_heights(xy[:, None, None] + rotated[:, :, None, :2] + footprint)
        clearance = .004 + torch.rand(n, device=self.device) * .003
        z = (tire_ground.amax(-1) - rotated[:, :, 2] + .0625).amax(-1) + clearance
        root = T(self._native_robot.data.default_root_state)[ids].clone()
        root[:, :2], root[:, 2], root[:, 3:7] = xy, z, quat
        root[:, 7:] = 0.
        moving = mode != 0
        vx = self.command_generator.command[ids, 0].clone()
        vx += torch.empty(n, device=self.device).uniform_(-.2, .2)
        vx = vx.clamp(-2.7, 2.7) * moving
        vy = torch.empty(n, device=self.device).uniform_(-.15, .15) * perturb
        wz = torch.empty(n, device=self.device).uniform_(-.4, .4) * perturb
        velocity = torch.stack((vx, vy, torch.zeros_like(vx)), -1)
        root[:, 7:10] = quat_apply(quat, velocity)
        root[:, 10:13] = quat_apply(quat, torch.stack((torch.zeros_like(wz), torch.zeros_like(wz), wz), -1))
        dq = torch.zeros_like(q)
        # Policy wheel coordinates use +Y rotation for forward rolling.
        wheel_speeds = torch.stack((vx - .2502 * wz, vx + .2437 * wz), -1) / .0625
        dq[:, self._wheel_idx] = wheel_speeds
        self.robot.write_joint_state_to_sim(q, dq, env_ids=ids)
        self.robot.set_joint_position_target(q, env_ids=ids)
        self.robot.set_joint_velocity_target(dq, env_ids=ids)
        self.robot.set_joint_effort_target(torch.zeros_like(q), env_ids=ids)
        self.robot.write_root_state_to_sim(root, ids)
        self._spring_outside[ids] = False
        self.spring_force_rand[ids] = 0.
        self._own_reset_count += n
        self._sync_policy_state()
        self._u_episode_start[ids] = xy
        self._u_reset_modes += torch.bincount(self._u_family[ids] * 3 + mode, minlength=15).reshape(5, 3)
        penetration = (tire_ground - (root[:, None, None, 2] + rotated[:, :, None, 2] - .0625)).max()
        self._u_peak_penetration = max(self._u_peak_penetration, float(penetration))
        assert float(penetration) <= 1e-5, 'Tire starts below terrain'
        if getattr(self, '_dr_ready', False):
            self._dr_clock[ids] = 0.
        if hasattr(self, '_slew_leg'):
            self._slew_leg[ids] = self.robot.data.joint_pos[ids][:, self._legs_act_idx]
            self._slew_wheel[ids] = dq[:, self._wheel_idx]

    def _u_closure_errors(self):
        bp = T(self._native_robot.data.body_pos_w)[:, self._audit_ids]
        bq = T(self._native_robot.data.body_quat_w)[:, self._audit_ids]
        points = bp + quat_apply(bq, self._audit_anchors.expand(self.num_envs, -1, -1, -1))
        return torch.linalg.vector_norm(points[:, :, 0] - points[:, :, 1], dim=-1)

    def _u_check_closure(self):
        error = self._u_closure_errors()
        finite = torch.isfinite(error).all(-1)
        finite &= torch.isfinite(T(self._native_robot.data.joint_pos)).all(-1)
        finite &= torch.isfinite(T(self._native_robot.data.joint_vel)).all(-1)
        finite &= torch.isfinite(T(self._native_robot.data.root_state_w)).all(-1)
        per_env = torch.nan_to_num(error, nan=float('inf'), posinf=float('inf')).amax(-1)
        self._u_closure_mask = (per_env > .005) | ~finite
        finite_peak = torch.where(torch.isfinite(per_env), per_env, 0.).max()
        self._u_closure_peak = max(self._u_closure_peak, float(finite_peak))
        self._audit_peak_m = max(self._audit_peak_m, float(finite_peak))
        count = int(self._u_closure_mask.sum())
        widespread = count >= max(4, math.ceil(self.num_envs * .1))
        self._u_closure_counts['widespread_streak'] = self._u_closure_counts['widespread_streak'] + 1 if widespread else 0
        if count:
            import time
            ids = self._u_closure_mask.nonzero().flatten()
            self._u_closure_counts['resets'] += count
            self._u_closure_counts['nonfinite_resets'] += int((~finite).sum())
            self._u_closure_counts['events'] += 1
            row = dict(unix=time.time(), policy_step=self.common_step_counter,
                affected=count, environment_ids=ids[:64].tolist(),
                finite_peak_m=float(finite_peak), nonfinite=int((~finite).sum()),
                action='terminate_and_reset_affected_environments')
            with self._audit_path.with_name('closure_recovery.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
        if self._u_closure_counts['widespread_streak'] >= 3:
            raise RuntimeError('RECOVERABLE_PHYSICS: widespread closed-loop failure persists after environment resets')
        return self._u_closure_mask

    def _get_dones(self):
        terminated, timeout = super()._get_dones()
        if not getattr(self, '_u_ready', False):
            return terminated, timeout
        closure = self._u_check_closure()
        self._termination_reason_masks['closed_loop'] = closure.clone()
        xy = T(self.robot.data.root_pos_w)[:, :2]
        relative = (xy - self.terrain.env_origins[:, :2]).abs()
        self._u_boundary_mask = (relative[:, 0] > 7.) | (relative[:, 1] > 5.)
        age = self.episode_length_buf * self.step_dt
        self._u_random_timeout_mask = age >= self._u_age_limit
        cmd = self.command_generator.command[:, 0]
        vx = T(self.robot.data.root_lin_vel_b)[:, 0]
        slow = (cmd.abs() > .7) & (vx.abs() < .12) & (age > 3.)
        self._u_stuck = torch.where(slow, self._u_stuck + self.step_dt, torch.zeros_like(self._u_stuck))
        self._u_stuck_mask = self._u_stuck > 2.
        self._termination_reason_masks['stuck'] = self._u_stuck_mask.clone()
        native_velocity = T(self._native_robot.data.joint_vel)
        native_outlier = (~torch.isfinite(native_velocity).all(-1)) | (native_velocity.abs().amax(-1) > 200.)
        self._termination_reason_masks['native_joint_outlier'] = native_outlier
        terminated |= self._u_stuck_mask | native_outlier | closure
        timeout |= self._u_boundary_mask | self._u_random_timeout_mask
        # A physics reset is a failure, never a truncation with value bootstrap.
        timeout &= ~closure
        self._u_errors += torch.nan_to_num((cmd - vx).abs(), nan=5.4, posinf=5.4, neginf=5.4)
        self._u_samples += 1
        edges = vx.new_tensor(SPEED_EDGES[1:-1])
        bins = torch.bucketize(cmd.contiguous(), edges)
        actual_bins = torch.bucketize(vx.contiguous(), edges)
        self._u_command_hist += torch.bincount(self._u_family * 9 + bins, minlength=45).reshape(5, 9)
        valid = ~closure
        self._u_velocity_hist += torch.bincount((self._u_family * 9 + actual_bins)[valid], minlength=45).reshape(5, 9)
        self._u_cmd_min = min(self._u_cmd_min, float(cmd.min()))
        self._u_cmd_max = max(self._u_cmd_max, float(cmd.max()))
        vz = T(self.robot.data.root_lin_vel_w)[:, 2]
        grade = torch.where(vz > .03, 2, torch.where(vz < -.03, 0, 1))
        grade_mask = (self._u_family >= 3) & (vx.abs() > .2) & valid
        self._u_grade_counts += torch.bincount(grade[grade_mask], minlength=3)
        return terminated, timeout

    def _get_rewards(self):
        rewards = super()._get_rewards()
        if getattr(self, '_u_ready', False):
            # Discard numerical reward spikes from broken states. Terminal
            # observations are generated only after the normal per-env reset.
            rewards = torch.where(self._u_closure_mask, torch.full_like(rewards, -1.), rewards)
        return rewards

    def _get_observations(self):
        observations = super()._get_observations()
        if getattr(self, '_u_ready', False):
            policy = observations['policy']
            if not torch.isfinite(policy).all():
                raise FloatingPointError('Unified policy observation contains nonfinite values')
            self._u_obs_min = torch.minimum(self._u_obs_min, policy.amin(0))
            self._u_obs_max = torch.maximum(self._u_obs_max, policy.amax(0))
        return observations

    def _unified_health(self):
        return dict(single_shared_policy=True, families=list(FAMILIES),
            environments_per_family=torch.bincount(self._u_family, minlength=5).tolist(),
            speed_bin_edges=list(SPEED_EDGES), command_samples=self._u_command_hist.tolist(),
            actual_velocity_samples=self._u_velocity_hist.tolist(),
            command_transitions=self._u_transitions,
            policy_observation_min=self._u_obs_min.tolist(), policy_observation_max=self._u_obs_max.tolist(),
            command_range_observed=[self._u_cmd_min, self._u_cmd_max],
            reset_modes=['stationary', 'rolling', 'recovery'], resets=self._u_reset_modes.tolist(),
            reset_causes=self._u_causes, slope_vertical_motion_samples_down_level_up=self._u_grade_counts.tolist(),
            mastery_level_histogram=torch.bincount(self._u_mastery, minlength=6).tolist(),
            sampled_level_histogram=torch.bincount(self.terrain.terrain_levels.long(), minlength=6).tolist(),
            max_initial_tire_penetration_m=self._u_peak_penetration,
            closure_recovery=dict(self._u_closure_counts, pre_reset_finite_peak_m=self._u_closure_peak,
                threshold_m=.005, mode='per_environment_failure_reset'),
            curriculum='two successful >=3s/>=1.5m episodes or >=1.2s/>=3m boundary crossings, vx MAE <.45; 20% easier replay, 10% next-level probes',
            termination_semantics='failure terminates; tile boundary and randomized 6-14s horizon truncate')
