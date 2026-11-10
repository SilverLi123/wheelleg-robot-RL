"""Measured-CAD nominal model with explicit, bounded training uncertainties.

Copied into the isolated server environment. All ranges are training hypotheses,
not hardware calibration results. Runtime values are read back before training.
"""
import json
import os
from pathlib import Path

import torch
from agent_world.sim620_compat import torch_data as T
from isaaclab.utils.math import quat_apply, quat_apply_inverse


def configure_domain_randomization(cfg, settings):
    from isaaclab.utils.noise import NoiseModelCfg, UniformNoiseCfg
    cfg.use_act_delay = cfg.use_obs_delay = True
    cfg.act_delay_cfg = {'leg_actions': [0, 3], 'wheel_actions': [0, 3]}
    cfg.obs_delay_cfg = {name: [0, 3] for name in
                         ['root_ang_vel_b', 'projected_gravity_b', 'joint_pos', 'joint_vel']}
    cfg.obs_default_time_lag = 0
    cfg.obs_history_len = 7
    cfg.use_self_obs_noise = True
    cfg.self_obs_noise_cfg = {
        name: NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-bound, n_max=bound))
        for name, bound in settings['sensor_noise_uniform_bounds'].items()
    }
    # Multiplication by this reference material leaves the sampled robot-side
    # friction and restitution as the effective wheel/ground pair coefficients.
    cfg.terrain = cfg.terrain.copy()
    cfg.terrain.physics_material = cfg.terrain.physics_material.copy()
    cfg.terrain.physics_material.static_friction = 1.0
    cfg.terrain.physics_material.dynamic_friction = 1.0
    cfg.terrain.physics_material.restitution = 1.0
    cfg.terrain.physics_material.friction_combine_mode = 'multiply'
    cfg.terrain.physics_material.restitution_combine_mode = 'multiply'
    cfg.scene.terrain = cfg.terrain.copy()
    factor = int(os.environ.get('WHEELBIPE_INTEGRATION_SUBSTEPS', '1'))
    if factor not in (1, 2):
        raise ValueError('Supported numerical refinement factors: 1, 2')
    cfg.own_integration_refinement_factor = factor
    cfg.sim.dt /= factor
    cfg.decimation *= factor
    cfg.sim.render_interval = cfg.decimation
    cfg.own_physics_hz *= factor


class DomainRandomizationMixin:
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # The saved delay configuration denotes 0, 5, 10 ms at the original
        # 200 Hz. Preserve those physical delays when refining integration.
        factor = getattr(self.cfg, 'own_integration_refinement_factor', 1)
        if factor > 1:
            ids = env_ids if env_ids is not None else torch.arange(self.num_envs, device=self.device)
            for name in ['act_delay', 'obs_delay']:
                if hasattr(self, name):
                    for buffer in getattr(self, name).values():
                        buffer.set_time_lag(buffer.time_lags[ids].clone() * factor, ids)

    def _init_domain_randomization(self):
        import warp as wp
        p = self.cfg.experiment_profile_settings['domain_randomization']
        robot = self._native_robot
        n = self.num_envs

        def uniform(shape, bounds, device=None):
            return torch.empty(shape, device=device or self.device).uniform_(*bounds)

        # Properties are sampled once per cloned environment, never on a live
        # mechanism during integration; the complete immutable CAD stays on disk.
        mass = T(robot.data.body_mass).clone()
        inertia = T(robot.data.body_inertia).clone()
        nominal_mass = mass.clone()
        base_id = robot.body_names.index('base_link')
        ratios = uniform(mass.shape, p['other_body_mass_scale'])
        ratios[:, base_id] = uniform((n,), p['base_mass_scale'])
        mass *= ratios
        inertia *= ratios[..., None]
        robot.set_masses_index(masses=mass)
        robot.set_inertias_index(inertias=inertia)
        actual_mass = T(robot.data.body_mass).clone()
        actual_inertia = T(robot.data.body_inertia).clone()
        assert torch.allclose(actual_mass, mass, rtol=1e-5, atol=1e-6), 'Mass randomization not applied'
        assert torch.allclose(actual_inertia, inertia, rtol=1e-5, atol=1e-8), 'Inertia randomization not applied'

        # Use a bounded material bucket set, shared across all robot contact
        # shapes of each environment. This includes the runtime tire cylinders.
        view = robot.root_physx_view
        backend_names = [Path(path).name for path in view.link_paths[0]]
        native_order = [backend_names.index(name) for name in robot.body_names]
        engine_mass = T(view.get_masses())[:, native_order].to(mass.device)
        engine_inertia = T(view.get_inertias())[:, native_order].to(inertia.device)
        assert torch.allclose(engine_mass, mass, rtol=1e-5, atol=1e-6), 'Native PhysX mass differs'
        assert torch.allclose(engine_inertia, inertia, rtol=1e-5, atol=1e-8), 'Native PhysX inertia differs'
        material = T(view.get_material_properties()).clone().cpu()
        buckets = torch.empty(64, 3)
        buckets[:, 0].uniform_(*p['static_friction'])
        buckets[:, 1] = buckets[:, 0] * uniform((64,), p['dynamic_static_ratio'], 'cpu')
        buckets[:, 2].uniform_(*p['restitution'])
        chosen = buckets[torch.randint(64, (n,), device='cpu')]
        material[:] = chosen[:, None, :]
        ids = torch.arange(n, dtype=torch.int32, device='cpu')
        view.set_material_properties(wp.from_torch(material, dtype=wp.float32), wp.from_torch(ids, dtype=wp.int32))
        actual_material = T(view.get_material_properties()).cpu()
        assert torch.allclose(actual_material, material, rtol=1e-5, atol=1e-6), 'Material randomization not applied'
        assert torch.all(actual_material[..., 1] <= actual_material[..., 0])

        actuator_ranges = {}
        for name in ['legs_act', 'wheel']:
            actuator = robot.actuators[name]
            actuator.stiffness *= uniform(actuator.stiffness.shape, p['stiffness_scale'])
            actuator.damping *= uniform(actuator.damping.shape, p['motor_damping_scale'])
            limit = actuator.actuator_effort_limit
            limit *= uniform(limit.shape, p['motor_effort_scale'])
            actuator_ranges[name] = dict(stiffness=[float(actuator.stiffness.min()), float(actuator.stiffness.max())],
                damping=[float(actuator.damping.min()), float(actuator.damping.max())],
                effort_limit=[float(limit.min()), float(limit.max())])

        self._dr_rr = uniform((n, 1), p['rolling_resistance_coefficient'])
        self._dr_wrench_bodies = [base_id] + self._wheel_link_idx
        self._dr_normal_cap = actual_mass.sum(1, keepdim=True) * 9.81 * 2.0
        self._dr_push_interval = uniform((n,), p['push_interval_s'])
        self._dr_clock = torch.zeros(n, device=self.device)
        self._dr_push = uniform((n, 2), p['push_force_xy_N'])
        self._dr_contacts = [self.contact_sensor.body_names.index(x) for x in ['left_wheel_link', 'right_wheel_link']]
        self._dr_peak_rolling_torque = 0.0
        self._dr_peak_positive_power = 0.0
        self._dr_contact_steps = 0
        self._dr_airborne_steps = 0
        self._dr_push_steps = 0
        self._dr_runtime = dict(enabled=True, sampling='physical parameters once per environment at startup; delays on resets; noise on observations',
            physics_dt_s=self.physics_dt, policy_dt_s=self.step_dt,
            integration_refinement_factor=getattr(self.cfg, 'own_integration_refinement_factor', 1),
            nominal_total_mass_kg=float(nominal_mass[0].sum()),
            actual_total_mass_range_kg=[float(actual_mass.sum(1).min()), float(actual_mass.sum(1).max())],
            base_mass_range_kg=[float(actual_mass[:, base_id].min()), float(actual_mass[:, base_id].max())],
            actual_mass_scale_range=[float(ratios.min()), float(ratios.max())],
            actual_static_friction_range=[float(actual_material[..., 0].min()), float(actual_material[..., 0].max())],
            actual_dynamic_friction_range=[float(actual_material[..., 1].min()), float(actual_material[..., 1].max())],
            actual_restitution_range=[float(actual_material[..., 2].min()), float(actual_material[..., 2].max())],
            ground_reference_material=dict(static_friction=1., dynamic_friction=1., restitution=1., combine='multiply'),
            rolling_resistance_range=[float(self._dr_rr.min()), float(self._dr_rr.max())],
            inertia_mass_scaling_verified=True, native_physx_mass_inertia_verified=True,
            material_readback_verified=True,
            actuator_actual_ranges=actuator_ranges, settings=p)
        assert self.use_act_delay and self.use_obs_delay and self.use_self_obs_noise, 'Configured delay/noise silently disabled'
        path = self._audit_path.parent / 'domain_randomization.json'
        path.write_text(json.dumps(self._dr_runtime, indent=2) + '\n')
        self._dr_ready = True

    def _apply_action(self):
        super()._apply_action()
        if not getattr(self, '_dr_ready', False):
            return
        robot = self._native_robot
        q = T(robot.data.body_quat_w)[:, self._wheel_link_idx]
        omega_w = T(robot.data.body_ang_vel_w)[:, self._wheel_link_idx]
        omega_y = quat_apply_inverse(q, omega_w)[..., 1]
        contacts = T(self.contact_sensor.data.net_forces_w)[:, self._dr_contacts]
        normal = contacts[..., 2].clamp(min=0.0).minimum(self._dr_normal_cap)
        grounded = normal > 1.0
        local_torque = torch.zeros(self.num_envs, 2, 3, device=self.device)
        local_torque[..., 1] = (-self._dr_rr * normal * 0.0625 * torch.tanh(omega_y / .5)).clamp(-1., 1.) * grounded
        torque_w = quat_apply(q, local_torque)
        power = (torque_w * omega_w).sum(-1)
        assert torch.isfinite(torque_w).all() and float(power.max()) < 1e-4, 'Rolling resistance adds energy'
        assert torch.count_nonzero(torque_w[~grounded]) == 0, 'Rolling resistance while airborne'
        self._dr_peak_rolling_torque = max(self._dr_peak_rolling_torque, float(local_torque.abs().max()))
        self._dr_peak_positive_power = max(self._dr_peak_positive_power, float(power.max()))
        self._dr_contact_steps += int(grounded.sum())
        self._dr_airborne_steps += int((~grounded).sum())

        forces = torch.zeros(self.num_envs, 3, 3, device=self.device)
        torques = torch.zeros_like(forces)
        torques[:, 1:] = torque_w
        self._dr_clock += self.physics_dt
        push_start = self._dr_clock >= self._dr_push_interval
        pushing = push_start & (self._dr_clock < self._dr_push_interval + .1)
        self._dr_push_steps += int(pushing.sum())
        forces[:, 0, :2] = self._dr_push * pushing[:, None]
        ended = self._dr_clock >= self._dr_push_interval + .1
        if ended.any():
            p = self.cfg.experiment_profile_settings['domain_randomization']
            self._dr_clock[ended] = 0.
            self._dr_push_interval[ended] = torch.empty_like(self._dr_push_interval[ended]).uniform_(*p['push_interval_s'])
            self._dr_push[ended] = torch.empty_like(self._dr_push[ended]).uniform_(*p['push_force_xy_N'])
        robot.set_external_force_and_torque(forces, torques, body_ids=self._dr_wrench_bodies, is_global=True)

    def _custom_reset_random(self, env_ids):
        super()._custom_reset_random(env_ids)
        if getattr(self, '_dr_ready', False):
            self._dr_clock[env_ids] = 0.

    def _domain_randomization_health(self):
        return dict(self._dr_runtime,
            peak_rolling_torque_Nm=self._dr_peak_rolling_torque,
            peak_positive_rolling_power_W=self._dr_peak_positive_power,
            wheel_contact_samples=self._dr_contact_steps, wheel_airborne_samples=self._dr_airborne_steps,
            push_environment_substeps=self._dr_push_steps)
