"""Install the unified experiment in an isolated copy of the verified server project."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--tools', type=Path, default=Path(__file__).parent)
a = p.parse_args()
r = a.root
envdir = r / 'project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own'
assert not (r / 'plan.json').exists(), 'Configure only an unused isolated root'
for source, target in [('terrain.py', 'unified_terrain.py'), ('sampling.py', 'unified_sampling.py'), ('environment.py', 'unified_environment.py')]:
    shutil.copyfile(a.tools / source, envdir / target)
profiles = json.loads((a.source / 'profiles.json').read_text())
settings = profiles['rough_curriculum']
settings.update(unified_locomotion=True, terrain_family='unified', command_speed=2.7,
                command_yaw=.8, command_resampling=[1.5, 4.], reset_band=.10,
                terrain_seed=20261108, command_sampler='70% full speed bins,20% slow,10% stop/pure yaw')
settings['reward_weights'].update(track_lin_vel_xy=4., track_lin_vel_xy_square=-.35,
    flat_orientation_x_exp=.15, flat_orientation_y_exp=.15)
settings['reward_parameters'].update(lin_vel_xy_sigma=.36, ang_vel_z_sigma=.16)
(r / 'profiles.json').write_text(json.dumps({'unified_locomotion': settings}, indent=2) + '\n')
s = (envdir / 'experiment.py').read_text()
s = s.replace('from .domain_randomization import', 'from .unified_environment import UnifiedLocomotionMixin, configure_unified\nfrom .domain_randomization import', 1)
s = s.replace('class WheelbipeOwnExperimentEnv(DomainRandomizationMixin, WheelbipeOwnEnv):',
    'class WheelbipeOwnExperimentEnv(UnifiedLocomotionMixin, DomainRandomizationMixin, WheelbipeOwnEnv):')
s = s.replace("        configure_domain_randomization(self, p['domain_randomization'])", "        configure_domain_randomization(self, p['domain_randomization'])\n        configure_unified(self, p)")
s = s.replace('        self._slew_wheel = self.robot.data.joint_vel[:, self._wheel_idx].clone()',
    '        self._slew_wheel = self.robot.data.joint_vel[:, self._wheel_idx].clone()\n        self._init_unified()')
s = s.replace('    def _custom_reset_random(self, env_ids):\n', "    def _custom_reset_random(self, env_ids):\n        if self.cfg.experiment_profile_settings.get('unified_locomotion'):\n            return self._unified_reset(env_ids)\n", 1)
s = s.replace("            tmp = self._audit_path.with_suffix('.tmp')", "            if getattr(self, '_u_ready', False): row['unified'] = self._unified_health()\n            tmp = self._audit_path.with_suffix('.tmp')")
s = s.replace("self.experiment_name = 'wheelbipe_v14_6_dr_20261108'", "self.experiment_name = 'wheelbipe_unified_locomotion_20261108'")
s = s.replace('            self.algorithm.learning_rate = 1.e-4', '            self.algorithm.learning_rate = 5.e-5')
s = s.replace('            self.policy.init_noise_std = .6', '            self.policy.init_noise_std = .35\n            self.algorithm.clip_param = .15\n            self.algorithm.max_grad_norm = .5')
(envdir / 'experiment.py').write_text(s)

# Fresh actor-only transfer is deliberately distinct from exact continuation.
train = r / 'project/scripts/rsl_rl/train.py'
s = train.read_text()
s = s.replace('parser.add_argument("--audit_updates"', 'parser.add_argument("--actor_only_finetune", action="store_true", help="Transfer actor only; reset critic, exploration and Adam for a new task distribution.")\nparser.add_argument("--audit_updates"', 1)
s = s.replace('    target_state.update(compatible_state)', "    if args_cli.actor_only_finetune:\n        compatible_state = {k:v for k,v in compatible_state.items() if k.startswith('actor.')}\n        expanded_state = {}\n        print('[FINETUNE] Actor only: critic and optimizer freshly initialized')\n    target_state.update(compatible_state)", 1)
s = s.replace('    runner.alg.policy.load_state_dict(target_state, strict=True)', '''    runner.alg.policy.load_state_dict(target_state, strict=True)
    if args_cli.actor_only_finetune:
        current = runner.alg.policy.state_dict()
        actor_exact = all(torch.equal(current[k], source_state[k]) for k in current if k.startswith('actor.'))
        critic_fresh = any(not torch.equal(current[k], source_state[k]) for k in current if k.startswith('critic.'))
        assert actor_exact and critic_fresh and not runner.alg.optimizer.state
        runner._actor_transfer_audit = dict(actor_exact_match=True, critic_reinitialized=True,
            optimizer_state_entries=0, iteration_restart=0, source_checkpoint=str(checkpoint_path))''', 1)
s = s.replace('    runner.add_git_repo_to_log(__file__)', '''    runner.add_git_repo_to_log(__file__)
    # Save mastery beside each checkpoint; restarting PPO must not restart its
    # terrain curriculum or silently lose the sampling coverage history.
    original_save = runner.save
    def save_with_curriculum(path, *args, **kwargs):
        result = original_save(path, *args, **kwargs)
        scene = env.unwrapped
        if getattr(scene, '_u_ready', False):
            payload = {name: getattr(scene, name).detach().cpu() for name in
                       ('_u_mastery', '_u_good', '_u_command_hist', '_u_velocity_hist', '_u_reset_modes', '_u_grade_counts', '_u_obs_min', '_u_obs_max')}
            payload.update(_u_causes=scene._u_causes, _u_cmd_min=scene._u_cmd_min, _u_cmd_max=scene._u_cmd_max,
                           _u_transitions=scene._u_transitions)
            torch.save(payload, str(path) + '.curriculum.pt')
        return result
    runner.save = save_with_curriculum''', 1)
s = s.replace('    _notify_env_training_progress(env, int(getattr(runner, "current_learning_iteration", 0)))', '''    if args_cli.resume_training and getattr(env.unwrapped, '_u_ready', False):
        from pathlib import Path
        sidecar = Path(str(args_cli.checkpoint) + '.curriculum.pt')
        if not sidecar.exists():
            raise ValueError('Unified continuation requires its curriculum sidecar: ' + str(sidecar))
        payload = torch.load(sidecar, map_location=env.unwrapped.device, weights_only=False)
        for name, value in payload.items():
            setattr(env.unwrapped, name, value)
        print('[RESUME] Unified terrain mastery and cumulative coverage restored')
    _notify_env_training_progress(env, int(getattr(runner, "current_learning_iteration", 0)))''', 1)
s = s.replace('    _attach_checkpoint_metadata(env_cfg, agent_cfg, args_cli.checkpoint)', '''    _attach_checkpoint_metadata(env_cfg, agent_cfg, args_cli.checkpoint)
    if hasattr(runner, '_actor_transfer_audit'):
        import json
        os.makedirs(os.path.join(log_dir, 'params'), exist_ok=True)
        with open(os.path.join(log_dir, 'params', 'actor_transfer.json'), 'w') as stream:
            json.dump(runner._actor_transfer_audit, stream, indent=2)''', 1)
train.write_text(s)
run = r / 'tools/run.sh'
run.write_text(run.read_text().replace(str(a.source), str(r)))
checkpoint = a.source / 'candidates/rough_curriculum/candidate.pt'
shutil.copyfile(checkpoint, r / 'initialization/rough_actor_3000.pt')
shutil.copytree(checkpoint.parent / 'params', r / 'initialization/params', dirs_exist_ok=True)
plan = dict(experiment='wheelbipe_unified_locomotion_20261108', root=str(r),
    profile='unified_locomotion', num_envs=256, seed=20261108, gpu=0, cpus='0-15',
    physics_dt_s=.0025, policy_dt_s=.02, pilot_updates=16, total_updates=12000,
    checkpoint_interval=100, evaluate_interval=500, validation_gpu=1,
    initialization=dict(checkpoint=str(r / 'initialization/rough_actor_3000.pt'),
        source=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_iteration=3000, transfer='actor only; fresh critic/Adam/iteration; exploration std .35'),
    one_policy_for_all_terrains=True, stairs=False,
    terrain_proportions=dict(flat=.30, wave=.20, rough=.20, slope_up=.15, slope_down=.15),
    command_vx_m_s=[-2.7, 2.7], execution_state='configured')
(r / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(plan, indent=2))
