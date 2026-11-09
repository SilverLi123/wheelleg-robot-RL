"""Held-out terrain/speed matrix and executable reset/physics checks."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=float, default=8.)
p.add_argument('--seed', type=int, default=71029)
p.add_argument('--level', type=int, default=0)
p.add_argument('--reset-checks', action='store_true')
p.add_argument('--recovery-checks', action='store_true')
p.add_argument('--profile', default='unified_locomotion')
p.add_argument('--reward-audit', action='store_true')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
os.environ['WHEELBIPE_EXPERIMENT'] = a.profile
a.output.mkdir(parents=True, exist_ok=True)
report = dict(status='INITIALIZING', checkpoint=str(a.checkpoint), seed=a.seed, level=a.level)
app = None
started = time.monotonic()
try:
    app = AppLauncher(a, fast_shutdown=True).app
    import torch
    from isaaclab.utils.math import quat_apply
    from agent_world.sim620_compat import torch_data as T
    from agent_tasks.direct.wheelbipe.wheelbipe_own.experiment import WheelbipeOwnExperimentEnv, WheelbipeOwnExperimentPlayCfg
    from agent_tasks.direct.wheelbipe.wheelbipe_own.unified_sampling import sample_commands
    from agent_tasks.direct.wheelbipe.wheelbipe_own.unified_terrain import FAMILIES, COLUMN_FAMILY, progressive_rough
    from scripts.rsl_rl.checkpoint_utils import prepare_checkpoint
    import numpy as np

    cfg = WheelbipeOwnExperimentPlayCfg()
    cfg.seed = a.seed
    cfg.sim.device = a.device or 'cuda:0'
    # No training tile seed is reused for this matrix.
    cfg.terrain.terrain_generator.seed = a.seed
    cfg.scene.terrain = cfg.terrain.copy()
    speeds = [-2.7, -2.1, -1.5, -.8, 0., .8, 1.5, 2.1, 2.7]
    count = len(speeds) * 5
    cfg.scene.num_envs = count
    prepare_checkpoint(cfg, None, str(a.checkpoint))
    saved = torch.load(a.checkpoint, map_location=cfg.sim.device, weights_only=False)
    state = {k.removeprefix('actor.'): v for k, v in saved['model_state_dict'].items() if k.startswith('actor.')}
    weights = sorted((int(k.split('.')[0]), v) for k, v in state.items() if k.endswith('.weight'))
    layers = []
    for i, (_, weight) in enumerate(weights):
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        if i < len(weights) - 1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to(cfg.sim.device).eval()
    actor.load_state_dict(state, strict=True)
    env = WheelbipeOwnExperimentEnv(cfg)
    env._u_eval = True
    ids = torch.arange(count, device=env.device)
    env._u_family = ids // len(speeds)
    columns = torch.tensor([0, 6, 10, 14, 17], device=env.device)[env._u_family]
    env.terrain.terrain_types[:] = columns
    env.terrain.terrain_levels[:] = a.level
    env.terrain.env_origins[:] = env.terrain.terrain_origins[a.level, columns]
    # Hold-out performance uses stationary starts. The reset tests explicitly
    # exercise rolling/recovery separately, so moving starts cannot hide failures.
    env._u_reset_mode_override = torch.zeros(count, device=env.device, dtype=torch.long)
    commands = torch.zeros(count, 3, device=env.device)
    commands[:, 0] = torch.tensor(speeds, device=env.device).repeat(5)
    cg = env.command_generator
    compute_original, resample_original = cg.compute, cg._resample_command
    def fixed():
        cg.command.copy_(commands)
        cg.is_standing_env[:] = False
    def compute(dt):
        compute_original(dt)
        fixed()
    def resample(env_ids):
        resample_original(env_ids)
        fixed()
    cg.compute, cg._resample_command = compute, resample

    checks = {}
    samples, _ = sample_commands(100000, env.device)
    assert samples[:, 0].min() == -2.7 and samples[:, 0].max() == 2.7
    hist = torch.histc(samples[:, 0], bins=9, min=-2.7, max=2.7)
    assert bool((hist > 3000).all()), hist
    assert float((samples[:, 0] * samples[:, 2]).abs().max()) <= 1.50001
    checks['sampler_100000'] = dict(histogram=hist.tolist(), min=float(samples[:, 0].min()), max=float(samples[:, 0].max()))
    singletons = torch.cat([sample_commands(1, env.device)[0] for _ in range(1000)])
    singleton_hist = torch.histc(singletons[:, 0], bins=9, min=-2.7, max=2.7)
    assert bool((singleton_hist > 35).all()), singleton_hist
    checks['singleton_sampler_1000'] = singleton_hist.tolist()
    c = cfg.terrain.terrain_generator.sub_terrains['rough'].copy()
    c.seed = cfg.terrain.terrain_generator.seed
    c.size = (4., 4.)
    c.horizontal_scale, c.vertical_scale = .1, .001
    np.random.seed(10)
    low = progressive_rough.__wrapped__(0., c)[0]
    np.random.seed(10)
    high = progressive_rough.__wrapped__(1., c)[0]
    assert np.ptp(high) > 3 * np.ptp(low)
    checks['rough_curriculum_height_range_mm'] = [int(np.ptp(low)), int(np.ptp(high))]

    if a.reset_checks:
        for mode in range(3):
            env._u_reset_mode_override[:] = mode
            obs, _ = env.reset()
            q = T(env._native_robot.data.joint_pos)
            dq = env.robot.data.joint_vel[:, env._wheel_idx]
            vx = T(env.robot.data.root_lin_vel_b)[:, 0]
            assert torch.isfinite(q).all()
            if mode == 0:
                assert float(vx.abs().max()) < 1e-4
            else:
                mismatch = (dq.mean(-1) * .0625 - vx).abs().max()
                assert float(mismatch) < .002, mismatch
            assert float(env._u_peak_penetration) <= 1e-5
            checks['reset_mode_' + str(mode)] = dict(velocity_range=[float(vx.min()), float(vx.max())], no_tire_penetration=True)
        # Verify truncation without integrating a teleported robot.
        root = T(env._native_robot.data.root_state_w).clone()
        root[:, 0] = env.terrain.env_origins[:, 0] + 7.1
        env.robot.write_root_state_to_sim(root, ids)
        _, timeouts = env._get_dones()
        assert bool(timeouts.all()) and bool(env._u_boundary_mask.all())
        checks['tile_boundary_is_truncation'] = True
        # Promotion must see pre-reset age and distance, and side effects must
        # not zero mastery before assessment.
        env._u_eval = False
        env._u_good[:] = 1
        env._u_mastery[:] = 0
        env.episode_length_buf[:] = int(4. / env.step_dt)
        env._u_samples[:] = 10
        env._u_errors[:] = .1
        env.reset_terminated[:] = False
        env._u_episode_start[:] = root[:, :2] - root.new_tensor([2., 0.])
        env._reset_idx(ids)
        assert bool((env._u_mastery == 1).all())
        checks['promotion_uses_pre_reset_episode'] = True
        env._u_eval = True
        env.terrain.terrain_levels[:] = a.level
        env.terrain.terrain_types[:] = columns
        env.terrain.env_origins[:] = env.terrain.terrain_origins[a.level, columns]

    env._u_reset_mode_override[:] = 0
    obs, _ = env.reset()
    fixed()
    env.command.copy_(commands)
    obs = env._get_observations()
    injected_recovery_resets = 0
    if a.recovery_checks:
        original_errors = env._u_closure_errors
        for index, fault in [(0, .008), (1, float('nan'))]:
            def injected(index=index, fault=fault):
                errors = original_errors().clone()
                errors[index, 0] = fault
                return errors
            env._u_closure_errors = injected
            with torch.inference_mode():
                obs, reward, done, timeout, _ = env.step(actor(obs['policy']))
            assert bool(done[index]) and not bool(timeout[index])
            assert int(env._u_closure_mask.sum()) == 1
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            assert float(reward[index]) == -1.
            env._u_closure_errors = original_errors
            with torch.inference_mode():
                obs, _, _, _, _ = env.step(actor(obs['policy']))
            assert not bool(env._u_closure_mask[index])
            checks['isolated_fault_' + str(index)] = dict(terminated=True, truncated=False,
                other_environments_preserved=True, next_step_finite=True)
        def widespread():
            return torch.full_like(original_errors(), .008)
        env._u_closure_errors = widespread
        detected = False
        try:
            with torch.inference_mode():
                for _ in range(3):
                    env.step(actor(obs['policy']))
        except RuntimeError as exc:
            assert 'RECOVERABLE_PHYSICS' in str(exc)
            detected = True
        assert detected
        checks['persistent_widespread_fault_requests_restart'] = True
        env._u_closure_errors = original_errors
        env._u_closure_counts['widespread_streak'] = 0
        with torch.inference_mode():
            obs, _ = env.reset()
            fixed()
            env.command.copy_(commands)
            obs = env._get_observations()
        injected_recovery_resets = env._u_closure_counts['resets']
    errors = torch.zeros(count, device=env.device)
    speed_sum = torch.zeros_like(errors)
    failures = torch.zeros_like(errors)
    truncations = torch.zeros_like(errors)
    leg_peak = wheel_peak = closure_peak = 0.
    frames = []
    height_audit = []
    steps = round(a.seconds / env.step_dt)
    with torch.inference_mode():
        for step in range(steps):
            action = actor(obs['policy'])
            if a.reward_audit and step % 25 == 0:
                root = T(env.robot.data.root_pos_w)
                ground = env._u_heights(root[:, None, :2]).squeeze(1)
                target = env._get_height_reward_target_height()
                ref = env._get_height_reward_reference_height(root[:, 2], T(env.robot.data.body_pos_w)[:, env._wheel_link_idx, 2])
                height_audit.append(torch.stack((root[:, 2], ground, target, ref), -1).cpu().numpy())
            obs, reward, done, timeout, _ = env.step(action)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all() and torch.isfinite(action).all()
            vx = T(env.robot.data.root_lin_vel_b)[:, 0]
            failures += done
            truncations += timeout
            if step >= 50:
                errors += (vx - commands[:, 0]).abs()
                speed_sum += vx
            tau = T(env._native_robot.data.applied_torque)
            leg_peak = max(leg_peak, float(tau[:, env._legs_act_idx].abs().max()))
            wheel_peak = max(wheel_peak, float(tau[:, env._wheel_idx].abs().max()))
            bp, bq = T(env._native_robot.data.body_pos_w)[:, env._audit_ids], T(env._native_robot.data.body_quat_w)[:, env._audit_ids]
            anchors = bp + quat_apply(bq, env._audit_anchors.expand(count, -1, -1, -1))
            closure_peak = max(closure_peak, float(torch.linalg.vector_norm(anchors[:, :, 0] - anchors[:, :, 1], dim=-1).max()))
            frames.append(torch.stack((commands[:, 0], vx, done, timeout), -1).cpu().numpy())
            if (step + 1) % 100 == 0:
                print(json.dumps(dict(step=step + 1, physics_closure_peak_mm=closure_peak * 1000, failures=int(failures.sum()))), flush=True)
    assert leg_peak <= 40.001 and wheel_peak <= 5.001
    closure_ok = closure_peak < .005
    denominator = max(steps - 50, 1)
    rows = []
    for index in range(count):
        rows.append(dict(family=FAMILIES[index // 9], command=speeds[index % 9],
            vx_mae=float(errors[index] / denominator), mean_vx=float(speed_sum[index] / denominator),
            failures=int(failures[index]), truncations=int(truncations[index])))
    np.savez_compressed(a.output / 'trajectory.npz', frames=np.stack(frames))
    if height_audit:
        heights = np.stack(height_audit)
        np.savez_compressed(a.output / 'height_audit.npz', samples=heights)
        report['height_reward_audit'] = dict(
            columns=['root_z_m', 'ground_z_m', 'target_m', 'active_reward_reference_m'],
            profile=a.profile,
            per_family={family: dict(
                absolute_height_error_rms_m=float(np.sqrt(np.mean((heights[:, index*9:(index+1)*9, 0] - heights[:, index*9:(index+1)*9, 2]) ** 2))),
                clearance_error_rms_m=float(np.sqrt(np.mean((heights[:, index*9:(index+1)*9, 0] - heights[:, index*9:(index+1)*9, 1] - heights[:, index*9:(index+1)*9, 2]) ** 2))),
                active_height_error_rms_m=float(np.sqrt(np.mean((heights[:, index*9:(index+1)*9, 3] - heights[:, index*9:(index+1)*9, 2]) ** 2))),
            ) for index, family in enumerate(FAMILIES)})
    recovery = env._unified_health().get('closure_recovery', {})
    normal_window_recovery_resets = recovery.get('resets', 0) - injected_recovery_resets
    warning = closure_peak >= .005 or normal_window_recovery_resets > 0
    report.update(status='WARN_CLOSURE' if warning else 'PASS', physics_pass=closure_ok and not warning, policy_convergence_claim=False,
        seconds=a.seconds, checks=checks, rows=rows,
        vx_mae=float(errors.mean() / denominator), failures=int(failures.sum()),
        leg_peak_Nm=leg_peak, wheel_peak_Nm=wheel_peak, closure_peak_m=closure_peak,
        normal_window_closure_resets=normal_window_recovery_resets,
        injected_fault_resets=injected_recovery_resets,
        pre_reset_closure_peak_m=recovery.get('pre_reset_finite_peak_m'),
        coverage=env._unified_health())
    env.close()
except BaseException:
    report.update(status='ERROR', error=traceback.format_exc())
    raise
finally:
    report['wall_seconds'] = time.monotonic() - started
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows', 'coverage')}), flush=True)
    if app is not None:
        app.close(exit_code=0 if report['status'] in ('PASS', 'WARN_CLOSURE') else 1)
