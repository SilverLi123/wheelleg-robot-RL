"""Record the unified policy on a declared terrain without changing training."""
import argparse
import hashlib
import importlib.metadata as md
import json
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--steps', type=int, default=1500)
parser.add_argument('--mode', choices=['policy', 'hold', 'continuous'], default='policy')
parser.add_argument('--standstill', action='store_true')
parser.add_argument('--fixed-command', type=float, nargs=2, metavar=('VX', 'YAW_RATE'))
parser.add_argument('--record', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--height', type=float, default=.255)
parser.add_argument('--profile', required=True)
parser.add_argument('--sequence', action='store_true', help='5s straight, 5s turn, 5s reverse; deterministic actor with DR.')
parser.add_argument('--terrain-column', type=int)
parser.add_argument('--terrain-level', type=int, default=0)
parser.add_argument(
    '--mixed-eval-terrain',
    action='store_true',
    help='Use the same one-column-per-family full-difficulty terrain as mixed_slopes_eval.py.',
)
parser.add_argument('--light-slope-case', choices=('flat', 'up', 'down'), help=argparse.SUPPRESS)
parser.add_argument(
    '--allow-profile-override',
    action='store_true',
    help='Allow a terrain-only profile override for a frozen checkpoint.',
)
parser.add_argument(
    '--random-terrain-overview',
    action='store_true',
    help='Render one complete connected random-terrain map from an overhead camera.',
)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.light_slope_case:
    import light_slope
    if (args.profile != 'flat_seed42' or args.terrain_column is not None
            or args.fixed_command != [.5, 0.] or args.mode != 'policy'
            or args.sequence or args.steps != 300):
        parser.error('Use light_slope.py for the fixed flat_seed42 diagnostic')
import os
os.environ['WHEELBIPE_EXPERIMENT'] = args.profile
if args.standstill and args.fixed_command:
    parser.error('Choose either standstill or a fixed command')
if args.mixed_eval_terrain and args.terrain_column is None:
    parser.error('--mixed-eval-terrain requires --terrain-column')
if args.mixed_eval_terrain and args.profile != 'mixed_slopes_20261111':
    parser.error('--mixed-eval-terrain requires --profile mixed_slopes_20261111')
if args.random_terrain_overview and args.profile != 'random_terrain_20261112':
    parser.error('--random-terrain-overview requires --profile random_terrain_20261112')
if args.record:
    args.enable_cameras = True
    args.video = True
args.output.mkdir(parents=True, exist_ok=True)
summary = {'status': 'INITIALIZING', 'steps': 0, 'physics_steps': 0, 'resets': 0,
           'reset_causes': {}, 'max_closure_m': 0., 'spring_effort_error_N': 0.,
           'longest_above_2mm_s': 0., 'mode': args.mode, 'standstill': args.standstill,
           'fixed_command': args.fixed_command,
           'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
started = time.perf_counter()
app = None
video = None
physics_out = None
try:
    app = AppLauncher(args, fast_shutdown=True).app
    import torch
    import yaml
    import warp as wp
    from pxr import Usd, UsdPhysics
    from isaaclab.utils.math import quat_apply
    from isaaclab.utils.io import dump_yaml
    from agent_tasks.direct.wheelbipe.wheelbipe_own.experiment import WheelbipeOwnExperimentPlayCfg as WheelbipeOwnFlatPlayEnvCfg
    from agent_tasks.direct.wheelbipe.wheelbipe_own.experiment import WheelbipeOwnExperimentEnv as WheelbipeOwnEnv

    def tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, 'torch'):
            return value.torch
        if isinstance(value, wp.array):
            return wp.to_torch(value)
        return value

    try:
        sim_version = md.version('isaacsim')
    except md.PackageNotFoundError:
        sim_version = '5.1.0-binary'
    assert sim_version == '6.2.0.0', sim_version
    from isaacsim.core.version import get_version
    summary['sim_build'] = get_version()
    assert get_version()[0].startswith('6.2.0'), get_version()
    modern = True
    cfg = WheelbipeOwnFlatPlayEnvCfg()
    cfg.seed = args.seed
    cfg.height_range = [args.height, args.height]
    cfg.default_height_cmd = args.height
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device or 'cuda:0'
    if modern:
        from isaaclab_physx.physics import PhysxCfg
        assert isinstance(cfg.sim.physics, PhysxCfg)
        assert cfg.sim.physics.enable_external_forces_every_iteration == cfg.experiment_profile_settings["external_forces_every_iteration"]
    if args.standstill:
        cfg.commands.rel_standing_envs = 1.0
        cfg.commands.ranges.lin_vel_x = (0., 0.)
        cfg.commands.ranges.ang_vel_z = (0., 0.)
    if args.fixed_command:
        vx, yaw_rate = args.fixed_command
        cfg.commands.rel_standing_envs = 0.
        cfg.commands.ranges.lin_vel_x = (vx, vx)
        cfg.commands.ranges.ang_vel_z = (yaw_rate, yaw_rate)
    if args.record:
        cfg.height_range = [args.height, args.height]
        cfg.default_height_cmd = args.height
        cfg.scene.light.spawn.intensity = 800.0
        from isaaclab_visualizers.kit import KitVisualizerCfg
        if args.random_terrain_overview:
            cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, origin_type='world',
                eye=(0., -48., 42.), lookat=(0., 0., 0.),
                window_width=1280, window_height=720, enable_live_plots=False)]
        else:
            cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, origin_type='asset',
                origin_track_path='robot', eye=(-.9, -1.15, .55), lookat=(0., 0., 0.),
                window_width=1280, window_height=720, enable_live_plots=False)]
    if args.mixed_eval_terrain:
        from agent_tasks.direct.wheelbipe.wheelbipe_own import mixed_slopes
        # This is the same exact full-difficulty terrain used by the repeated
        # frozen-policy evaluation. It changes only terrain geometry; policy,
        # observation, action, physics and domain-randomization stay frozen.
        mixed_slopes.configure_evaluation(cfg)
    if args.random_terrain_overview:
        from agent_tasks.direct.wheelbipe.wheelbipe_own import random_terrain_map

        # One map copy lets the whole connected 3x3 terrain fit in one frame.
        random_terrain_map.configure_preview(cfg)
    from scripts.rsl_rl.checkpoint_utils import prepare_checkpoint
    prepare_checkpoint(cfg, None, str(args.checkpoint))
    summary['saved_observation_action_fields_match'] = True
    if args.light_slope_case:
        light_slope.configure(cfg, args.light_slope_case)
    dump_yaml(str(args.output / 'env.yaml'), cfg)
    summary['experiment_profile'] = args.profile
    summary['experiment_profile_settings'] = cfg.experiment_profile_settings
    from scripts.rsl_rl.checkpoint_utils import _plain
    saved_cfg = _plain(yaml.load((args.checkpoint.parent / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader))
    if (
        'experiment_profile_settings' in saved_cfg
        and not args.allow_profile_override
        and (
            saved_cfg['experiment_profile_name'] != args.profile
            or saved_cfg['experiment_profile_settings'] != cfg.experiment_profile_settings
        )
    ):
        raise RuntimeError('Evaluation profile differs from the saved training profile')
    saved_agent = yaml.safe_load((args.checkpoint.parent / 'params/agent.yaml').read_text())
    if saved_agent.get('empirical_normalization') or saved_agent['policy']['activation'] != 'elu':
        raise ValueError('This evaluator requires the original non-normalized ELU actor')
    saved_launch = json.loads((args.checkpoint.parent / 'params/launch.json').read_text())
    import agent_world.assets.wheelbipe_own as asset_module
    project = Path(asset_module.__file__).resolve().parents[4]
    for file in asset_module.ASSET_DIR.iterdir():
        if file.is_file():
            relative = str(file.relative_to(project))
            if saved_launch['own_asset_sha256'].get(relative) != hashlib.sha256(file.read_bytes()).hexdigest():
                raise ValueError('Asset differs from original checkpoint: ' + relative)
    saved = torch.load(args.checkpoint, map_location=cfg.sim.device, weights_only=False)
    state = {k.removeprefix('actor.'): v for k, v in saved['model_state_dict'].items() if k.startswith('actor.')}
    weights = [(int(k.split('.')[0]), v) for k, v in state.items() if k.endswith('.weight')]
    weights.sort()
    layers = []
    for i, (_, weight) in enumerate(weights):
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        if i != len(weights) - 1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to(cfg.sim.device).eval()
    actor.load_state_dict(state, strict=True)
    summary['actor_layers'] = [[int(v.shape[1]), int(v.shape[0])] for _, v in weights]
    summary['checkpoint_iteration'] = saved['iter']
    env = WheelbipeOwnEnv(cfg)
    native = env._native_robot
    # Freeze only evaluation tile assignment; keep the training geometry,
    # constraints, disturbances, sensor noise and termination rules intact.
    env._u_eval = True
    env._u_reset_mode_override = torch.zeros(env.num_envs, device=cfg.sim.device, dtype=torch.long)
    summary['initial_condition'] = 'stationary reset from the original constraint-preserving bank'
    if args.terrain_column is not None:
        generated_profile = (
            cfg.experiment_profile_settings.get('terrain_mix')
            or cfg.experiment_profile_settings.get('mixed_slopes')
        )
        assert generated_profile, 'Terrain selection requires a generated terrain profile'
        origins = env.terrain.terrain_origins
        assert 0 <= args.terrain_level < origins.shape[0]
        assert 0 <= args.terrain_column < origins.shape[1]
        def fixed_terrain(env_ids, move_up, move_down):
            env.terrain.terrain_types[env_ids] = args.terrain_column
            env.terrain.terrain_levels[env_ids] = args.terrain_level
            env.terrain.env_origins[env_ids] = origins[args.terrain_level, args.terrain_column]
        # Hold a declared existing terrain tile for visual comparison. Training
        # scene geometry and all physical/DR settings are retained.
        env.terrain.update_env_origins = fixed_terrain
        fixed_terrain(torch.arange(env.num_envs, device=cfg.sim.device), None, None)
        proportions = [v.proportion for v in cfg.terrain.terrain_generator.sub_terrains.values()]
        index = int(__import__('numpy').searchsorted(__import__('numpy').cumsum(proportions) / sum(proportions), (args.terrain_column + .5) / origins.shape[1]))
        terrain_name = list(cfg.terrain.terrain_generator.sub_terrains)[index]
        # The personal mixed-slope build does not include the separate unified
        # locomotion mixin, where this diagnostics-only family marker is defined.
        if hasattr(env, '_u_family'):
            env._u_family[:] = index
        summary['terrain_selection'] = dict(column=args.terrain_column, level=args.terrain_level,
            max_level=int(origins.shape[0])-1, family=terrain_name, fixed_for_video=True,
            origin=origins[args.terrain_level, args.terrain_column].tolist(),
            full_difficulty=args.mixed_eval_terrain,
            note=('Full-difficulty evaluation tile.' if args.mixed_eval_terrain
                  else 'Declared training tile selection, not training progress or full-terrain validation.'))
    else:
        summary['terrain_selection'] = dict(family=args.light_slope_case or 'flat', level=None)
    if args.light_slope_case:
        summary['light_slope'] = light_slope.prepare_start(env, args.light_slope_case)
    summary.update(sim_version=sim_version, python=__import__('sys').version,
                   physics_dt=cfg.sim.dt, policy_dt=env.step_dt,
                   bodies=len(native.body_names), dofs=len(native.joint_names),
                   position_iterations=cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count,
                   velocity_iterations=cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count)
    if modern:
        summary['physics_backend'] = getattr(native, '__backend_name__', type(native).__module__)
        assert 'physx' in summary['physics_backend'].lower()
    (args.output / 'runtime_indices.json').write_text(json.dumps(env.index_manifest(), indent=2))
    loops = []
    for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')):
        if prim.IsA(UsdPhysics.SphericalJoint):
            joint = UsdPhysics.Joint(prim)
            assert joint.GetJointEnabledAttr().Get() and joint.GetExcludeFromArticulationAttr().Get()
            targets = [joint.GetBody0Rel().GetTargets()[0], joint.GetBody1Rel().GetTargets()[0]]
            assert all(str(t).startswith('/World/envs/env_0/Robot/') for t in targets)
            loops.append(([native.body_names.index(t.name) for t in targets],
                          [list(joint.GetLocalPos0Attr().Get()), list(joint.GetLocalPos1Attr().Get())]))
    assert len(loops) == 6 and len(native.body_names) == 34 and len(native.joint_names) == 33
    summary['loops'] = len(loops)
    summary['external_forces_every_iteration'] = env.sim.stage.GetPrimAtPath('/physicsScene').GetAttribute(
        'physxScene:enableExternalForcesEveryIteration').Get()
    assert summary['external_forces_every_iteration'] == cfg.experiment_profile_settings['external_forces_every_iteration']
    ids = torch.tensor([j[0] for j in loops], device=cfg.sim.device)
    anchors = torch.tensor([j[1] for j in loops], device=cfg.sim.device)
    # Fixed command changes are video inputs; the saved controller/physics are retained.
    commanded = torch.tensor([[.8, 0., 0.]], device=cfg.sim.device)
    cg = env.command_generator
    def fixed():
        if args.sequence:
            cg.command.copy_(commanded)
            if hasattr(cg, 'is_standing_env'): cg.is_standing_env[:] = False
        env.height_cmd.fill_(args.height)
    original_compute, original_resample = cg.compute, cg._resample_command
    def compute(dt):
        original_compute(dt)
        fixed()
    def resample(ids):
        original_resample(ids)
        fixed()
    cg.compute, cg._resample_command = compute, resample
    obs, _ = env.reset()
    fixed()
    if args.random_terrain_overview:
        origin = tensor(env.terrain.env_origins)[0].cpu().numpy()
        summary['overview'] = {
            'environment_origin_m': origin.tolist(),
            'camera_eye_m': (origin + (-34.0, -42.0, 38.0)).tolist(),
            'camera_target_m': (origin + (0.0, 0.0, 0.0)).tolist(),
        }
        env.sim.set_camera_view(
            origin + (-34.0, -42.0, 38.0),
            origin,
        )
    if args.light_slope_case:
        light_slope.validate_reset_state(env, summary['light_slope'])
    if args.sequence:
        env.command.copy_(commanded)
        obs = env._get_observations()
    summary['command_sequence'] = ([{'start_s':0.,'end_s':5.,'vx':.8,'yaw':0.},
        {'start_s':5.,'end_s':10.,'vx':.6,'yaw':.4},
        {'start_s':10.,'end_s':15.,'vx':-.6,'yaw':0.}] if args.sequence else None)
    summary['domain_randomization_enabled'] = bool(env._domain_randomization_health()['enabled'])
    summary['domain_randomization_initial'] = env._domain_randomization_health()
    summary['comparison_controls'] = {
        'profile': args.profile,
        'seed': args.seed,
        'checkpoint_sha256': summary['checkpoint_sha256'],
        'fixed_command': args.fixed_command,
        'steps': args.steps,
        'policy_mode': 'deterministic actor mean',
        'domain_randomization': 'enabled',
        'profile_override_allowed': args.allow_profile_override,
        'pairing_note': (
            'Same seed, profile, checkpoint and command are used. DR remains enabled, '
            'so terrain-dependent random draws evolve with each trajectory; runs are '
            'comparable but are not a strict paired experiment.'
        ),
    }
    summary['training_modified'] = False
    assert obs['policy'].shape == (1, 35) and obs['critic'].shape == (1, 78)
    summary['observation_shapes'] = {k: list(v.shape) for k, v in obs.items()}
    (args.output / 'initial_obs.json').write_text(json.dumps({k: v.tolist() for k, v in obs.items()}))
    physics_out = (args.output / 'physics.jsonl').open('w', buffering=1)
    (args.output / 'diagnostic_manifest.json').write_text(json.dumps({
        'native_joint_names': native.joint_names,
        'contact_body_names': env.contact_sensor.body_names,
        'leg_indices': env._legs_act_idx, 'wheel_indices': env._wheel_idx,
        'spring_indices': env._spring_idx,
        'leg_action_scale': cfg.leg_action_scale,
        'use_action_low_pass_filter': env.use_action_low_pass_filter,
        'reward_weights': dict(cfg.rewards),
        'policy_dt': env.step_dt, 'physics_dt': cfg.sim.dt,
        'policy_kind': 'deterministic actor mean, torch.inference_mode',
    }, indent=2))
    original_step, original_update = env.sim.step, env.scene.update
    armed = False
    above_run = 0

    def step_hook(*a, **kw):
        global armed
        result = original_step(*a, **kw)
        armed = True
        return result

    def update_hook(dt):
        global armed, above_run
        original_update(dt)
        if not armed:
            return
        armed = False
        pos, quat = tensor(native.data.body_pos_w)[0, ids], tensor(native.data.body_quat_w)[0, ids]
        points = pos + quat_apply(quat, anchors)
        residual = torch.linalg.vector_norm(points[:, 0] - points[:, 1], dim=-1)
        assert torch.isfinite(residual).all()
        peak = float(residual.max())
        summary['max_closure_m'] = max(summary['max_closure_m'], peak)
        above_run = above_run + 1 if peak > .002 else 0
        summary['longest_above_2mm_s'] = max(summary['longest_above_2mm_s'], above_run * dt)
        error = float((tensor(native.data.applied_torque)[:, env._spring_idx] - env._spring_commanded_net).abs().max())
        summary['spring_effort_error_N'] = max(summary['spring_effort_error_N'], error)
        assert torch.isfinite(tensor(native.data.root_state_w)).all()
        summary['physics_steps'] += 1
        physics_out.write(json.dumps({
            'substep': summary['physics_steps'],
            'q': tensor(native.data.joint_pos)[0].tolist(),
            'dq': tensor(native.data.joint_vel)[0].tolist(),
            'tau': tensor(native.data.applied_torque)[0].tolist(),
            'computed_tau': tensor(native.data.computed_torque)[0].tolist(),
            'q_target': tensor(native.data.joint_pos_target)[0].tolist(),
            'dq_target': tensor(native.data.joint_vel_target)[0].tolist(),
            'root_state': tensor(native.data.root_state_w)[0].tolist(),
            'velocity_b': tensor(native.data.root_lin_vel_b)[0].tolist(),
            'ang_vel_b': tensor(native.data.root_ang_vel_b)[0].tolist(),
            'spring_command': env._spring_commanded_net[0].tolist(),
            'contact_forces_w': tensor(env.contact_sensor.data.net_forces_w)[0].tolist(),
            'closure_m': residual.tolist(),
            'height_command': float(env._get_effective_height_cmd()[0]),
            **({'raw_q': tensor(native.root_physx_view.get_dof_positions())[0].tolist(), 'raw_dq': tensor(native.root_physx_view.get_dof_velocities())[0].tolist()} if summary['physics_steps'] % 200 == 0 else {}),
        }) + '\n')


    env.sim.step, env.scene.update = step_hook, update_hook
    video = None
    if args.record:
        import imageio.v2 as imageio
        viz = next(v for v in env.sim.visualizers if v.cfg.visualizer_type == 'kit')
        from pxr import UsdGeom
        camera = UsdGeom.Camera(env.sim.stage.GetPrimAtPath('/OmniverseKit_Persp'))
        camera.GetFocalLengthAttr().Set(32.0)
        camera.GetClippingRangeAttr().Set((0.01, 1000.))
        for _ in range(12):
            env.sim.render()
            frame = viz.render_rgb_array()
        assert frame.shape == (720, 1280, 3) and frame.std() > 2.0, 'Blank renderer output'
        imageio.imwrite(args.output / 'warmup.png', frame)
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
        small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        video = imageio.get_writer(str(args.output / 'policy.mp4'), fps=50, codec='libx264',
            quality=8, macro_block_size=1, output_params=['-movflags','+faststart'])
    errors_vx, errors_yaw, abs_vx, heights = [], [], [], []
    with torch.inference_mode(), (args.output / 'rollout.jsonl').open('w', buffering=1) as out:
        for step in range(args.steps):
            if args.sequence:
                desired = [.8, 0., 0.] if step < 250 else ([.6, 0., .4] if step < 500 else [-.6, 0., 0.])
                new_command = torch.tensor([desired], device=cfg.sim.device)
                changed = not torch.equal(commanded, new_command)
                commanded.copy_(new_command)
                fixed()
                env.command.copy_(commanded)
                if changed: obs = env._get_observations()
            command = env.command_generator.command[0].clone()
            velocity = tensor(native.data.root_lin_vel_b)[0].clone()
            yaw = float(tensor(native.data.root_ang_vel_b)[0, 2])
            action = actor(obs['policy']) if args.mode == 'policy' else torch.zeros((1, 6), device=cfg.sim.device)
            assert torch.isfinite(action).all()
            pre_state = tensor(native.data.root_state_w)[0].tolist()
            if args.mode == 'continuous':
                env._pre_physics_step(action)
                for _ in range(cfg.decimation):
                    env._apply_action()
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(cfg.sim.dt)
                done = torch.zeros(1, dtype=torch.bool, device=cfg.sim.device)
                timeout = done
                reward = torch.zeros(1, device=cfg.sim.device)
                causes = {}
            else:
                obs, reward, done, timeout, _ = env.step(action)
                assert all(torch.isfinite(value).all() for value in obs.values()) and torch.isfinite(reward).all()
                causes = {key: bool(mask[0]) for key, mask in env._termination_reason_masks.items()}
            reset = bool(done[0] or timeout[0])
            summary['resets'] += int(reset)
            for key, value in causes.items():
                summary['reset_causes'][key] = summary['reset_causes'].get(key, 0) + int(reset and value)
            errors_vx.append(abs(float(velocity[0] - command[0])))
            errors_yaw.append(abs(yaw - float(command[2])))
            abs_vx.append(abs(float(velocity[0])))
            heights.append(pre_state[2])
            out.write(json.dumps({'step': step, 'command': command.tolist(), 'velocity_b': velocity.tolist(),
                                  'yaw_rate': yaw, 'root_state': pre_state, 'actions': action[0].tolist(),
                                  'reward': float(reward[0]), 'reset': reset, 'causes': causes, 'reward_terms': {k: float(v[0]) for k, v in env._last_reward_terms.items()}}) + '\n')
            summary['steps'] = step + 1
            if video is not None:
                env.sim.render()
                frame = viz.render_rgb_array()
                assert frame.shape == (720, 1280, 3) and frame.std() > 2.0
                annotated = Image.fromarray(frame)
                draw = ImageDraw.Draw(annotated, 'RGBA')
                draw.rounded_rectangle((12,12,1268,106), radius=8, fill=(10,18,28,205))
                title = 'Frozen policy' if args.light_slope_case else 'Unified policy'
                draw.text((25,20), f'{title} | {args.profile} | cp {saved["iter"]} | DR ON | {summary["terrain_selection"]["family"]}',
                    font=title_font, fill=(240,247,255,255))
                draw.text((25,53), f't={(step+1)*env.step_dt:05.2f}s  |  command vx={float(command[0]):+.2f} m/s, yaw={float(command[2]):+.2f} rad/s',
                    font=small_font, fill=(190,221,248,255))
                draw.text((25,78), f'actual vx={float(velocity[0]):+.2f} m/s, yaw={yaw:+.2f} rad/s  |  resets={summary["resets"]}  |  simulation replay',
                    font=small_font, fill=(190,221,248,255))
                video.append_data(np.asarray(annotated))
                if step in (0, args.steps // 2, args.steps - 1):
                    imageio.imwrite(args.output / f'frame_{step:04d}.png', frame)
            if (step + 1) % 100 == 0:
                print(json.dumps({'steps': step + 1, 'closure_mm': summary['max_closure_m'] * 1000,
                                  'resets': summary['resets']}), flush=True)
    if video is not None:
        video.close()
        video = None
    summary['domain_randomization_final'] = env._domain_randomization_health()
    assert summary['physics_steps'] == args.steps * cfg.decimation, 'Unexpected physics-step count'
    summary.update(mean_abs_vx_error_m_s=sum(errors_vx) / len(errors_vx),
                   mean_abs_yaw_error_rad_s=sum(errors_yaw) / len(errors_yaw),
                   mean_abs_vx_m_s=sum(abs_vx) / len(abs_vx),
                   base_height_range_m=[min(heights), max(heights)],
                   sim_seconds=summary['steps'] * env.step_dt)
    if args.light_slope_case:
        light_slope.add_metrics(env, args.output / 'rollout.jsonl', summary['light_slope'])
    mechanics_ok = (summary['longest_above_2mm_s'] <= .1
                    and summary['max_closure_m'] < .005
                    and summary['spring_effort_error_N'] <= .1
                    and not any(summary['reset_causes'].values()))
    light_slope_ok = (not args.light_slope_case
                      or summary['light_slope']['status'] == 'PASS')
    summary['status'] = 'PASS' if mechanics_ok and light_slope_ok else 'FAIL'
    physics_out.close()
    env.close()
except BaseException:
    summary['status'] = 'ERROR'
    summary['error'] = traceback.format_exc()
    raise
finally:
    if video is not None: video.close()
    if physics_out is not None and not physics_out.closed: physics_out.close()
    summary['wall_seconds'] = time.perf_counter() - started
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if app is not None:
        import inspect
        if 'exit_code' in inspect.signature(app.close).parameters:
            app.close(exit_code=0 if summary['status'] == 'PASS' else 1)
        else:
            app.close()
