# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import inspect
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--finetune_noise_std", type=float, default=None, help="Explicitly reset exploration std for a new fine-tuning experiment; optimizer is fresh.")
parser.add_argument("--actor_only_finetune", action="store_true", help="Transfer actor only; reset critic, exploration and Adam for a new task distribution.")
parser.add_argument("--audit_updates", action="store_true", help="Record real PPO losses, gradients and parameter changes.")
parser.add_argument('--allow_partial_checkpoint', action='store_true', help='Explicitly allow partial parameter transfer; this is not a compatible policy load.')
parser.add_argument(
    "--resume_training",
    action="store_true",
    default=False,
    help="Resume optimizer state and iteration counter from --checkpoint, then continue from the next iteration.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.resume or args_cli.load_run:
    parser.error('--resume/--load_run are not implemented. Use --checkpoint PATH --resume_training.')
if args_cli.finetune_noise_std is not None and (args_cli.resume_training or not args_cli.checkpoint or args_cli.finetune_noise_std <= 0):
    parser.error("--finetune_noise_std requires positive std, --checkpoint and a fresh fine-tuning optimizer")
if args_cli.resume_training and not args_cli.checkpoint:
    parser.error('--resume_training requires --checkpoint')

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
if args_cli.task and "-Own-" in args_cli.task:
    from scripts.own_robot.preflight620 import require_target
    require_target()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import warnings
import logging
# 抑制quat_rotate弃用警告，避免卡顿
# 这个警告来自isaaclab.utils.math，通过warnings和logging都可能输出
warnings.filterwarnings("ignore", message=".*quat_rotate.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="isaaclab.utils.math")
# 设置math模块的logging级别为ERROR，抑制WARNING
logging.getLogger("isaaclab.utils.math").setLevel(logging.ERROR)
# 添加自定义过滤器，过滤包含quat_rotate的警告消息
class QuatRotateFilter(logging.Filter):
    def filter(self, record):
        return "quat_rotate" not in str(record.getMessage())
logging.getLogger().addFilter(QuatRotateFilter())

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from agent_rl.rsl_rl.legacy_cfg import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import resolve_task_config

from rsl_rl.runners import *  # noqa: F401
from agent_rl.rsl_rl.runners import *  # noqa: F401
import agent_world  # noqa: F401
import agent_tasks  # noqa: F401
from checkpoint_utils import prepare_checkpoint, write_launch_manifest, verify_resume

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# 清理任务名称，去除可能的前导 '=' 符号
if args_cli.task and args_cli.task.startswith('='):
    args_cli.task = args_cli.task[1:]
    print(f"[INFO] 已清理任务名称，去除前导 '=' 符号: {args_cli.task}")


def _load_policy_for_finetune(runner, checkpoint_path: str) -> None:
    """Load matching policy tensors only, without optimizer or iteration state."""
    loaded_dict = torch.load(checkpoint_path, weights_only=False, map_location=runner.device)
    source_state = loaded_dict["model_state_dict"]
    target_state = runner.alg.policy.state_dict()
    mismatches = [key for key,value in target_state.items() if key not in source_state or source_state[key].shape != value.shape]
    if (mismatches or source_state.keys() != target_state.keys()) and not args_cli.allow_partial_checkpoint:
        raise ValueError('Incompatible checkpoint tensors: '+str(mismatches)+'. Use --checkpoint_architecture for saved PPO widths; partial transfer requires --allow_partial_checkpoint.')
    compatible_state = {}
    expanded_state = {}
    skipped = []
    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is not None and tuple(target_value.shape) == tuple(value.shape):
            compatible_state[key] = value
        elif (
            target_value is not None
            and value.ndim == 2
            and target_value.ndim == 2
            and target_value.shape[0] == value.shape[0]
            and target_value.shape[1] >= value.shape[1]
        ):
            expanded_value = target_value.clone()
            expanded_value[:, : value.shape[1]] = value
            expanded_state[key] = expanded_value
        else:
            skipped.append((key, tuple(value.shape), tuple(target_value.shape) if target_value is not None else None))
    if args_cli.actor_only_finetune:
        compatible_state = {k:v for k,v in compatible_state.items() if k.startswith('actor.')}
        expanded_state = {}
        print('[FINETUNE] Actor only: critic and optimizer freshly initialized')
    target_state.update(compatible_state)
    target_state.update(expanded_state)
    runner.alg.policy.load_state_dict(target_state, strict=True)
    if args_cli.actor_only_finetune:
        current = runner.alg.policy.state_dict()
        actor_exact = all(torch.equal(current[k], source_state[k]) for k in current if k.startswith('actor.'))
        critic_fresh = any(not torch.equal(current[k], source_state[k]) for k in current if k.startswith('critic.'))
        assert actor_exact and critic_fresh and not runner.alg.optimizer.state
        runner._actor_transfer_audit = dict(actor_exact_match=True, critic_reinitialized=True,
            optimizer_state_entries=0, iteration_restart=0, source_checkpoint=str(checkpoint_path))
    if getattr(runner, 'empirical_normalization', False):
        runner.obs_normalizer.load_state_dict(loaded_dict['obs_norm_state_dict'])
        runner.privileged_obs_normalizer.load_state_dict(loaded_dict['privileged_obs_norm_state_dict'])
    runner.current_learning_iteration = 0
    print(
        f"[INFO]: Finetune checkpoint load: loaded {len(compatible_state)} policy tensors, "
        f"expanded {len(expanded_state)} input tensors, "
        f"skipped {len(skipped)} incompatible tensors."
    )
    for key, value in expanded_state.items():
        source_shape = tuple(source_state[key].shape)
        target_shape = tuple(value.shape)
        print(f"[INFO]:   expanded {key}: checkpoint{source_shape} -> model{target_shape}")
    if skipped:
        for key, source_shape, target_shape in skipped[:20]:
            print(f"[INFO]:   skipped {key}: checkpoint{source_shape} -> model{target_shape}")
        if len(skipped) > 20:
            print(f"[INFO]:   ... {len(skipped) - 20} more skipped tensors")


def _notify_env_training_progress(env, iteration: int) -> None:
    """Seed env-side step curricula with the runner iteration before learn()."""

    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "set_training_progress"):
        unwrapped.set_training_progress(iteration=iteration)


def _attach_checkpoint_metadata(env_cfg, agent_cfg, checkpoint_path: str | None) -> None:
    """Record CLI checkpoint provenance in dumped training YAML files."""

    if checkpoint_path is None:
        return

    checkpoint_path = os.path.abspath(checkpoint_path)
    checkpoint_metadata = {
        "path": checkpoint_path,
        "dir": os.path.dirname(checkpoint_path),
        "file": os.path.basename(checkpoint_path),
        "resume_training": bool(args_cli.resume_training),
    }
    setattr(env_cfg, "launch_checkpoint", checkpoint_metadata)
    setattr(agent_cfg, "launch_checkpoint", checkpoint_metadata)


def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    prepare_checkpoint(env_cfg, agent_cfg, args_cli.checkpoint, args_cli.checkpoint_architecture, args_cli.checkpoint_observations)
    if args_cli.resume_training and hasattr(env_cfg, 'experiment_profile_settings'):
        import yaml
        from checkpoint_utils import _plain
        from pathlib import Path
        previous = Path(args_cli.checkpoint).parent / 'params'
        saved_env = _plain(yaml.load((previous / 'env.yaml').read_text(), Loader=yaml.BaseLoader))
        saved_agent = yaml.safe_load((previous / 'agent.yaml').read_text())
        if saved_env.get('experiment_profile_settings') != env_cfg.experiment_profile_settings:
            raise ValueError('Exact continuation requires the saved experiment profile; use explicit fresh fine-tuning for changed rewards/physics.')
        current_alg = agent_cfg.algorithm.to_dict()
        keys = ('schedule', 'gamma', 'lam', 'entropy_coef', 'clip_param', 'value_loss_coef',
                'max_grad_norm', 'desired_kl', 'num_learning_epochs', 'num_mini_batches',
                'use_clipped_value_loss', 'normalize_advantage_per_mini_batch')
        differences = {k: (saved_agent['algorithm'].get(k), current_alg.get(k))
                       for k in keys if saved_agent['algorithm'].get(k) != current_alg.get(k)}
        if differences:
            raise ValueError('Continuation PPO configuration mismatch: ' + str(differences))
        print('[RESUME CONFIG] Saved physics/reward profile and PPO update settings match')
        old_dt = float(saved_env['sim']['dt'])
        new_dt = float(env_cfg.sim.dt)
        policy_dt_before = old_dt * int(saved_env['decimation'])
        policy_dt_after = new_dt * int(env_cfg.decimation)
        if abs(policy_dt_before - policy_dt_after) > 1e-9 or new_dt > old_dt + 1e-9:
            raise ValueError('Continuation cannot change policy period or coarsen numerical integration')
        numerical_integration_transition = dict(previous_physics_dt_s=old_dt,
            current_physics_dt_s=new_dt, policy_dt_s=policy_dt_after,
            refinement_explicit=int(os.environ.get('WHEELBIPE_INTEGRATION_SUBSTEPS', '1')),
            model_and_optimizer_restored=True)
        print('[NUMERICAL INTEGRATION] ' + str(numerical_integration_transition))
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    runner_class = eval(getattr(agent_cfg, "runner_class", "OnPolicyRunner"))
    # wrap around environment for rsl-rl
    if runner_class is OnPolicyRunner:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    else:
        from agent_rl.rsl_rl.env import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # Save mastery beside each checkpoint; restarting PPO must not restart its
    # terrain curriculum or silently lose the sampling coverage history.
    original_save = runner.save
    def save_with_curriculum(path, *args, **kwargs):
        result = original_save(path, *args, **kwargs)
        scene = env.unwrapped
        if getattr(scene, '_u_ready', False):
            payload = {name: getattr(scene, name).detach().cpu() for name in
                       ('_u_mastery', '_u_good', '_u_command_hist', '_u_velocity_hist', '_u_reset_modes', '_u_grade_counts', '_u_obs_min', '_u_obs_max')}
            payload.update(_u_causes=scene._u_causes, _u_cmd_min=scene._u_cmd_min, _u_cmd_max=scene._u_cmd_max, _u_transitions=scene._u_transitions)
            torch.save(payload, str(path) + '.curriculum.pt')
        return result
    runner.save = save_with_curriculum
    # load the checkpoint
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if args_cli.resume_training:
            # True continuation: restore optimizer/iteration and continue from the next iteration index.
            load_signature = inspect.signature(runner.load)
            if "load_iteration" in load_signature.parameters:
                runner.load(resume_path, load_optimizer=True, load_iteration=True)
            else:
                runner.load(resume_path, load_optimizer=True)
            resume_verification = verify_resume(runner, resume_path)
            runner.current_learning_iteration += 1
            import json
            os.makedirs(os.path.join(log_dir, 'params'), exist_ok=True)
            with open(os.path.join(log_dir, 'params', 'resume_verification.json'), 'w') as f:
                json.dump(resume_verification, f, indent=2)
            if hasattr(env_cfg, 'experiment_profile_settings'):
                with open(os.path.join(log_dir, 'params', 'numerical_integration_transition.json'), 'w') as f:
                    json.dump(numerical_integration_transition, f, indent=2)
            if hasattr(runner.alg, 'learning_rate'):
                runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']
            print(
                f"[INFO]: Resuming training from next iteration: {runner.current_learning_iteration}"
            )
        else:
            # Weight initialization / finetune-from-checkpoint workflow.
            _load_policy_for_finetune(runner, resume_path)
            if args_cli.finetune_noise_std is not None:
                import math
                with torch.no_grad():
                    if hasattr(runner.alg.policy, 'std'):
                        runner.alg.policy.std.fill_(args_cli.finetune_noise_std)
                    elif hasattr(runner.alg.policy, 'log_std'):
                        runner.alg.policy.log_std.fill_(math.log(args_cli.finetune_noise_std))
                    else:
                        raise RuntimeError('Unknown policy noise parameter')
                agent_cfg.policy.init_noise_std = args_cli.finetune_noise_std
                print(f'[FINETUNE] Fresh Adam, iteration 0; exploration std explicitly set to {args_cli.finetune_noise_std}')


    if args_cli.resume_training and getattr(env.unwrapped, '_u_ready', False):
        from pathlib import Path
        sidecar = Path(str(args_cli.checkpoint) + '.curriculum.pt')
        if not sidecar.exists():
            raise ValueError('Unified continuation requires its curriculum sidecar: ' + str(sidecar))
        payload = torch.load(sidecar, map_location=env.unwrapped.device, weights_only=False)
        for name, value in payload.items():
            setattr(env.unwrapped, name, value)
        print('[RESUME] Unified terrain mastery and cumulative coverage restored')
    _notify_env_training_progress(env, int(getattr(runner, "current_learning_iteration", 0)))
    _attach_checkpoint_metadata(env_cfg, agent_cfg, args_cli.checkpoint)
    if hasattr(runner, '_actor_transfer_audit'):
        import json
        os.makedirs(os.path.join(log_dir, 'params'), exist_ok=True)
        with open(os.path.join(log_dir, 'params', 'actor_transfer.json'), 'w') as stream:
            json.dump(runner._actor_transfer_audit, stream, indent=2)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    write_launch_manifest(os.path.join(log_dir,'params'),args_cli.task,args_cli.checkpoint,
                          'resume' if args_cli.resume_training else ('finetune' if args_cli.checkpoint else 'scratch'))

    if args_cli.audit_updates:
        from training_audit import attach_training_audit
        attach_training_audit(runner, log_dir)

    # run training
    if os.environ.get('WHEELBIPE_AUTOMATIC_RECOVERY') == '1':
        from training_recovery import learn_with_recovery
        learn_with_recovery(runner, agent_cfg.max_iterations, log_dir)
    else:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main(*resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point", overrides=hydra_args))
    # close sim app
    simulation_app.close()
