"""Bound PPO learning rate and checkpoint recoverable physics interruptions."""
import json
import math
import os
from pathlib import Path
import time


def learn_with_recovery(runner, iterations, log_dir):
    import torch
    root = Path(log_dir)
    cap = float(os.environ.get('WHEELBIPE_LEARNING_RATE_CAP', '0.0003'))
    last_good = int(runner.current_learning_iteration) - 1
    updating = False
    original_update = runner.alg.update

    def clamp_lr(optimizer, args, kwargs):
        rate = min(float(runner.alg.learning_rate), cap)
        runner.alg.learning_rate = rate
        for group in optimizer.param_groups:
            group['lr'] = min(float(group['lr']), cap)

    # PPO changes adaptive LR inside its minibatch loop. Clamp immediately before
    # every actual optimizer step, rather than only once per rollout.
    hook = runner.alg.optimizer.register_step_pre_hook(clamp_lr)
    (root / 'params/recovery_policy.json').write_text(json.dumps(dict(
        learning_rate_cap=cap, starting_iteration=runner.current_learning_iteration,
        model_optimizer_restored_before_intervention=not hasattr(runner, '_actor_transfer_audit'),
        recoverable_physics='checkpoint last completed PPO update, discard partial rollout, restart simulator'), indent=2) + '\n')

    def update():
        nonlocal last_good, updating
        updating = True
        losses = original_update()
        if not all(math.isfinite(float(value)) for value in losses.values()):
            raise FloatingPointError('RECOVERABLE_PHYSICS: nonfinite PPO losses; roll back saved checkpoint')
        if not all(bool(torch.isfinite(value).all()) for value in runner.alg.policy.state_dict().values()):
            raise FloatingPointError('RECOVERABLE_PHYSICS: nonfinite PPO parameters; roll back saved checkpoint')
        last_good += 1
        updating = False
        return losses

    runner.alg.update = update
    try:
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    except Exception as exc:
        message = str(exc)
        recoverable = any(token in message.lower() for token in
            ('recoverable_physics', 'closed-loop', 'closed loop', 'nonfinite', 'observation mismatch'))
        if recoverable:
            checkpoint = None
            save_error = None
            if not updating and last_good >= 0:
                try:
                    runner.current_learning_iteration = last_good
                    path = root / f'model_{last_good}.pt'
                    runner.save(str(path))
                    checkpoint = str(path)
                except Exception as save_exc:
                    save_error = repr(save_exc)
            request = dict(unix=time.time(), error=message, checkpoint=checkpoint,
                last_completed_iteration=last_good, partial_rollout_discarded=True,
                optimizer_update_interrupted=updating, checkpoint_error=save_error)
            (root / 'recovery_request.json').write_text(json.dumps(request, indent=2) + '\n')
            print('[AUTOMATIC RECOVERY] ' + json.dumps(request), flush=True)
        raise
    finally:
        hook.remove()
