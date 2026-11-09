"""Persistent single-policy pilot, exact PPO continuation, and balanced evaluation."""
import argparse
import fcntl
import fnmatch
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--resume', action='store_true')
p.add_argument('--checkpoint', type=Path, help='Explicit checkpoint for a reviewed rollback')
a = p.parse_args()
r = a.root.resolve()
plan = json.loads((r / 'plan.json').read_text())
training = r / 'training'
training.mkdir(exist_ok=True)
lock = (training / 'supervisor.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state = dict(profile='unified_locomotion', root=str(r), supervisor_pid=os.getpid(),
    num_envs=plan['num_envs'], shared_policy_count=1, total_updates=plan['total_updates'])


def status(stage, **changes):
    state.update(stage=stage, updated_unix=time.time(), **changes)
    tmp = training / 'status.tmp'
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(training / 'status.json')
    if plan.get('execution_state') != stage:
        plan['execution_state'] = stage
        plan_tmp = r / 'plan.tmp'
        plan_tmp.write_text(json.dumps(plan, indent=2) + '\n')
        plan_tmp.replace(r / 'plan.json')
    print(json.dumps(state), flush=True)


def environment(phase, evaluation=False):
    v = os.environ.copy()
    v.update(WHEELBIPE_EXPERIMENT='unified_locomotion', WHEELBIPE_STAGE='3',
        WHEELBIPE_PHASE=phase, WHEELBIPE_INTEGRATION_SUBSTEPS='2',
        WHEELBIPE_GPU=str(plan['validation_gpu'] if evaluation else plan['gpu']),
        WHEELBIPE_CPUSET='16-23' if evaluation else plan['cpus'])
    if plan.get('automatic_recovery'):
        v.update(WHEELBIPE_AUTOMATIC_RECOVERY='1',
            WHEELBIPE_LEARNING_RATE_CAP=str(plan.get('learning_rate_cap', .0003)))
    return v


def launch(entry, arguments):
    return ['bash', str(r / 'tools/run.sh'), str(entry), *map(str, arguments)]


def checkpoints():
    paths = (r / 'logs/rsl_rl' / plan['experiment']).glob('*/model_*.pt')
    suffixes = plan.get('checkpoint_run_suffixes', ['*'])
    paths = [p for p in paths if Path(str(p) + '.curriculum.pt').exists()
             and p.parent.name not in plan.get('ignored_checkpoint_runs', [])
             and any(fnmatch.fnmatchcase(p.parent.name, '*' + suffix) for suffix in suffixes)]
    if plan.get('recovery_start_checkpoint'):
        paths.append(Path(plan['recovery_start_checkpoint']))
    return sorted(paths,
                  key=lambda p: (int(p.stem.split('_')[-1]), p.stat().st_mtime))


def train(updates, phase, checkpoint, resume):
    argv = ['--task', 'Robotics-Wheelbipe-Own-Explore-v0', '--num_envs', plan['num_envs'],
        '--max_iterations', updates, '--device', 'cuda:0', '--viz', 'none',
        '--seed', plan['seed'], '--run_name', 'unified_' + plan.get('run_tag', '') + phase,
        '--checkpoint', checkpoint, '--checkpoint_architecture', '--audit_updates']
    if resume:
        argv += ['--resume_training']
    else:
        argv += ['--actor_only_finetune', '--finetune_noise_std', '.35']
    log = (training / (phase + '.log')).open('w')
    process = subprocess.Popen(launch(r / 'project/scripts/rsl_rl/train.py', argv),
        env=environment(phase), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, log


def evaluate(checkpoint, label, level=1, seed=71903):
    label = plan.get('run_tag', '') + label
    output = r / 'evaluation' / label
    output.mkdir(parents=True, exist_ok=True)
    log = (output / 'evaluate.log').open('w')
    argv = ['--root', r, '--checkpoint', checkpoint, '--output', output,
        '--level', level, '--seed', seed, '--seconds', '8', '--device', 'cuda:0', '--viz', 'none']
    process = subprocess.Popen(launch(r / 'tools/validate.py', argv), env=environment('evaluation_' + label, True),
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, log, output, checkpoint


def score_evaluation(job):
    process, log, output, checkpoint = job
    log.close()
    summary_path = output / 'summary.json'
    if process.returncode or not summary_path.exists():
        raise RuntimeError('Evaluation failed: ' + str(output))
    result = json.loads(summary_path.read_text())
    if result.get('status') != 'PASS' or not result.get('physics_pass'):
        raise RuntimeError('Evaluation physics checks failed: ' + str(output))
    families = {}
    for row in result['rows']:
        families.setdefault(row['family'], []).append(row)
    scores = {key: sum(x['vx_mae'] + .5 * x['failures'] / result['seconds'] for x in rows) / len(rows)
              for key, rows in families.items()}
    score = sum(scores.values()) / len(scores) + .2 * max(scores.values())
    record = dict(score=score, per_family_scores=scores, checkpoint=str(checkpoint),
        evaluation=str(output), selection_distribution='five families x nine signed speed commands',
        policy_convergence_claim=False, deployment_candidate_only=True)
    (output / 'selection_metrics.json').write_text(json.dumps(record, indent=2) + '\n')
    candidate = r / 'candidates'
    candidate.mkdir(exist_ok=True)
    previous = json.loads((candidate / 'selection.json').read_text()) if (candidate / 'selection.json').exists() else {'score': float('inf')}
    eligible = result['level'] == 1 and result['seed'] == 71903
    if eligible and score < previous['score']:
        shutil.copyfile(checkpoint, candidate / 'candidate.pt')
        shutil.copytree(checkpoint.parent / 'params', candidate / 'params', dirs_exist_ok=True)
        (candidate / 'selection.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def finish_evaluation(job):
    """An unusable evaluator result disqualifies a candidate, not the trainer."""
    if not plan.get('automatic_recovery'):
        return score_evaluation(job)
    try:
        return score_evaluation(job)
    except Exception as exc:
        process, log, output, checkpoint = job
        if not log.closed:
            log.close()
        summary = output / 'summary.json'
        detail = None
        if summary.exists():
            try:
                detail = json.loads(summary.read_text())
            except (ValueError, OSError):
                pass
        record = dict(status='EVALUATION_WARNING', checkpoint=str(checkpoint),
            evaluation=str(output), error=str(exc), returncode=process.returncode,
            physics_pass=False, candidate_eligible=False,
            training_continues=True, unix=time.time(),
            evaluator_status=detail.get('status') if detail else None)
        (output / 'evaluation_warning.json').write_text(json.dumps(record, indent=2) + '\n')
        with (training / 'evaluation_warnings.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        state['evaluation_warning_count'] = state.get('evaluation_warning_count', 0) + 1
        print('[EVALUATION WARNING] ' + json.dumps(record), flush=True)
        return record


child = None
job = None
try:
    if plan.get('require_preflight_on_resume'):
        preflight = json.loads((r / plan.get('preflight_report', 'reports/preflight/summary.json')).read_text())
        if preflight.get('status') != 'PASS' or not preflight.get('physics_pass'):
            raise RuntimeError('Reset/physics preflight must pass before continuation')
    if not a.resume:
        preflight = json.loads((r / 'reports/preflight/summary.json').read_text())
        if preflight['status'] != 'PASS':
            raise RuntimeError('Run and pass reset/physics preflight before starting PPO')
        if checkpoints():
            raise RuntimeError('Existing unified checkpoints: use --resume')
        status('pilot_training', initialization=plan['initialization'])
        child, log = train(plan['pilot_updates'], 'pilot', Path(plan['initialization']['checkpoint']), False)
        status('pilot_training', training_pid=child.pid)
        if child.wait():
            raise RuntimeError('PPO pilot failed; inspect training/pilot.log')
        log.close()
        checkpoint = checkpoints()[-1]
        assert int(checkpoint.stem.split('_')[-1]) == plan['pilot_updates'] - 1
        status('pilot_evaluation', latest_checkpoint=str(checkpoint), latest_saved_iteration=plan['pilot_updates'] - 1)
        job = evaluate(checkpoint, 'pilot')
        if job[0].wait():
            raise RuntimeError('Pilot evaluation failed')
        result = finish_evaluation(job)
        job = None
        status('pilot_passed', pilot_evaluation=result)
    else:
        checkpoint = a.checkpoint.resolve() if a.checkpoint else checkpoints()[-1]
        assert checkpoint.exists() and Path(str(checkpoint) + '.curriculum.pt').exists()
    completed = int(checkpoint.stem.split('_')[-1]) + 1
    remaining = plan['total_updates'] - completed
    recovery_attempts = 0
    while remaining > 0:
        phase = 'long_' + str(completed) + (f'_retry_{recovery_attempts}' if recovery_attempts else '')
        child, log = train(remaining, phase, checkpoint, True)
        status('training', training_pid=child.pid, start_checkpoint=str(checkpoint), completed_updates=completed)
        next_evaluation = (completed // plan['evaluate_interval'] + 1) * plan['evaluate_interval']
        while child.poll() is None:
            latest = checkpoints()[-1]
            iteration = int(latest.stem.split('_')[-1])
            status('training', latest_checkpoint=str(latest), latest_saved_iteration=iteration,
                completed_updates=iteration + 1, remaining_updates=plan['total_updates'] - iteration - 1)
            if job and job[0].poll() is not None:
                result = finish_evaluation(job)
                job = None
                status('training', latest_evaluation=result)
            if not job and iteration >= next_evaluation:
                job = evaluate(latest, f'iteration_{iteration}')
                next_evaluation = (iteration // plan['evaluate_interval'] + 1) * plan['evaluate_interval']
            time.sleep(plan.get('poll_interval_s', 30))
        log.close()
        if child.returncode:
            if not plan.get('automatic_recovery'):
                raise RuntimeError('PPO training failed; inspect ' + phase + '.log')
            run_name = '*_unified_' + plan.get('run_tag', '') + phase
            runs = sorted((r / 'logs/rsl_rl' / plan['experiment']).glob(run_name))
            request = runs[-1] / 'recovery_request.json' if runs else None
            if request is not None and request.exists():
                details = json.loads(request.read_text())
            else:
                tail = (training / (phase + '.log')).read_text(errors='replace')[-12000:]
                if not any(token in tail.lower() for token in ('closed-loop', 'nonfinite', 'recoverable_physics')):
                    raise RuntimeError('Non-recoverable PPO failure; inspect ' + phase + '.log')
                details = dict(checkpoint=None, error=tail[-2000:], optimizer_update_interrupted=True,
                    reason='Physics initialization or pre-run failure; use verified rollback')
            recovery_attempts += 1
            saved = Path(details['checkpoint']) if details.get('checkpoint') else None
            if recovery_attempts % 5 == 0 or details.get('optimizer_update_interrupted'):
                checkpoint = Path(plan['recovery_start_checkpoint'])
                abandoned = (r / 'logs/rsl_rl' / plan['experiment']).glob('*_unified_' + plan.get('run_tag', '') + '*')
                plan['ignored_checkpoint_runs'] = sorted(set(plan.get('ignored_checkpoint_runs', [])) |
                    {path.name for path in abandoned})
            elif saved and saved.exists() and Path(str(saved) + '.curriculum.pt').exists():
                checkpoint = saved
            else:
                checkpoint = checkpoints()[-1]
            completed = int(checkpoint.stem.split('_')[-1]) + 1
            remaining = plan['total_updates'] - completed
            record = dict(unix=time.time(), attempt=recovery_attempts,
                checkpoint=str(checkpoint), discarded_partial_rollout=True, details=details)
            with (training / 'automatic_restarts.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            status('recovering', recovery=record, completed_updates=completed,
                remaining_updates=remaining)
            time.sleep(min(60, plan.get('recovery_backoff_s', 10) * recovery_attempts))
            continue
        break
    if job:
        job[0].wait()
        finish_evaluation(job)
        job = None
    final = checkpoints()[-1]
    if int(final.stem.split('_')[-1]) != plan['total_updates'] - 1:
        raise RuntimeError('Final checkpoint iteration does not match plan')
    status('final_evaluation', final_checkpoint=str(final))
    final_results = []
    # A separate seed is reserved for final reporting at every difficulty band.
    for level in (0, 2, 5):
        job = evaluate(final, 'final_level_' + str(level), level=level, seed=91073)
        job[0].wait()
        final_results.append(finish_evaluation(job))
        job = None
    status('complete', completed_updates=plan['total_updates'], final_evaluations=final_results,
        final_physics_evaluations_passed=all(x.get('status') != 'EVALUATION_WARNING' for x in final_results))
except BaseException:
    # Fail closed; an evaluation failure must not leave an orphan training job.
    import signal
    for process in (child, job[0] if job else None):
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    status('failed', error=traceback.format_exc())
    raise
