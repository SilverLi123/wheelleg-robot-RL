"""Controlled reward pilot; keep live production training in its own run namespace."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=32)
    parser.add_argument('--num-envs', type=int, default=512)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='RUNNING', checkpoint=str(args.checkpoint),
        training_seed=86117, num_envs=args.num_envs, updates=args.updates,
        transitions_per_arm=args.num_envs * 24 * args.updates,
        actor_only_transfer=True, fresh_critic_and_adam=True,
        initial_mastery_level=2, exploration_std=.35,
        note='Short screening experiment with one training seed; no production promotion.',
        training=[], evaluations=[])
    state_path = output / 'experiment.json'

    def save():
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest, indent=2) + '\n')
        temp.replace(state_path)

    def launch(entry, argv, phase, gpu, cpus, profile):
        env = os.environ.copy()
        env.update(WHEELBIPE_EXPERIMENT=profile, WHEELBIPE_STAGE='3',
            WHEELBIPE_PHASE=phase, WHEELBIPE_INTEGRATION_SUBSTEPS='2',
            WHEELBIPE_GPU=str(gpu), WHEELBIPE_CPUSET=cpus,
            WHEELBIPE_AUTOMATIC_RECOVERY='1', WHEELBIPE_LEARNING_RATE_CAP='.0003')
        log = (output / (phase + '.log')).open('w')
        process = subprocess.Popen(['bash', str(root / 'tools/run.sh'), str(entry), *map(str, argv)],
            env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)
        return dict(process=process, log=log, started=time.time(), phase=phase)

    def wait_jobs(jobs):
        while jobs:
            for job in list(jobs):
                code = job['process'].poll()
                if code is None:
                    continue
                job['log'].close()
                row = job['record']
                row.update(returncode=code, wall_seconds=time.time() - job['started'], status='COMPLETE' if code == 0 else 'FAILED')
                if job.get('training'):
                    runs = sorted((root / 'logs/rsl_rl/wheelbipe_reward_ablation_20261109').glob('*_' + job['phase']))
                    if runs:
                        run = runs[-1]
                        row['run'] = str(run)
                        row['checkpoint'] = str(run / f'model_{args.updates - 1}.pt')
                        audit = run / 'training_updates.jsonl'
                        row['actual_updates'] = len(audit.read_text().splitlines()) if audit.exists() else 0
                        row['status'] = 'COMPLETE' if code == 0 and row['actual_updates'] == args.updates and Path(row['checkpoint']).exists() else 'FAILED'
                else:
                    report_path = Path(row['output']) / 'summary.json'
                    if report_path.exists():
                        result = json.loads(report_path.read_text())
                        row.update(physics_pass=result.get('physics_pass', False),
                            vx_mae=result.get('vx_mae'), failures=result.get('failures'),
                            no_failure_cases=sum(x['failures'] == 0 for x in result.get('rows', [])))
                save()
                print(json.dumps(row), flush=True)
                jobs.remove(job)
            if jobs:
                time.sleep(5)

    save()
    jobs = []
    for arm, gpu, cpus in [('control', 4, '0-3'), ('relative', 5, '40-43')]:
        phase = f'reward_height_{arm}_s86117'
        profile = 'unified_height_' + arm
        argv = ['--task', 'Robotics-Wheelbipe-Own-Explore-v0', '--num_envs', args.num_envs,
            '--max_iterations', args.updates, '--device', 'cuda:0', '--viz', 'none',
            '--seed', 86117, '--run_name', phase,
            '--experiment_name', 'wheelbipe_reward_ablation_20261109',
            '--checkpoint', args.checkpoint, '--checkpoint_architecture',
            '--actor_only_finetune', '--finetune_noise_std', '.35', '--audit_updates']
        job = launch(root / 'project/scripts/rsl_rl/train.py', argv, phase, gpu, cpus, profile)
        row = dict(arm=arm, profile=profile, pid=job['process'].pid, gpu=gpu, cpus=cpus, phase=phase, status='RUNNING')
        manifest['training'].append(row)
        job.update(record=row, training=True)
        jobs.append(job)
    save()
    wait_jobs(jobs)
    if any(row['status'] != 'COMPLETE' for row in manifest['training']):
        manifest['status'] = 'TRAINING_FAILED'
        save()
        raise RuntimeError('Pilot failed; inspect train logs and recovery requests. Main training is separate.')
    for seed, level in [(71903, 1), (71903, 5), (83021, 3)]:
        jobs = []
        for row in manifest['training']:
            phase = f"eval_height_{row['arm']}_s{seed}_l{level}"
            directory = output / phase
            argv = ['--root', root, '--checkpoint', row['checkpoint'], '--output', directory,
                '--level', level, '--seed', seed, '--seconds', 8, '--device', 'cuda:0',
                '--viz', 'none', '--reward-audit']
            # Evaluate all actors with identical original observation/physics
            # configuration. Shaping reward has no role in deterministic scoring.
            job = launch(root / 'tools/validate.py', argv, phase, row['gpu'], row['cpus'], 'unified_locomotion')
            result = dict(arm=row['arm'], level=level, seed=seed, output=str(directory), status='RUNNING')
            manifest['evaluations'].append(result)
            job.update(record=result, training=False)
            jobs.append(job)
        save()
        wait_jobs(jobs)
    manifest['status'] = 'COMPLETE' if all(row['status'] == 'COMPLETE' for row in manifest['evaluations']) else 'EVALUATION_FAILED'
    manifest['production_promoted'] = False
    save()


if __name__ == '__main__':
    main()
