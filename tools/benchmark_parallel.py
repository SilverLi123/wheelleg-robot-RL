"""Run isolated real PPO benchmarks without interrupting the active trainer."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--counts', type=int, nargs='+', default=[256, 4096])
p.add_argument('--gpus', type=int, nargs='+', default=[2, 3])
p.add_argument('--updates', type=int, default=6)
a = p.parse_args()
assert len(a.counts) == len(a.gpus)
r = a.root.resolve()
source = json.loads((r / 'initialization/source.json').read_text())
jobs = []
for count, gpu in zip(a.counts, a.gpus):
    directory = r / 'benchmarks' / str(count)
    directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(WHEELBIPE_EXPERIMENT='unified_locomotion', WHEELBIPE_STAGE='3',
        WHEELBIPE_PHASE='bench_' + str(count), WHEELBIPE_INTEGRATION_SUBSTEPS='2',
        WHEELBIPE_GPU=str(gpu), WHEELBIPE_CPUSET=('24-27' if count <= 256 else
            '40-43' if gpu == 4 else '44-47' if gpu == 5 else '28-39'))
    argv = ['bash', str(r / 'tools/run.sh'), str(r / 'project/scripts/rsl_rl/train.py'),
        '--task', 'Robotics-Wheelbipe-Own-Explore-v0', '--num_envs', str(count),
        '--max_iterations', str(a.updates), '--device', 'cuda:0', '--viz', 'none',
        '--seed', '20261108', '--run_name', 'bench_' + str(count), '--checkpoint', source['copy'],
        '--checkpoint_architecture', '--audit_updates']
    log = (directory / 'train.log').open('w')
    process = subprocess.Popen(argv, env=environment, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True)
    row = dict(count=count, gpu=gpu, pid=process.pid, started_unix=time.time(), status='RUNNING',
        process=process, log=log, directory=directory, peak_gpu_memory_MiB=0)
    jobs.append(row)


def snapshot(row):
    return {k: str(v) if isinstance(v, Path) else v for k, v in row.items() if k not in ('process', 'log')}


while any(row['status'] == 'RUNNING' for row in jobs):
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    usage = {int(index): int(value) for index, value in (line.split(',') for line in memory.splitlines())}
    for row in jobs:
        if row['status'] != 'RUNNING':
            continue
        row['peak_gpu_memory_MiB'] = max(row['peak_gpu_memory_MiB'], usage[row['gpu']])
        code = row['process'].poll()
        if code is None:
            continue
        row['log'].close()
        row.update(returncode=code, wall_seconds=time.time() - row['started_unix'])
        text = re.sub(r'\x1b\[[0-9;]*m', '', (row['directory'] / 'train.log').read_text())
        metrics = re.findall(r'Computation:\s*(\d+) steps/s \(collection:\s*([\d.]+)s, learning\s*([\d.]+)s\)', text)
        row['iterations'] = [dict(steps_per_second=int(v), collection_seconds=float(c), learning_seconds=float(l)) for v, c, l in metrics]
        health = r / 'diagnostics/unified_locomotion' / ('bench_' + str(row['count'])) / 'training_health.json'
        row['health'] = json.loads(health.read_text()) if health.exists() else None
        runs = sorted((r / 'logs/rsl_rl/wheelbipe_unified_parallel_20261108').glob('*_bench_' + str(row['count'])))
        updates = []
        if runs:
            row['run'] = str(runs[-1])
            audit = runs[-1] / 'training_updates.jsonl'
            if audit.exists():
                updates = [json.loads(line) for line in audit.read_text().splitlines()]
        row['actual_ppo_updates'] = len(updates)
        row['last_update'] = updates[-1] if updates else None
        row['status'] = 'PASS' if code == 0 and len(updates) == a.updates and row['health'] else 'FAIL'
        if row['status'] == 'FAIL':
            row['error_tail'] = text[-8000:]
        (row['directory'] / 'result.json').write_text(json.dumps(snapshot(row), indent=2) + '\n')
        print(json.dumps({k:v for k,v in snapshot(row).items() if k not in ('health', 'error_tail')}), flush=True)
    (r / 'benchmarks' / ('status_' + '_'.join(map(str, a.counts)) + '.json')).write_text(json.dumps([snapshot(row) for row in jobs], indent=2) + '\n')
    if any(row['status'] == 'RUNNING' for row in jobs):
        time.sleep(15)
