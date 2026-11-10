"""One-shot handoff: retire only the previous root after real 4096 PPO updates."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import signal
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--previous-root', type=Path, required=True)
a = p.parse_args()
r, old = a.root.resolve(), a.previous_root.resolve()
assert r != old


def read(path):
    return json.loads(path.read_text())


def argv(pid):
    try:
        return Path('/proc', str(pid), 'cmdline').read_bytes().decode().split('\0')
    except FileNotFoundError:
        return []


def matches(pid, entry):
    arguments = argv(pid)
    if str(entry) in arguments:
        return True
    cwd = Path('/proc', str(pid), 'cwd').resolve()
    return any(argument and not argument.startswith('-') and Path(argument).name == entry.name
               and (cwd / argument).resolve() == entry for argument in arguments)


state = read(r / 'training/status.json')
assert state['stage'] == 'training' and state['num_envs'] == 4096
assert str(r / 'project/scripts/rsl_rl/train.py') in argv(state['training_pid'])
assert str(r / 'tools/supervise.py') in argv(state['supervisor_pid'])
runs = sorted((r / 'logs/rsl_rl' / read(r / 'plan.json')['experiment']).glob('*_unified_long_6'))
run = runs[-1]
resume = read(run / 'params/resume_verification.json')
assert resume['model_exact_match'] and resume['optimizer_exact_match']
assert resume['optimizer_state_restored'] and resume['optimizer_saved_values_exact']
assert resume['next_iteration'] == 6
rows = [json.loads(s) for s in (run / 'training_updates.jsonl').read_text().splitlines()]
assert len(rows) >= 5
for row in rows:
    assert all(math.isfinite(v) for v in row['losses'].values())
    assert math.isfinite(row['last_gradient_l2']) and row['last_gradient_l2'] > 0
    assert math.isfinite(row['parameter_delta_l2']) and row['parameter_delta_l2'] > 0
assert time.time() - (run / 'training_updates.jsonl').stat().st_mtime < 90
health_path = r / 'diagnostics/unified_locomotion/long_6/training_health.json'
health = read(health_path)
assert health['environments'] == 4096 and health['unified']['single_shared_policy']
assert health['domain_randomization']['enabled']
assert time.time() - health_path.stat().st_mtime < 180
pilot = read(r / 'reports/parallel_pilot/summary.json')
assert pilot['status'] == 'PASS' and pilot['physics_pass']

previous = read(old / 'training/status.json')
history = r / 'history/previous_256'
history.mkdir(parents=True, exist_ok=True)
if not (history / 'status_before.json').exists():
    (history / 'status_before.json').write_text(json.dumps(previous, indent=2) + '\n')
    (history / 'plan_before.json').write_text((old / 'plan.json').read_text())

# Match complete entry-point arguments to avoid touching unrelated processes.
targets = []
for role, entry in [('supervisor', old / 'tools/supervise.py'),
                    ('training', old / 'project/scripts/rsl_rl/train.py')]:
    pid = previous.get(role + '_pid')
    if pid and argv(pid):
        assert matches(pid, entry), (pid, argv(pid))
        targets.append((pid, role))
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and str(old / 'tools/validate.py') in argv(int(proc.name)):
        targets.append((int(proc.name), 'evaluation'))
stopped = []
for pid, role in targets:
    if not argv(pid):
        continue
    if role == 'supervisor':
        os.kill(pid, signal.SIGTERM)
    else:
        assert os.getpgid(pid) == pid, (pid, os.getpgid(pid))
        os.killpg(pid, signal.SIGTERM)
    stopped.append({'pid': pid, 'role': role, 'signal': 'SIGTERM'})
deadline = time.monotonic() + 15
while any(argv(pid) for pid, _ in targets) and time.monotonic() < deadline:
    time.sleep(.5)
assert all(not argv(pid) for pid, _ in targets), 'Previous processes have not exited'
assert str(r / 'project/scripts/rsl_rl/train.py') in argv(state['training_pid'])

now = time.time()
previous.update(stage='superseded_by_parallel', updated_unix=now,
                replacement_root=str(r), reason='User requested larger shared-policy training and irregular terrain')
(old / 'training/status.json').write_text(json.dumps(previous, indent=2) + '\n')
previous_plan = read(old / 'plan.json')
previous_plan.update(execution_state='superseded_by_parallel', replacement_root=str(r))
(old / 'plan.json').write_text(json.dumps(previous_plan, indent=2) + '\n')
proof = dict(status='PASS', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             root=str(r), previous_root=str(old), training_pid=state['training_pid'],
             supervisor_pid=state['supervisor_pid'], run=str(run), resume_verification=resume,
             verified_production_updates=len(rows), last_production_update=rows[-1],
             health=health, pilot_summary={k: pilot[k] for k in
                 ('status', 'physics_pass', 'vx_mae', 'failures', 'closure_peak_m')},
             stopped_previous_processes=stopped,
             previous_latest_checkpoint=previous.get('latest_checkpoint'),
             previous_checkpoints_preserved=True)
(r / 'reports/deployment.json').write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps({k: v for k, v in proof.items() if k != 'health'}, indent=2))
