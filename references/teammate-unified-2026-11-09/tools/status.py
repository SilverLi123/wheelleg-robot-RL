"""Read the unified job, PPO progress and empirical sampling coverage."""
import argparse
import json
from pathlib import Path
import re
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
a = p.parse_args()
r = a.root
state_path = r / 'training/status.json'
state = json.loads(state_path.read_text()) if state_path.exists() else {'stage': 'not_started'}
for role in ('supervisor', 'training'):
    pid = state.get(role + '_pid')
    state[role + '_alive'] = bool(pid and Path('/proc', str(pid)).exists())
logs = sorted((path for path in (r / 'training').glob('*.log')
               if path.name == 'pilot.log' or path.name.startswith('long_')),
              key=lambda path: path.stat().st_mtime)
if logs:
    log = logs[-1]
    with log.open('rb') as stream:
        stream.seek(max(0, log.stat().st_size - 60000))
        tail = stream.read().decode(errors='replace')
    iterations = re.findall(r'Learning iteration\s+(\d+)\s*/\s*(\d+)', tail)
    seconds = re.findall(r'Iteration time:\s*([\d.]+)s', tail)
    throughput = re.findall(r'Computation:\s*(\d+) steps/s', tail)
    state.update(log=str(log), log_age_s=round(time.time() - log.stat().st_mtime, 1),
        current_iteration=int(iterations[-1][0]) if iterations else None,
        iteration_seconds=float(seconds[-1]) if seconds else None,
        transitions_per_second=int(throughput[-1]) if throughput else None,
        activity=('updating' if iterations else 'initializing') if state.get('training_alive') else 'stopped')
health_paths = [path for path in (r / 'diagnostics/unified_locomotion').glob('*/training_health.json')
                if path.parent.name == 'pilot' or path.parent.name.startswith('long_')]
if health_paths:
    active_phase = logs[-1].stem if logs else None
    active_path = r / 'diagnostics/unified_locomotion' / str(active_phase) / 'training_health.json'
    path = active_path if active_path.exists() else max(health_paths, key=lambda path: path.stat().st_mtime)
    state['current_phase_health_pending'] = not active_path.exists()
    health = json.loads(path.read_text())
    state['health_phase'] = path.parent.name
    state['health_age_s'] = round(time.time() - path.stat().st_mtime, 1)
    state['physics'] = {key: health[key] for key in ('environments', 'max_loop_error_m', 'latest_loop_error_m')}
    state['coverage'] = health.get('unified')
    if not active_path.exists():
        state['previous_phase_health'] = dict(phase=path.parent.name,
            age_s=state['health_age_s'], physics=state.pop('physics'))
        state['coverage'] = None
print(json.dumps(state, ensure_ascii=False, indent=2))
