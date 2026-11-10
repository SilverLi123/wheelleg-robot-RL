"""Exercise actual supervisor restarts and LR/checkpoint hooks with injected faults."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
r = p.parse_args().root
results = {}

with tempfile.TemporaryDirectory(prefix='wheelbipe-supervisor-recovery-') as name:
    t = Path(name)
    for directory in ['tools', 'project/scripts/rsl_rl', 'training', 'initial',
                      'logs/rsl_rl/test/old_unified_long_6']:
        (t / directory).mkdir(parents=True, exist_ok=True)
    (t / 'initial/model_0.pt').write_text('baseline')
    (t / 'initial/model_0.pt.curriculum.pt').write_text('curriculum')
    # A higher checkpoint from the abandoned branch must never be selected.
    old = t / 'logs/rsl_rl/test/old_unified_long_6'
    (old / 'model_200.pt').write_text('abandoned')
    (old / 'model_200.pt.curriculum.pt').write_text('abandoned')
    plan = dict(experiment='test', num_envs=4096, total_updates=6, seed=1,
        gpu=0, validation_gpu=1, cpus='0', evaluate_interval=1,
        automatic_recovery=True, run_tag='test_',
        checkpoint_run_suffixes=['_unified_test_*'],
        recovery_start_checkpoint=str(t / 'initial/model_0.pt'),
        poll_interval_s=.03, recovery_backoff_s=.01)
    (t / 'plan.json').write_text(json.dumps(plan))
    (t / 'tools/run.sh').write_text('#!/bin/bash\ncd -- "$(dirname -- "$0")/.."\nexec python3 "$@"\n')
    trainer = '''import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run_name');p.add_argument('--checkpoint');a,_=p.parse_known_args()
r=Path.cwd();d=r/'logs/rsl_rl/test'/('2026_'+a.run_name);d.mkdir()
with (r/'launches.jsonl').open('a') as f:f.write(json.dumps(vars(a))+'\\n')
def save(i):
 q=d/f'model_{i}.pt';q.write_text('model');Path(str(q)+'.curriculum.pt').write_text('curriculum');return q
if 'retry' not in a.run_name:
 save(2);time.sleep(.7);q=save(3)
 (d/'recovery_request.json').write_text(json.dumps({'checkpoint':str(q),'optimizer_update_interrupted':False,'error':'RECOVERABLE_PHYSICS: injected'}))
 raise SystemExit(1)
save(4);time.sleep(.15);save(5)
'''
    (t / 'project/scripts/rsl_rl/train.py').write_text(trainer)
    evaluator = '''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output');a,_=p.parse_known_args()
Path(a.output,'summary.json').write_text(json.dumps({'status':'ERROR','error':'injected closure failure'}))
raise SystemExit(1)
'''
    (t / 'tools/validate.py').write_text(evaluator)
    proc = subprocess.run([sys.executable, str(r / 'tools/supervise.py'), '--root', str(t),
        '--resume'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
    assert proc.returncode == 0, proc.stdout[-5000:]
    state = json.loads((t / 'training/status.json').read_text())
    launches = [json.loads(v) for v in (t / 'launches.jsonl').read_text().splitlines()]
    assert len(launches) == 2 and launches[0]['checkpoint'].endswith('model_0.pt')
    assert launches[1]['checkpoint'].endswith('model_3.pt')
    assert state['stage'] == 'complete' and state['completed_updates'] == 6
    assert not state['final_physics_evaluations_passed']
    assert (t / 'training/evaluation_warnings.jsonl').exists()
    assert len((t / 'training/automatic_restarts.jsonl').read_text().splitlines()) == 1
    assert not (t / 'candidates/candidate.pt').exists()
    results['supervisor'] = dict(status='PASS', evaluation_failure_keeps_training=True,
        physics_failure_restarts_from_saved_checkpoint=True, abandoned_checkpoint_excluded=True,
        failed_evaluation_never_promoted=True, training_finishes_with_explicit_evaluation_warnings=True)

import torch
spec = importlib.util.spec_from_file_location('training_recovery', r / 'project/scripts/rsl_rl/training_recovery.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Runner:
    def __init__(self, directory, failure):
        self.current_learning_iteration = 7
        self.failure, self.directory, self.saved = failure, directory, []
        policy = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(policy.parameters(), lr=.02)
        self.alg = SimpleNamespace(policy=policy, optimizer=optimizer, learning_rate=.02, update=self.update)

    def update(self):
        if self.failure == 'during_update':
            raise FloatingPointError('RECOVERABLE_PHYSICS: nonfinite PPO update')
        self.alg.learning_rate = .02
        self.alg.optimizer.param_groups[0]['lr'] = .02
        self.alg.optimizer.zero_grad()
        loss = self.alg.policy(torch.ones(1, 2)).square().mean()
        loss.backward()
        self.alg.optimizer.step()
        self.used_lr = self.alg.optimizer.param_groups[0]['lr']
        return {'loss': float(loss.detach())}

    def learn(self, **kwargs):
        self.alg.update()
        if self.failure == 'during_collection':
            raise RuntimeError('RECOVERABLE_PHYSICS: widespread closed-loop failure')

    def save(self, path):
        torch.save({'iter': self.current_learning_iteration,
            'model_state_dict': self.alg.policy.state_dict()}, path)
        Path(path + '.curriculum.pt').write_text('{}')
        self.saved.append(path)

os.environ['WHEELBIPE_LEARNING_RATE_CAP'] = '.0003'
for failure in ['none', 'during_collection', 'during_update']:
    with tempfile.TemporaryDirectory(prefix='wheelbipe-ppo-recovery-') as name:
        t = Path(name)
        (t / 'params').mkdir()
        runner = Runner(t, failure)
        try:
            module.learn_with_recovery(runner, 2, str(t))
        except (RuntimeError, FloatingPointError):
            assert failure != 'none'
        if failure != 'during_update':
            assert runner.used_lr <= .0003
        if failure == 'during_collection':
            request = json.loads((t / 'recovery_request.json').read_text())
            assert request['last_completed_iteration'] == 7 and request['partial_rollout_discarded']
            assert torch.load(request['checkpoint'], weights_only=False)['iter'] == 7
        if failure == 'during_update':
            request = json.loads((t / 'recovery_request.json').read_text())
            assert request['checkpoint'] is None and not runner.saved
        assert not runner.alg.optimizer._optimizer_step_pre_hooks
results['ppo_hooks'] = dict(status='PASS', adaptive_lr_capped_at_each_optimizer_step=True,
    last_completed_iteration_saved=True, partial_optimizer_update_not_saved=True)
(r / 'reports/recovery_logic_tests.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
