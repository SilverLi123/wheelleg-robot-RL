"""Apply the reviewed integration hooks to the archived server checkout."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
r = p.parse_args().root
path = r / 'project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/experiment.py'
s = path.read_text()
old = "        if peak > .02 or self._audit_bad_streak > 5:\n"
assert s.count(old) == 1
s = s.replace(old, "        if (peak > .02 or self._audit_bad_streak > 5) and not getattr(self, '_u_ready', False):\n")
old = "        return super()._postprocess_reward_terms(terms)\n"
assert s.count(old) == 1
s = s.replace(old, """        if getattr(self, '_u_ready', False):
            mask = self._u_closure_mask
            terms = {key: torch.where(mask.reshape(self.num_envs, *([1] * (value.ndim - 1))),
                         torch.zeros_like(value), value) for key, value in terms.items()}
        return super()._postprocess_reward_terms(terms)
""")
path.write_text(s)
path = r / 'project/scripts/rsl_rl/train.py'
s = path.read_text()
old = '    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)'
assert s.count(old) == 1
s = s.replace(old, """    if os.environ.get('WHEELBIPE_AUTOMATIC_RECOVERY') == '1':
        from training_recovery import learn_with_recovery
        learn_with_recovery(runner, agent_cfg.max_iterations, log_dir)
    else:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)""")
path.write_text(s)
print('Installed per-environment reset and PPO recovery hooks')
