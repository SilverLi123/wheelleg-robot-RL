"""Shared command sampler; the full speed support is present from iteration zero."""
import torch
from agent_tasks.manager.mdp.isaaclab.commands import SpecialModeUniformVelocityCommand

SPEED_EDGES = (-2.7, -2.1, -1.5, -.9, -.3, .3, .9, 1.5, 2.1, 2.7)


def sample_commands(n, device):
    # 70% stratified full-range, 20% low-speed, 10% stop. Half the stop
    # samples are pure yaw. Five percent of all samples include exact limits.
    edges = torch.tensor(SPEED_EDGES, device=device)
    # A random offset is essential when asynchronous resets request one env.
    # Otherwise every singleton resample would always select the first bin.
    bins = (torch.arange(n, device=device) + torch.randint(9, (), device=device)) % 9
    bins = bins[torch.randperm(n, device=device)]
    vx = edges[bins] + torch.rand(n, device=device) * (edges[bins + 1] - edges[bins])
    choice = torch.rand(n, device=device)
    slow = (choice >= .7) & (choice < .9)
    vx[slow] = torch.empty_like(vx[slow]).uniform_(-.4, .4)
    stop = choice >= .9
    vx[stop] = 0.
    edge = (torch.rand(n, device=device) < .05) & ~stop & ~slow
    vx[edge] = torch.where(vx[edge] >= 0., 2.7, -2.7)
    yaw = torch.empty(n, device=device).uniform_(-.8, .8)
    yaw[torch.rand(n, device=device) < .55] = 0.
    # Bound the lateral acceleration request, while keeping turns at all speeds.
    yaw_limit = torch.minimum(torch.full_like(vx, .8), 1.5 / vx.abs().clamp_min(.1))
    yaw = yaw.clamp(-yaw_limit, yaw_limit)
    standing = stop & (torch.rand(n, device=device) < .5)
    yaw[standing] = 0.
    return torch.stack((vx, torch.zeros_like(vx), yaw), -1), standing


class BalancedVelocityCommand(SpecialModeUniformVelocityCommand):
    def _resample_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        previous = self.vel_command_b[ids].clone()
        commands, standing = sample_commands(len(ids), self.device)
        self.vel_command_b[ids] = commands
        self.is_standing_env[ids] = standing
        self.is_heading_env[ids] = False
        if hasattr(self, 'special_mode_id'):
            self.special_mode_id[ids] = -1
        env = self._env
        if getattr(env, '_u_ready', False):
            values = dict(resamples=len(ids), sign_reversals=int(((previous[:, 0] * commands[:, 0]) < -.1).sum()),
                braking_to_zero=int(((previous[:, 0].abs() > .5) & (commands[:, 0] == 0.)).sum()),
                pure_yaw=int(((commands[:, 0] == 0.) & (commands[:, 2] != 0.)).sum()),
                combined_turn=int(((commands[:, 0].abs() > .5) & (commands[:, 2] != 0.)).sum()))
            for key, value in values.items():
                env._u_transitions[key] += value
