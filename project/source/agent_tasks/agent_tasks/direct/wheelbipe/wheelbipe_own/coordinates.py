"""Policy-coordinate view over an unchanged native articulation.

Only environment reads/writes cross this boundary. InteractiveScene keeps the
native articulation for simulation; USD anchors and native PhysX coordinates
are never altered. q_policy = sign*q_native + offset; power is invariant.
"""
import torch
from agent_world.sim620_compat import torch_data


class PolicyData:
    def __init__(self, native, sign, offset):
        self._native, self._sign, self._offset = native, sign, offset

    def __getattr__(self, name):
        value = torch_data(getattr(self._native, name))
        if name in ("joint_pos", "default_joint_pos", "joint_pos_target"):
            return value * self._sign + self._offset
        if name in ("joint_vel", "default_joint_vel", "joint_acc", "applied_torque", "computed_torque",
                    "joint_vel_target", "joint_effort_target"):
            return value * self._sign
        if name in ("joint_pos_limits", "soft_joint_pos_limits"):
            transformed = value * self._sign.unsqueeze(-1) + self._offset.unsqueeze(-1)
            return transformed.sort(dim=-1).values
        return value


class PolicyArticulation:
    def __init__(self, native, mapping):
        self.native = native
        self.sign = torch.tensor([mapping[n]["sign"] for n in native.joint_names], device=native.device)
        self.offset = torch.tensor([mapping[n]["offset"] for n in native.joint_names], device=native.device)
        self.data = PolicyData(native.data, self.sign, self.offset)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _native(self, value, joint_ids, position=False):
        ids = slice(None) if joint_ids is None else joint_ids
        return (value - (self.offset[ids] if position else 0)) * self.sign[ids]

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        self.native.set_joint_position_target(self._native(target, joint_ids, True), joint_ids, env_ids)

    def set_joint_velocity_target(self, target, joint_ids=None, env_ids=None):
        self.native.set_joint_velocity_target(self._native(target, joint_ids), joint_ids, env_ids)

    def set_joint_effort_target(self, target, joint_ids=None, env_ids=None):
        self.native.set_joint_effort_target(self._native(target, joint_ids), joint_ids, env_ids)

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
        self.native.write_joint_state_to_sim(self._native(position, joint_ids, True),
                                             self._native(velocity, joint_ids), joint_ids, env_ids)

    def write_joint_position_to_sim(self, position, joint_ids=None, env_ids=None):
        self.native.write_joint_position_to_sim(self._native(position, joint_ids, True), joint_ids, env_ids)

    def write_joint_velocity_to_sim(self, velocity, joint_ids=None, env_ids=None):
        self.native.write_joint_velocity_to_sim(self._native(velocity, joint_ids), joint_ids, env_ids)
