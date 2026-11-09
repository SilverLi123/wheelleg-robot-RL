"""V14 reuse with explicit coordinate, reset, contact and gas-spring boundaries."""
from agent_world.sim620_compat import torch_data
import json
import os
from pathlib import Path

import torch
from isaaclab.utils.math import quat_apply

from agent_world.assets.wheelbipe_own import (
    ASSET_DIR, MECHANISM, GROUPS, GUIDE_BODIES, UNDESIRED_BODIES, RESET_BODIES,
)
from agent_tasks.direct.wheelbipe.wheelbipe_V14.env import WheelbipeV14Env
from .coordinates import PolicyArticulation


class WheelbipeOwnEnv(WheelbipeV14Env):
    def _setup_scene(self):
        # Lab 3 constructs and clones the declared scene before this callback.
        self.robot = self.scene.articulations["robot"]
        self.contact_sensor = self.scene.sensors["contact_sensor"]
        for name in ("height_scanner", "right_wheel_height_scanner", "left_wheel_height_scanner", "dot_scanner"):
            setattr(self, name, self.scene.sensors.get(name))
        self.terrain = self.scene.terrain
        self._set_play_height_scanner_debug_vis()

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._native_robot = self.robot
        if set(self.robot.joint_names) != set(MECHANISM["joints"]):
            raise RuntimeError("Own articulation DOFs do not match audited 33 ordinary joints")
        if set(self.robot.body_names) != set(MECHANISM["bodies"]):
            raise RuntimeError("Own articulation bodies do not match audited 34 bodies")
        self.robot = PolicyArticulation(self._native_robot, MECHANISM["joints"])
        # Fix the policy action order explicitly, independently of PhysX traversal order.
        self._legs_act_idx, self._legs_act_idx_name = self.robot.find_joints(list(GROUPS["legs_act"]), preserve_order=True)
        self._wheel_idx, _ = self.robot.find_joints(list(GROUPS["wheel"]), preserve_order=True)
        self._spring_idx, _ = self.robot.find_joints(list(GROUPS["spring"]), preserve_order=True)
        self._actuate_idx = self._legs_act_idx + self._wheel_idx
        self._left_front1_leg_act_local_idx, self._left_rear1_leg_act_local_idx = 0, 1
        self._wheel_link_idx, _ = self.robot.find_bodies(["left_wheel_link", "right_wheel_link"], preserve_order=True)
        self._guide_link_idx, _ = self.robot.find_bodies(list(GUIDE_BODIES), preserve_order=True)
        self._undesired_contact_link_idx = self._strict_contacts(UNDESIRED_BODIES)
        self._desired_contact_link_idx = self._strict_contacts(("left_wheel_link", "right_wheel_link"))
        self._reset_contact_link_idx = self._strict_contacts(RESET_BODIES)
        self._init_static_index_layouts()
        self.max_wheel_vel = float(cfg.max_wheel_vel)  # base class adds a reference-specific 1.5 multiplier
        nominal = torch_data(self.robot.data.default_joint_pos)[0, self._legs_act_idx]
        self._own_lower_target = nominal - 0.30
        self._own_upper_target = nominal + 0.30
        bank = json.loads((ASSET_DIR / "reset_bank.json").read_text())
        self._reset_bank = {s: torch.tensor(v, device=self.device, dtype=torch.float)
                            for s, v in bank["banks"].items()}
        self._bank_names = bank["names"]
        self._bank_body_z = {s: torch.tensor(v, device=self.device, dtype=torch.float)[:, 2]
                             for s, v in bank["wheel_endpoints"].items()}
        self._spring_outside = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._spring_commanded_net = torch.zeros(self.num_envs, 2, device=self.device)
        self._own_reset_count = 0
        self._sync_policy_state()
        self._own_index_manifest = self.index_manifest()
        self._verify_actuators()
        dest = Path(os.environ.get("WHEELBIPE_DIAGNOSTICS_DIR", "logs/own_robot"))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "runtime_indices.json").write_text(json.dumps(self._own_index_manifest, indent=2))

    def _strict_contacts(self, names):
        result = []
        for name in names:
            if self.contact_sensor.body_names.count(name) != 1:
                raise RuntimeError(f"Contact sensor coverage mismatch: {name}")
            result.append(self.contact_sensor.body_names.index(name))
        if len(result) != len(set(result)):
            raise RuntimeError("Repeated contact body")
        return result

    def _verify_actuators(self):
        covered = []
        for group, expected in GROUPS.items():
            act = self._native_robot.actuators[group]
            actual = list(act.joint_names)
            if set(actual) != set(expected):
                raise RuntimeError(f"Actuator group mismatch: {group}: {actual}")
            covered += actual
        if len(covered) != 33 or len(set(covered)) != 33:
            raise RuntimeError("Actuator coverage is missing or duplicated")
        # Explicit actuator must have cleared the native simulation drives.
        view = self._native_robot.root_physx_view
        for label, values in (("native_K", torch_data(view.get_dof_stiffnesses())), ("native_D", torch_data(view.get_dof_dampings()))):
            if torch.count_nonzero(values).item():
                raise RuntimeError(f"Explicit actuator has residual {label}")
        spring = self._native_robot.actuators["spring"]
        if torch.any(spring.stiffness != 0) or torch.any(spring.damping != 50):
            raise RuntimeError("Spring must have exactly one D=50 and K=0")

    def index_manifest(self):
        actions = {n: i for i, n in enumerate((*GROUPS["legs_act"], *GROUPS["wheel"]))}
        native_k = torch_data(self._native_robot.root_physx_view.get_dof_stiffnesses())[0].tolist()
        native_d = torch_data(self._native_robot.root_physx_view.get_dof_dampings())[0].tolist()
        actuator_rows = {}
        for group, actuator in self._native_robot.actuators.items():
            ids = [self.robot.joint_names.index(n) for n in actuator.joint_names]
            actuator_rows[group] = {"joint_names": list(actuator.joint_names), "joint_indices": ids,
                "first_environment_parameters": {key: (torch_data(self._native_robot.data.joint_armature)[0, ids]
                    if key == "armature" else getattr(actuator, key)[0]).tolist()
                    for key in ("stiffness", "damping", "armature", "effort_limit", "velocity_limit")},
                "native_physx_K": [native_k[i] for i in ids], "native_physx_D": [native_d[i] for i in ids],
                "native_effort_limit": torch_data(self._native_robot.data.joint_effort_limits)[0, ids].tolist(),
                "native_velocity_limit": torch_data(self._native_robot.data.joint_vel_limits)[0, ids].tolist()}
        return {"equation": "q_policy=s*q_native+b", "yaw_strategy": "passive D=0.5, K=0; no pitch",
            "actuators": actuator_rows,
            "joints": [{"index": i, "name": n, "sign": MECHANISM["joints"][n]["sign"],
                "offset": MECHANISM["joints"][n]["offset"], "unit": MECHANISM["joints"][n]["unit"],
                "actuator": next(k for k, v in GROUPS.items() if n in v),
                "action": actions.get(n), "mode": "position" if n in GROUPS["legs_act"] else
                "velocity" if n in GROUPS["wheel"] else "gas+damper" if n in GROUPS["spring"] else "passive"}
                for i, n in enumerate(self.robot.joint_names)],
            "bodies": [{"index": i, "name": n, "contact_index": self.contact_sensor.body_names.index(n),
                "undesired": n in UNDESIRED_BODIES, "reset_contact": n in RESET_BODIES}
                for i, n in enumerate(self.robot.body_names)], "randomization": "disabled"}

    def _sync_policy_state(self):
        self.joint_pos = torch_data(self.robot.data.joint_pos)
        self.joint_vel = torch_data(self.robot.data.joint_vel)
        self.default_joint_pos = torch_data(self.robot.data.default_joint_pos)
        if not torch.isfinite(self.joint_pos).all() or not torch.isfinite(self.joint_vel).all():
            raise FloatingPointError("Nonfinite native/policy joint state; abort instead of sanitizing")
        for name in ("root_state_w", "applied_torque", "joint_acc"):
            if not torch.isfinite(getattr(self.robot.data, name)).all():
                raise FloatingPointError(f"Nonfinite {name} before observation/reward sanitizing")

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Nonfinite policy actions")
        self._sync_policy_state()
        super()._pre_physics_step(actions)

    def _decode_leg_policy_actions(self, actions):
        # Keep cfg serializable for Hydra/checkpoint YAML; per-joint tensors
        # belong to the runtime controller, not the persistent configuration.
        decoded = super()._decode_leg_policy_actions(actions)
        return decoded.clamp(self._own_lower_target, self._own_upper_target)

    def _update_obs(self, is_obs=False):
        if hasattr(self, "_native_robot"):
            self._sync_policy_state()
        super()._update_obs(is_obs)

    def _get_rewards(self):
        self._sync_policy_state()
        return super()._get_rewards()

    def _get_observations(self):
        if hasattr(self, "_native_robot"):
            self._sync_policy_state()
        obs = super()._get_observations()
        for name, dim in (("policy", 35), ("critic", 78)):
            if obs[name].shape != (self.num_envs, dim) or not torch.isfinite(obs[name]).all():
                raise RuntimeError(f"Own {name} observation mismatch: {obs[name].shape}")
        return obs

    def get_privilaged_obs(self):
        if hasattr(self, "_native_robot"):
            self._sync_policy_state()
        return super().get_privilaged_obs()

    def _debug_obs_alert(self, blocks, stream_name):
        for name, value in blocks.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"{stream_name}.{name} is nonfinite before clipping")
        super()._debug_obs_alert(blocks, stream_name)

    def _postprocess_reward_terms(self, terms):
        terms = super()._postprocess_reward_terms(terms)
        for name, weight in self.cfg.rewards.items():
            if weight and name not in terms:
                raise RuntimeError(f"Enabled reward has no implementation: {name}")
        for name, value in terms.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Reward {name} is nonfinite before sanitizing")
        self._own_raw_rewards = {k: v.detach() for k, v in terms.items() if k in self.cfg.rewards}
        return terms

    def _apply_spring(self):
        q = torch_data(self._native_robot.data.joint_pos)[:, self._spring_idx]
        dq = torch_data(self._native_robot.data.joint_vel)[:, self._spring_idx]
        self.spring_force = 580.0 - (180.0 / 0.042) * q.clamp(0.0, 0.042)
        # D=50 is applied exactly once inside IdealPD, NOT here.
        self.robot.set_joint_velocity_target(torch.zeros_like(q), joint_ids=self._spring_idx)
        self.robot.set_joint_effort_target(self.spring_force, joint_ids=self._spring_idx)
        self._spring_commanded_net = (self.spring_force - self._native_robot.actuators["spring"].damping * dq).clamp(-1000.0, 1000.0)
        tolerance = self.cfg.own_spring_tolerance_m
        # Latch across all substeps so a brief out-of-band excursion cannot disappear.
        self._spring_outside |= ((q < -tolerance) | (q > 0.042 + tolerance)).any(dim=-1)

    def _get_dones(self):
        self._sync_policy_state()
        terminated, timeouts = super()._get_dones()
        outside = getattr(self, "_spring_outside", torch.zeros_like(terminated))
        self._termination_reason_masks["spring_working_range"] = outside.clone()
        return terminated | outside, timeouts

    def _custom_reset_random(self, env_ids):
        # Called through the original reset bookkeeping. No legacy IK or random
        # root/velocity event is invoked. Both springs and every passive DOF reset.
        n = len(env_ids)
        native_q = torch_data(self._native_robot.data.default_joint_pos)[env_ids].clone()
        z = torch.full((n,), 0.265, device=self.device)
        if self.cfg.own_random_reset:
            depths = []
            for side in ("left", "right"):
                bank = self._reset_bank[side]
                samples = torch.randint(bank.shape[0], (n,), device=self.device)
                names = [x for x in self._bank_names if x.startswith(side + "_")]
                src = [self._bank_names.index(x) for x in names]
                dst = [self.robot.joint_names.index(x) for x in names]
                native_q[:, dst] = bank[samples][:, src]
                depths.append(-self._bank_body_z[side][samples])
            z = torch.stack(depths, dim=-1).max(dim=-1).values + 0.0625 + 0.005
        policy_q = native_q * self.robot.sign + self.robot.offset
        zeros = torch.zeros_like(policy_q)
        self.robot.write_joint_state_to_sim(policy_q, zeros, env_ids=env_ids)
        self.robot.set_joint_position_target(policy_q, env_ids=env_ids)
        self.robot.set_joint_velocity_target(zeros, env_ids=env_ids)
        self.robot.set_joint_effort_target(zeros, env_ids=env_ids)
        root = torch_data(self._native_robot.data.default_root_state)[env_ids].clone()
        root[:, :3] = self.terrain.env_origins[env_ids]
        root[:, 2] += z
        root[:, 3:7] = root.new_tensor([0.0, 0.0, 0.0, 1.0])
        root[:, 7:] = 0.0
        self.robot.write_root_state_to_sim(root, env_ids)
        self._spring_outside[env_ids] = False
        self._own_reset_count += n
        self.spring_force_rand[env_ids] = 0.0
        self._sync_policy_state()

    def _inverse_kinematics(self, leg_length, leg_angle):
        raise ValueError("Own robot uses the validated native reset bank; use scripts/own_robot/kinematics.py "
                         "Mechanism.solve(endpoint=...) for constraint-checked offline IK")

    def _get_wheel_motor_z_axis_align_error_sq(self):
        # Preserve the legacy reward's meaning: leg-root -> wheel vs gravity.
        ends = torch_data(self.robot.data.body_pos_w)[:, self._wheel_link_idx]
        pivots = torch.tensor([MECHANISM["joints"][s + "_front1_joint"]["frame0"]
                               for s in ("left", "right")], device=self.device)[:, :3, 3]
        origins = torch_data(self.robot.data.root_pos_w)[:, None] + quat_apply(
            torch_data(self.robot.data.root_quat_w)[:, None].expand(-1, 2, -1), pivots[None].expand(self.num_envs, -1, -1))
        leg = ends - origins
        norms = torch.linalg.vector_norm(leg, dim=-1, keepdim=True)
        if torch.any(norms < 1e-6):
            raise FloatingPointError("Degenerate leg-root to wheel vector")
        gravity = leg.new_tensor(self.cfg.sim.gravity)
        gravity /= torch.linalg.vector_norm(gravity)
        cosine = ((leg / norms) * gravity).sum(-1)
        return (1.0 - cosine.square()).clamp_min(0).mean(-1)
