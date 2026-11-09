"""Observe the real PPO update path without changing its optimization or rewards."""
import json
import math
from pathlib import Path

import torch


def attach_training_audit(runner, directory):
    from pxr import Usd, UsdPhysics
    env = runner.env.unwrapped
    counts = []
    for index in range(env.num_envs):
        prefix = f"/World/envs/env_{index}/Robot"
        count = 0
        for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath(prefix)):
            if prim.IsA(UsdPhysics.SphericalJoint):
                joint = UsdPhysics.Joint(prim)
                if not joint.GetJointEnabledAttr().Get() or not joint.GetExcludeFromArticulationAttr().Get():
                    raise RuntimeError("Closed loop disabled: " + str(prim.GetPath()))
                for target in (*joint.GetBody0Rel().GetTargets(), *joint.GetBody1Rel().GetTargets()):
                    if not str(target).startswith(prefix + "/"):
                        raise RuntimeError("Closed loop crosses environments: " + str(prim.GetPath()))
                count += 1
        if count != 6:
            raise RuntimeError(f"Expected six closed loops in environment {index}, got {count}")
        counts.append(count)
    (Path(directory) / "cloned_mechanism_audit.json").write_text(json.dumps({"envs": env.num_envs,
        "closed_loops_per_env": counts, "bodies_per_env": len(env._native_robot.body_names),
        "dofs_per_env": len(env._native_robot.joint_names)}, indent=2))
    parameters = list(runner.alg.policy.named_parameters())
    initial = {name: value.detach().clone() for name, value in parameters}
    metrics = {"optimizer_steps": 0, "updates": 0}
    stream = (Path(directory) / "training_updates.jsonl").open("w", buffering=1)
    original_step = runner.alg.optimizer.step
    original_update = runner.alg.update

    def step(*args, **kwargs):
        gradients = [p.grad.detach().reshape(-1) for _, p in parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError("PPO optimizer has no gradients")
        grad = torch.cat(gradients)
        if not torch.isfinite(grad).all():
            raise FloatingPointError("Nonfinite PPO gradient")
        metrics["last_gradient_l2"] = float(grad.norm())
        result = original_step(*args, **kwargs)
        metrics["optimizer_steps"] += 1
        return result

    def update(*args, **kwargs):
        before = {name: p.detach().clone() for name, p in parameters}
        losses = original_update(*args, **kwargs)
        finite_losses = {name: float(value) for name, value in losses.items()}
        if not all(math.isfinite(x) for x in finite_losses.values()):
            raise FloatingPointError("Nonfinite PPO loss")
        row = dict(metrics)
        row["losses"] = finite_losses
        row["parameter_delta_l2"] = math.sqrt(sum(float((p.detach() - before[name]).square().sum()) for name, p in parameters))
        row["total_parameter_delta_l2"] = math.sqrt(sum(float((p.detach() - initial[name]).square().sum()) for name, p in parameters))
        row["actor_delta_l2"] = math.sqrt(sum(float((p.detach() - initial[name]).square().sum()) for name, p in parameters if name.startswith("actor.")))
        row["critic_delta_l2"] = math.sqrt(sum(float((p.detach() - initial[name]).square().sum()) for name, p in parameters if name.startswith("critic.")))
        if not all(torch.isfinite(p).all() for _, p in parameters):
            raise FloatingPointError("Nonfinite PPO parameters")
        if row["parameter_delta_l2"] == 0:
            raise RuntimeError("PPO update did not change any parameter")
        metrics["updates"] += 1
        row["updates"] = metrics["updates"]
        row["learning_rate"] = runner.alg.optimizer.param_groups[0]["lr"]
        stream.write(json.dumps(row) + "\n")
        return losses

    runner.alg.optimizer.step = step
    runner.alg.update = update
    print("[TRAINING AUDIT] Enabled finite loss/gradient/parameter and update checks")
