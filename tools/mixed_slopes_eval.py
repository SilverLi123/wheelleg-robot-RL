#!/usr/bin/env python3
"""Frozen-policy evaluation over every mixed-slope surface family."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


checkpoint_arg, output_arg = sys.argv[1:]
checkpoint_path = Path(checkpoint_arg).resolve()
output_path = Path(output_arg).resolve()
seed = int(os.environ.get("WHEELBIPE_EVAL_SEED", "20261111"))
steps = int(os.environ.get("WHEELBIPE_EVAL_STEPS", "600"))
if steps <= 0:
    raise ValueError("WHEELBIPE_EVAL_STEPS must be positive")

app = AppLauncher(
    headless=True,
    device="cuda:0",
    livestream=0,
    fast_shutdown=True,
    kit_args=f"--portable-root={Path.cwd() / 'kit'}",
).app

try:
    import torch

    from agent_tasks.direct.wheelbipe.wheelbipe_own.experiment import (
        WheelbipeOwnExperimentCfg,
        WheelbipeOwnExperimentEnv,
    )
    from agent_tasks.direct.wheelbipe.wheelbipe_own.mixed_slopes import (
        EVAL_TILES,
        GRADES,
        configure_evaluation,
    )
    from agent_world.sim620_compat import torch_data as T
    from scripts.rsl_rl.checkpoint_utils import prepare_checkpoint

    if os.environ.get("WHEELBIPE_EXPERIMENT") != "mixed_slopes_20261111":
        raise RuntimeError("This evaluator requires WHEELBIPE_EXPERIMENT=mixed_slopes_20261111")

    cfg = WheelbipeOwnExperimentCfg()
    cfg.seed = seed
    cfg.scene.num_envs = len(EVAL_TILES)
    cfg.sim.device = "cuda:0"
    cfg.height_range = [0.255, 0.255]
    cfg.default_height_cmd = 0.255
    cfg.commands.rel_standing_envs = 0.0
    cfg.commands.rel_heading_envs = 0.0
    cfg.commands.heading_command = False
    cfg.commands.ranges.lin_vel_x = (0.5, 0.5)
    cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.resampling_time_range = (100.0, 100.0)
    configure_evaluation(cfg)
    prepare_checkpoint(cfg, None, checkpoint_path)
    print("[MIXED-EVAL] config and checkpoint metadata ready", flush=True)

    source = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
    actor_state = {
        key.removeprefix("actor."): value
        for key, value in source["model_state_dict"].items()
        if key.startswith("actor.")
    }
    weights = sorted(
        (int(key.split(".")[0]), value)
        for key, value in actor_state.items()
        if key.endswith(".weight")
    )
    if weights[0][1].shape[1] != 35 or weights[-1][1].shape[0] != 6:
        raise RuntimeError("Actor input/output dimensions changed")
    layers = []
    for index, (_, weight) in enumerate(weights):
        layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
        if index + 1 < len(weights):
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).to("cuda:0").eval()
    actor.load_state_dict(actor_state, strict=True)
    print("[MIXED-EVAL] actor ready", flush=True)

    env = WheelbipeOwnExperimentEnv(cfg)
    print("[MIXED-EVAL] environment constructed", flush=True)
    terrain_types = env.terrain.terrain_types.tolist()
    print("[MIXED-EVAL] terrain types", terrain_types, flush=True)
    if terrain_types != list(range(len(EVAL_TILES))):
        raise RuntimeError(f"Unexpected evaluation tile order: {terrain_types}")
    if cfg.sim.dt != 0.0025 or env.step_dt != 0.02:
        raise RuntimeError("Physics/policy timing changed")
    print("[MIXED-EVAL] timing verified", flush=True)

    # Mesh vertices are checked separately by tools/test_mixed_slopes.py before
    # launch. Avoid Warp raycasts here because this Isaac Sim build exits during
    # the post-construction mesh query even though training and stepping are fine.
    from agent_tasks.direct.wheelbipe.wheelbipe_own.mixed_slopes import GRADES

    contact_names = list(env.contact_sensor.body_names)
    nonwheel_indices = [
        index for index, name in enumerate(contact_names) if "wheel" not in name
    ]
    peak_nonwheel = torch.zeros(
        env.num_envs, len(nonwheel_indices), device=env.device
    )
    update = env.scene.update

    def update_and_sample_contacts(dt):
        update(dt)
        force = T(env.contact_sensor.data.net_forces_w)[:, nonwheel_indices]
        peak_nonwheel.copy_(
            torch.maximum(
                peak_nonwheel,
                torch.linalg.vector_norm(force, dim=-1),
            )
        )

    env.scene.update = update_and_sample_contacts
    obs, _ = env.reset()
    print("[MIXED-EVAL] rollout started", flush=True)
    initial_pos = T(env._native_robot.data.root_pos_w).clone()
    speed_error = torch.zeros(env.num_envs, device=env.device)
    resets = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    pitch_min = torch.full((env.num_envs,), math.inf, device=env.device)
    pitch_max = torch.full((env.num_envs,), -math.inf, device=env.device)

    def world_pitch_rad(quaternion: torch.Tensor) -> torch.Tensor:
        # Isaac Lab quaternions use (w, x, y, z).
        w, x, y, z = quaternion.unbind(-1)
        sine = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
        return torch.asin(sine)

    with torch.inference_mode():
        for _ in range(steps):
            if obs["policy"].shape != (env.num_envs, 35):
                raise RuntimeError("Policy observation shape changed")
            if obs["critic"].shape != (env.num_envs, 78):
                raise RuntimeError("Critic observation shape changed")
            velocity = T(env._native_robot.data.root_lin_vel_b)[:, 0]
            speed_error += (velocity - env.command_generator.command[:, 0]).abs()
            pitch = world_pitch_rad(T(env._native_robot.data.root_quat_w))
            pitch_min.copy_(torch.minimum(pitch_min, pitch))
            pitch_max.copy_(torch.maximum(pitch_max, pitch))
            obs, rewards, done, timeout, _ = env.step(actor(obs["policy"]))
            if not torch.isfinite(rewards).all():
                raise FloatingPointError("Nonfinite reward")
            resets += (done | timeout).long()

    final_pos = T(env._native_robot.data.root_pos_w)
    displacement = final_pos[:, :2] - initial_pos[:, :2]
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_iteration": source.get("iter"),
        "seed": seed,
        "steps": steps,
        "policy_dt_s": env.step_dt,
        "command": [0.5, 0.0],
        "terrain_types": terrain_types,
        "target_grades": {tile: GRADES[tile] for tile in EVAL_TILES},
        "results": [
            {
                "tile": tile,
                "target_grade": GRADES[tile],
                "resets": int(resets[index]),
                "vx_mae": float(speed_error[index] / steps),
                "forward_m": float(displacement[index, 0]),
                "lateral_m": float(displacement[index, 1]),
                "mean_forward_speed_m_s": float(
                    displacement[index, 0] / (steps * env.step_dt)
                ),
                "vertical_change_m": float(final_pos[index, 2] - initial_pos[index, 2]),
                "pitch_range_deg": [
                    math.degrees(float(pitch_min[index])),
                    math.degrees(float(pitch_max[index])),
                ],
                "nonwheel_contact_peak_n": {
                    contact_names[contact_index]: float(peak_nonwheel[index, column])
                    for column, contact_index in enumerate(nonwheel_indices)
                    if float(peak_nonwheel[index, column]) > 0.0
                },
            }
            for index, tile in enumerate(EVAL_TILES)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print("[MIXED-EVAL] report written", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    env.close()
finally:
    app.close()
