"""Offline unit tests for Stage 3 curriculum scheduling, block sampling, and terrain-relative orientation."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RANDOM_TERRAIN_MAP_PATH = (
    ROOT
    / "project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own/random_terrain_map.py"
)
PROFILES_PATH = ROOT / "profiles.json"


def _load_random_terrain_map():
    spec = importlib.util.spec_from_file_location("random_terrain_map_unit", RANDOM_TERRAIN_MAP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rotated_z_axis(roll: float, pitch: float, yaw: float) -> tuple[float, float, float]:
    """Return R_z(yaw) @ R_y(pitch) @ R_x(roll) @ [0, 0, 1]^T (intrinsic ZYX convention)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # R_y(p) R_x(r) e_z = (sin(p)*cos(r), -sin(r), cos(p)*cos(r))
    lx = sp * cr
    ly = -sr
    lz = cp * cr
    return (
        cy * lx - sy * ly,
        sy * lx + cy * ly,
        lz,
    )


def test_curriculum_progression_and_interpolation(rtm) -> None:
    profiles = json.loads(PROFILES_PATH.read_text())
    stage3 = profiles["random_terrain_stage3_curriculum_20261115"]
    assert "random_terrain_stage3_curriculum_20261115" in rtm.RANDOM_TERRAIN_PROFILES
    cfg = stage3["curriculum"]

    assert rtm.curriculum_progress_fraction(0, cfg["warmup_iters"], cfg["ramp_iters"]) == 0.0
    assert rtm.curriculum_progress_fraction(10, cfg["warmup_iters"], cfg["ramp_iters"]) == 0.0
    assert math.isclose(rtm.curriculum_progress_fraction(50, cfg["warmup_iters"], cfg["ramp_iters"]), 0.5)
    assert rtm.curriculum_progress_fraction(90, cfg["warmup_iters"], cfg["ramp_iters"]) == 1.0
    assert rtm.curriculum_progress_fraction(140, cfg["warmup_iters"], cfg["ramp_iters"]) == 1.0

    s0 = rtm.interpolate_stage_curriculum(cfg, 0)
    s10 = rtm.interpolate_stage_curriculum(cfg, 10)
    s50 = rtm.interpolate_stage_curriculum(cfg, 50)
    s90 = rtm.interpolate_stage_curriculum(cfg, 90)
    s140 = rtm.interpolate_stage_curriculum(cfg, 140)

    assert s0["progress"] == 0.0 and s10["progress"] == 0.0
    assert s0["lin_vel_x"] == (-0.9, 1.35)
    assert s0["ang_vel_z"] == (-1.05, 1.05)
    assert math.isclose(s0["standing_fraction"], 0.22)
    assert math.isclose(s0["steep_block_prob"], 0.22)

    assert math.isclose(s50["progress"], 0.5)
    assert math.isclose(s50["lin_vel_x"][0], -1.05)
    assert math.isclose(s50["lin_vel_x"][1], 1.675)
    assert math.isclose(s50["ang_vel_z"][0], -1.275)
    assert math.isclose(s50["ang_vel_z"][1], 1.275)
    assert math.isclose(s50["standing_fraction"], 0.235)
    assert math.isclose(s50["steep_block_prob"], 0.335)

    assert s90["progress"] == 1.0 and s140["progress"] == 1.0
    assert s140["lin_vel_x"] == (-1.2, 2.0)
    assert s140["ang_vel_z"] == (-1.5, 1.5)
    assert math.isclose(s140["standing_fraction"], 0.25)
    assert math.isclose(s140["steep_block_prob"], 0.45)


def test_block_sampling_probs(rtm) -> None:
    steep_mask = [abs(rtm.GRADES[name]) >= 0.14 for name in rtm.TERRAIN_TYPES]
    assert sum(1 for flag in steep_mask if flag) == 4
    assert len(steep_mask) == 9

    for p_steep in (0.22, 0.335, 0.45):
        probs = rtm.compute_block_sampling_probs(steep_mask, p_steep)
        assert len(probs) == 9
        assert math.isclose(sum(probs), 1.0, rel_tol=0.0, abs_tol=1e-12)
        steep_total = sum(w for w, flag in zip(probs, steep_mask) if flag)
        normal_total = sum(w for w, flag in zip(probs, steep_mask) if not flag)
        assert math.isclose(steep_total, p_steep, abs_tol=1e-12)
        assert math.isclose(normal_total, 1.0 - p_steep, abs_tol=1e-12)

    uniform = rtm.compute_block_sampling_probs([False] * 9, 0.45)
    assert all(math.isclose(w, 1.0 / 9.0, abs_tol=1e-12) for w in uniform)


def test_terrain_relative_orientation_alignment() -> None:
    cases = [
        (0.0, 0.0, 0.0),
        (0.30, 0.0, 0.0),
        (-0.30, 0.0, 0.0),
        (0.30, 0.0, math.pi / 2.0),
        (-0.30, 0.0, -math.pi / 2.0),
        (0.0, 0.30, 0.0),
        (0.24, -0.18, 0.75),
    ]
    for gx, gy, yaw in cases:
        roll = math.atan(gy * math.cos(yaw) - gx * math.sin(yaw))
        pitch = -math.atan(gx * math.cos(yaw) + gy * math.sin(yaw))
        zx, zy, zz = _rotated_z_axis(roll, pitch, yaw)
        nx, ny, nz = -gx, -gy, 1.0
        n_norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        dot = (zx * nx + zy * ny + zz * nz) / n_norm
        assert dot > 0.9999, f"misaligned normal for gx={gx}, gy={gy}, yaw={yaw}: dot={dot}"


def test_centripetal_accel_clamp(rtm) -> None:
    profiles = json.loads(PROFILES_PATH.read_text())
    for name in ("random_terrain_stage3_dirH2a_20261115", "random_terrain_stage3_dirH2b_yaw_20261115"):
        assert name in profiles, f"missing profile {name}"
        assert name in rtm.RANDOM_TERRAIN_PROFILES, f"unregistered profile {name}"
        assert profiles[name]["command_centripetal_accel_limit"] == 1.5

    cmds = [
        [0.65, 0.0, 1.5],   # |0.65 * 1.5| = 0.975 <= 1.5 (unchanged)
        [1.50, 0.0, 1.5],   # |1.50 * 1.5| = 2.25 > 1.5 -> 1.0
        [-1.20, 0.0, -1.5], # |-1.20 * -1.5| = 1.80 > 1.5 -> -1.25
        [2.00, 0.0, -1.5],  # |2.00 * -1.5| = 3.00 > 1.5 -> -0.75
    ]
    rtm.clamp_command_centripetal_accel(cmds, 1.5, env_ids=[0, 1, 2, 3])
    assert math.isclose(cmds[0][2], 1.5, abs_tol=1e-9)
    assert math.isclose(cmds[1][2], 1.0, abs_tol=1e-9)
    assert math.isclose(cmds[2][2], -1.25, abs_tol=1e-9)
    assert math.isclose(cmds[3][2], -0.75, abs_tol=1e-9)


def test_high_speed_command_boost(rtm) -> None:
    import random

    profiles = json.loads(PROFILES_PATH.read_text())
    dir_i4 = profiles["random_terrain_stage4_dirI4_20261116"]
    assert "random_terrain_stage4_dirI4_20261116" in rtm.RANDOM_TERRAIN_PROFILES
    frac = dir_i4["command_high_speed_fraction"]
    rng_lo, rng_hi = dir_i4["command_high_speed_range"]
    assert frac == 0.35 and rng_lo == 1.55 and rng_hi == 2.05

    cmds = [[0.0, 0.0, 0.0], [0.4, 0.0, 1.5], [0.6, 0.0, -1.5], [0.5, 0.0, 1.2]]
    standing = [True, False, False, False]
    rtm.boost_command_high_speed(
        cmds,
        1.0,
        (rng_lo, rng_hi),
        env_ids=[0, 1, 2, 3],
        is_standing=standing,
        rng=random.Random(42),
    )
    rtm.clamp_command_centripetal_accel(cmds, dir_i4["command_centripetal_accel_limit"], env_ids=[0, 1, 2, 3])
    assert cmds[0] == [0.0, 0.0, 0.0], "standing command must never be overwritten"
    for row in cmds[1:]:
        assert rng_lo <= row[0] <= rng_hi
        assert abs(row[0] * row[2]) <= 1.5 + 1e-9

    dir_i5 = profiles["random_terrain_stage4_dirI5_20261116"]
    assert "random_terrain_stage4_dirI5_20261116" in rtm.RANDOM_TERRAIN_PROFILES
    assert dir_i5["reward_parameters"]["lin_vel_xy_square_sigma"] == 1.0
    assert dir_i5["reward_weights"]["track_lin_vel_xy_square"] == -3.2
    assert dir_i5["random_terrain_stairs_fraction"] == 0.1
    assert dir_i5["random_terrain_transition_fraction"] == 0.12
    assert dir_i5["curriculum"]["steep_block_prob_end"] == 0.14

    dir_i6 = profiles["random_terrain_stage4_dirI6_20261116"]
    assert "random_terrain_stage4_dirI6_20261116" in rtm.RANDOM_TERRAIN_PROFILES
    assert dir_i6["command_high_speed_smooth_only"] is True
    eligible_cmds = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.2], [0.5, 0.0, 0.2], [0.5, 0.0, 0.2]]
    rtm.boost_command_high_speed(
        eligible_cmds,
        1.0,
        (1.6, 2.04),
        env_ids=[0, 1, 2, 3],
        is_standing=[True, False, False, False],
        eligible=[True, False, True, False],
        rng=random.Random(7),
    )
    assert eligible_cmds[0][0] == 0.0
    assert eligible_cmds[1][0] == 0.5 and eligible_cmds[3][0] == 0.5, "hazard envs must never receive high-speed boost"
    assert 1.6 <= eligible_cmds[2][0] <= 2.04


def main() -> int:
    rtm = _load_random_terrain_map()
    test_curriculum_progression_and_interpolation(rtm)
    test_block_sampling_probs(rtm)
    test_terrain_relative_orientation_alignment()
    test_centripetal_accel_clamp(rtm)
    test_high_speed_command_boost(rtm)
    print("stage3 curriculum contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
