#!/usr/bin/env python3
"""Offline unit tests for tools/eval_speed_sweep.py helpers."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TERRAIN_PATH = (
    ROOT
    / "project/source/agent_tasks/agent_tasks/direct/wheelbipe/"
    "wheelbipe_own/random_terrain_map.py"
)
SWEEP_PATH = ROOT / "tools/eval_speed_sweep.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    terrain = _load("rtm_for_sweep_test", TERRAIN_PATH)
    sweep = _load("eval_speed_sweep_test", SWEEP_PATH)

    layout = terrain.make_layout(terrain.LAYOUT_SEED)
    env_specs, block_ids = sweep.resolve_scenario_blocks(layout, repeats=3)
    assert len(env_specs) == len(sweep.SPEED_SWEEP_SCENARIOS) * 3
    assert len(block_ids) == len(env_specs)
    assert all(terrain.terrain_at_block(bid, layout) == spec["terrain"] for spec, bid in zip(env_specs, block_ids))

    # Constant vs e-stop command trajectories
    assert sweep.command_for_scenario({"mode": "const", "speed_m_s": 1.5, "yaw_rate_rad_s": -0.5}, 4.0) == (1.5, -0.5)
    estop_spec = {"mode": "estop", "speed_m_s": 1.5, "brake_time_s": 2.5}
    assert sweep.command_for_scenario(estop_spec, 2.49) == (1.5, 0.0)
    assert sweep.command_for_scenario(estop_spec, 2.50) == (0.0, 0.0)

    # S-turn and rapid reversal modes + highspeed suite
    sturn_spec = {"mode": "sturn", "speed_m_s": 1.5, "yaw_rate_rad_s": 1.0, "alternating_yaw_period_s": 2.0}
    assert sweep.command_for_scenario(sturn_spec, 0.50) == (1.5, 1.0)
    assert sweep.command_for_scenario(sturn_spec, 1.50) == (1.5, -1.0)
    rev_spec = {"mode": "reversal", "speed_m_s": 1.5, "brake_time_s": 2.5, "reversal_speed_m_s": -0.8}
    assert sweep.command_for_scenario(rev_spec, 2.49) == (1.5, 0.0)
    assert sweep.command_for_scenario(rev_spec, 2.50) == (-0.8, 0.0)
    hs_specs, hs_blocks = sweep.resolve_scenario_blocks(layout, repeats=3, scenarios=sweep.SUITES["highspeed"])
    assert len(hs_specs) == len(sweep.HIGHSPEED_MANEUVER_SCENARIOS) * 3
    assert len(hs_blocks) == len(hs_specs)

    # Slope-relative roll expectation: climbing straight along +x has zero expected roll,
    # while turning +90 deg across a +20% x-slope tilts the axle roll by -atan(0.20).
    assert abs(float(sweep.expected_terrain_roll_rad(0.20, 0.0, 0.0))) < 1e-12
    assert abs(float(sweep.expected_terrain_roll_rad(0.20, 0.0, math.pi / 2.0)) + math.atan(0.20)) < 1e-12

    print("speed sweep helper contract: PASS")


if __name__ == "__main__":
    main()
