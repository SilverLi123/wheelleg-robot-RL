#!/usr/bin/env python3
"""Fast checks for forward-only legacy terrain curriculum decisions."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "project/source/agent_tasks/agent_tasks/direct/wheelbipe/"
    "wheelbipe_own/terrain_curriculum.py"
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("terrain_curriculum_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    up, down = module.terrain_curriculum_decisions(3.0, 2.0, False)
    assert up is True and down is False

    # Backward and lateral motion have zero or negative forward projection.
    up, down = module.terrain_curriculum_decisions(5.0, -3.0, False)
    assert up is False and down is False
    up, down = module.terrain_curriculum_decisions(5.0, 0.0, False)
    assert up is False and down is False

    # Distance alone cannot promote a one-second episode. This is the regression
    # guard for reading episode_length_buf after DirectRLEnv has cleared it.
    up, down = module.terrain_curriculum_decisions(1.0, 3.0, False)
    assert up is False and down is False

    # Early physical failure demotes; later failures do not create an extra
    # downward step because they have already demonstrated basic survival.
    up, down = module.terrain_curriculum_decisions(1.9, 0.2, True)
    assert up is False and down is True
    up, down = module.terrain_curriculum_decisions(3.0, 3.0, True)
    assert up is False and down is False
    print("legacy terrain curriculum contract: PASS")


if __name__ == "__main__":
    main()
