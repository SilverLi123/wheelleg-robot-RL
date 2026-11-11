#!/usr/bin/env python3
"""Install the mixed-slope development files into a copied training workspace.

This script edits only the workspace containing it.  It refuses to run when the
expected base sources are absent, making accidental execution against a shared
team checkout fail before any file is changed.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OWN = ROOT / "project/source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_own"
EXPERIMENT = OWN / "experiment.py"
PROFILES = ROOT / "profiles.json"


def patch_experiment() -> None:
    text = EXPERIMENT.read_text()
    if "mixed_slopes" not in text.split("PROFILE_FILE =", 1)[0]:
        slope_import = "from . import slope_pilot\n"
        if text.count(slope_import) == 1:
            text = text.replace(
                slope_import,
                "from . import slope_pilot, mixed_slopes\n",
            )
        else:
            anchor = "from .runner_cfg import WheelbipeOwnPPORunnerCfg\n"
            if text.count(anchor) != 1:
                raise RuntimeError("Expected one runner-config import anchor")
            text = text.replace(anchor, anchor + "from . import mixed_slopes\n")

    if "        mixed_slopes.configure(self)\n" not in text:
        anchors = (
            "        slope_pilot.configure(self)\n",
            "        configure_unified(self, p)\n",
        )
        for anchor in anchors:
            if text.count(anchor) == 1:
                text = text.replace(
                    anchor,
                    anchor + "        mixed_slopes.configure(self)\n",
                )
                break
        else:
            raise RuntimeError("Expected one terrain configuration anchor")

    old_condition = (
        "if self.cfg.experiment_profile_settings.get('terrain_mix') "
        "and hasattr(self, '_slew_leg'):"
    )
    new_condition = (
        "if (self.cfg.experiment_profile_settings.get('terrain_mix') "
        "or self.cfg.experiment_profile_settings.get('mixed_slopes')) "
        "and hasattr(self, '_slew_leg'):"
    )
    already_handles_mixed = (
        new_condition in text
        or (
            "experiment_profile_settings.get('terrain_mix')" in text
            and "experiment_profile_settings.get('mixed_slopes')" in text
        )
    )
    if not already_handles_mixed:
        if text.count(old_condition) != 1:
            raise RuntimeError("Expected exactly one terrain progression condition")
        text = text.replace(old_condition, new_condition)

    save_anchor = "        if self.experiment_stage == 3:\n"
    save_override = (
        "        save_interval = os.environ.get('WHEELBIPE_SAVE_INTERVAL')\n"
        "        if save_interval is not None:\n"
        "            self.save_interval = int(save_interval)\n"
        "            if self.save_interval <= 0:\n"
        "                raise ValueError('WHEELBIPE_SAVE_INTERVAL must be positive')\n\n"
    )
    if save_override not in text:
        if text.count(save_anchor) != 1:
            raise RuntimeError("Expected exactly one stage-3 runner branch")
        text = text.replace(save_anchor, save_override + save_anchor)

    learning_override = (
        "        learning_rate = os.environ.get('WHEELBIPE_LEARNING_RATE')\n"
        "        if learning_rate is not None:\n"
        "            value = float(learning_rate)\n"
        "            if value <= 0.0:\n"
        "                raise ValueError('WHEELBIPE_LEARNING_RATE must be positive')\n"
        "            self.algorithm.learning_rate = value\n"
    )
    if learning_override not in text:
        anchor = save_override
        if text.count(anchor) != 1:
            raise RuntimeError("Expected one save-interval override block")
        text = text.replace(anchor, anchor + "\n" + learning_override)
    EXPERIMENT.write_text(text)


def patch_profile() -> None:
    data = json.loads(PROFILES.read_text())
    if "mixed_slopes_20261111" in data:
        return
    if "flat_seed42" not in data:
        raise RuntimeError("flat_seed42 control profile is missing")
    profile = json.loads(json.dumps(data["flat_seed42"]))
    profile.update(
        {
            "command_speed": 0.65,
            "command_yaw": 0.45,
            "command_resampling": [2.0, 5.0],
            "terrain_mix": False,
            "terrain_family": "mixed_slopes",
            "mixed_slopes": True,
            "terrain_seed": 20261111,
            "terrain_version": "mixed_slopes_straight_v1",
            "terrain_generation": (
                "three flat columns plus six/twelve/fourteen/eighteen percent "
                "straight up and down ramps; five-row distance curriculum"
            ),
        }
    )
    # Interface, physics, effort limits and randomization are copied unchanged.
    data["mixed_slopes_20261111"] = profile
    PROFILES.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    if not EXPERIMENT.is_file() or not PROFILES.is_file():
        raise SystemExit(f"Not a compatible wheelbipe workspace: {ROOT}")
    patch_experiment()
    patch_profile()
    print("Installed mixed_slopes_20261111 profile and experiment hooks.")
