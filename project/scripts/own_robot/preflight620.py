"""Verify the isolated Sim 6.2 / Lab 3 / original PPO runtime before launch."""
import importlib.metadata as md
import importlib.util
import json
import sys
from pathlib import Path


def require_target():
    import torch
    expected = {"isaacsim": "6.2.0.0", "torch": "2.11.0", "rsl-rl-lib": "3.0.1", "warp-lang": "1.17.0"}
    actual = {name: md.version(name) for name in expected}
    errors = [f"{name}: expected {version}, got {actual[name]}" for name, version in expected.items()
              if actual[name].split("+")[0] != version]
    if sys.version_info[:2] != (3, 12):
        errors.append("Python 3.12 required")
    spec = importlib.util.find_spec("isaaclab")
    lab = Path(next(iter(spec.submodule_search_locations))).parents[2]
    if (lab / "VERSION").read_text().strip() != "3.0.0":
        errors.append("Isaac Lab v3.0.0 required")
    if torch.version.cuda != "13.0" or not torch.cuda.is_available():
        errors.append("Working CUDA 13 PyTorch required")
    if errors:
        raise RuntimeError("; ".join(errors))
    print("[PREFLIGHT620] " + json.dumps({"packages": actual, "lab": str(lab), "torch_cuda": torch.version.cuda}))
    return actual
