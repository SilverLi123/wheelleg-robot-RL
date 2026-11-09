"""Read existing target environment without installing or replacing anything."""
import argparse
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as ex:
        return {"error": str(ex)}


def collect():
    packages = {}
    for name in ("isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch", "torchvision", "gymnasium",
                 "numpy", "tensordict", "hydra-core", "pytorch-kinematics", "tensorboard"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    data = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages,
            "os_release": Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else None,
            "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]),
            "vulkan": command(["vulkaninfo", "--summary"]), "simulator_start": "NOT RUN"}
    errors = []
    if sys.version_info[:2] != (3, 11):
        errors.append("Target requires Python 3.11")
    if platform.machine() != "x86_64":
        errors.append("This deployment profile targets x86_64")
    # Official binary installations expose isaacsim through python_packages and
    # do not necessarily contain pip distribution metadata.
    sim_version = None
    candidates = [Path(os.environ[n]) for n in ("ISAAC_PATH", "ISAACSIM_PATH") if os.environ.get(n)]
    try:
        sim_spec = importlib.util.find_spec("isaacsim")
        if sim_spec and sim_spec.origin:
            candidates += list(Path(sim_spec.origin).resolve().parents)[:4]
    except (ImportError, ValueError):
        pass
    for p in candidates:
        if (p / "VERSION").is_file():
            sim_version = (p / "VERSION").read_text().strip()
            data["isaac_sim_binary_version_file"] = str(p / "VERSION")
            break
    data["isaac_sim_binary_version"] = sim_version
    if packages["isaacsim"] is not None:
        if packages["isaacsim"].split("+")[0] not in ("5.1.0", "5.1.0.0"):
            errors.append("Expected Isaac Sim5.1, got pip package " + packages["isaacsim"])
    elif not sim_version or not sim_version.startswith("5.1.0"):
        errors.append("Cannot verify Isaac Sim5.1 from pip metadata or binary VERSION")
    if sim_version and not sim_version.startswith("5.1.0"):
        errors.append("Conflicting Isaac Sim binary: " + sim_version)
    for name, expected in (("rsl-rl-lib", "3.0.1"), ("torch", "2.7.0"),
                           ("torchvision", "0.22.0")):
        if not packages[name] or packages[name].split("+")[0] not in (expected, expected + ".0"):
            errors.append(f"Expected {name}=={expected}, got {packages[name]}")
    try:
        spec = importlib.util.find_spec("isaaclab")
        source = Path(spec.submodule_search_locations[0])
        versions = [p / "VERSION" for p in (source, *list(source.parents)[:4]) if (p / "VERSION").exists()]
        data["isaaclab_source"] = str(source)
        data["isaaclab_release"] = versions[0].read_text().strip() if versions else None
        if data["isaaclab_release"] != "2.3.0":
            errors.append("Isaac Lab repository must be the reviewed v2.3.0 tag")
    except (ImportError, AttributeError, TypeError, IndexError):
        errors.append("Cannot locate Isaac Lab source/release")
    try:
        import torch
        data["torch_cuda"] = torch.version.cuda
        data["torch_cuda_available"] = torch.cuda.is_available()
        if torch.version.cuda != "12.8" or not torch.cuda.is_available():
            errors.append("Need working torch 2.7.0 cu128; driver CUDA string/toolkit need not match")
    except ImportError:
        errors.append("Cannot import torch")
    data["errors"] = errors
    return data


def require_target():
    data = collect()
    if data["errors"]:
        raise RuntimeError("Target environment mismatch: " + "; ".join(data["errors"]))
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("environment.json"))
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    data = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))
    if args.strict and data["errors"]:
        raise SystemExit(1)
