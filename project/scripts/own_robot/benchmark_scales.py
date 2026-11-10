"""Measure actual PPO throughput and total device memory sequentially.

Each scale uses a fresh scratch policy and its own log/checkpoints. A zero
process status is insufficient: the final checkpoint must exist at the expected
iteration. These short runs never count toward the requested final training.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scales", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--iterations", type=int, default=10)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    results = []
    for n in args.scales:
        name = f"throughput_{n}"
        run_root = args.output / name
        command = ["bash", str(ROOT / "scripts/run_project.sh"), "scripts/rsl_rl/train.py",
                   "--task", "Robotics-Wheelbipe-Own-Flat-v0", "--num_envs", str(n),
                   "--max_iterations", str(args.iterations), "--seed", "42", "--headless",
                   "--device", "cuda:0", "--run_name", name]
        env = dict(os.environ, WHEELBIPE_PYTHON=sys.executable, WHEELBIPE_RUN_ROOT=str(run_root))
        row = {"num_envs": n, "iterations": args.iterations, "command": command,
               "max_gpu_memory_MiB": 0, "max_cgroup_memory_bytes": 0}
        started = time.monotonic()
        with (args.output / f"{name}.log").open("w") as f:
            child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            while child.poll() is None:
                sample = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True)
                if sample.returncode == 0:
                    row["max_gpu_memory_MiB"] = max(row["max_gpu_memory_MiB"], int(sample.stdout.strip().splitlines()[0]))
                cgroup = Path("/sys/fs/cgroup/memory.current")
                if cgroup.exists():
                    row["max_cgroup_memory_bytes"] = max(row["max_cgroup_memory_bytes"], int(cgroup.read_text()))
                # Stop only this benchmark process group before exhausting the
                # platform's 50GB RAM / 24GB GPU. Preserve the failed result.
                if row["max_gpu_memory_MiB"] > 23500 or row["max_cgroup_memory_bytes"] > 48 * 1024**3:
                    row["failure"] = "memory budget reached"
                    os.killpg(child.pid, signal.SIGTERM)
                    break
                if time.monotonic() - started > 900:
                    row["failure"] = "900 second benchmark deadline reached"
                    os.killpg(child.pid, signal.SIGTERM)
                    break
                time.sleep(2)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        row.update(returncode=child.returncode, wall_seconds=time.monotonic() - started)
        runs = list((run_root / "logs/rsl_rl/wheelbipe_own_flat").glob(f"*_{name}"))
        checkpoint = runs[0] / f"model_{args.iterations - 1}.pt" if len(runs) == 1 else None
        row["checkpoint"] = str(checkpoint) if checkpoint else None
        if checkpoint and checkpoint.exists() and child.returncode == 0 and "failure" not in row:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            row["status"] = "PASS" if saved["iter"] == args.iterations - 1 else "FAIL"
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events = EventAccumulator(str(runs[0])).Reload()
            row["scalar_tags"] = events.Tags()["scalars"]
            row["last_scalars"] = {tag: events.Scalars(tag)[-1].value for tag in row["scalar_tags"]}
            for tag in row["scalar_tags"]:
                if "total_fps" in tag:
                    values = events.Scalars(tag)
                    row["steady_env_steps_per_second"] = sum(x.value for x in values[2:]) / len(values[2:])
        else:
            row["status"] = "FAIL"
        results.append(row)
        (args.output / "benchmark.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k not in ("last_scalars", "scalar_tags", "command")}), flush=True)
        if row["status"] != "PASS":
            break


if __name__ == "__main__":
    main()
