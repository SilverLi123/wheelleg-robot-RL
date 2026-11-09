"""Run the authorized 3000-iteration job after reviewed cloud validation.

Launch this process inside tmux. State and logs remain on the persistent volume.
This does not turn a successful subprocess exit into a walking-performance claim.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-dir", type=Path, required=True)
    ap.add_argument("--validation-report", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--video-steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.iterations <= 0 or args.num_envs <= 0 or args.video_steps <= 0:
        ap.error("iteration, environment and video-step counts must be positive")
    proof = json.loads(args.validation_report.read_text())
    if proof.get("approved_for_training") is not True:
        raise RuntimeError("Review reference/Own physics and small PPO before starting the long job")
    job = args.job_dir.resolve()
    job.mkdir(parents=True, exist_ok=True)
    if (job / "state.json").exists():
        raise FileExistsError("Use a new job directory; existing jobs are never overwritten")
    (job / "validation_report.json").write_text(json.dumps(proof, indent=2))
    state = {"created_at": timestamp(), "iterations_requested": args.iterations, "num_envs": args.num_envs,
             "seed": args.seed, "task": "Robotics-Wheelbipe-Own-Flat-v0", "job_dir": str(job)}
    def update(stage, **extra):
        state.update(stage=stage, updated_at=timestamp(), **extra)
        tmp = job / "state.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(job / "state.json")
        print(json.dumps(state, indent=2), flush=True)
    env = dict(os.environ, WHEELBIPE_PYTHON=sys.executable, WHEELBIPE_RUN_ROOT=str(job / "run"),
               WHEELBIPE_DIAGNOSTICS_DIR=str(job / "diagnostics"), TZ="Asia/Shanghai")
    def run(script, argv, log_name):
        command = ["bash", str(ROOT / "scripts/run_project.sh"), script, *map(str, argv)]
        (job / (log_name + ".command.json")).write_text(json.dumps(command, indent=2))
        with (job / (log_name + ".log")).open("w") as f:
            subprocess.run(command, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    try:
        update("PREFLIGHT")
        run("scripts/own_robot/preflight.py", ["--strict", "--output", job / "environment.json"], "preflight")
        with (job / "requirements.actual.txt").open("w") as f:
            subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], stdout=f, check=True)
        update("TRAINING")
        run("scripts/rsl_rl/train.py", ["--task", state["task"], "--num_envs", args.num_envs,
            "--max_iterations", args.iterations, "--seed", args.seed, "--device", "cuda:0", "--headless",
            "--run_name", "authorized_3000"], "train")
        runs = list((job / "run/logs/rsl_rl/wheelbipe_own_flat").glob("*_authorized_3000"))
        if len(runs) != 1:
            raise RuntimeError(f"Expected one isolated training run, found {runs}")
        checkpoint = runs[0] / f"model_{args.iterations - 1}.pt"
        if not checkpoint.is_file():
            raise RuntimeError("Training exited without the requested final checkpoint")
        import torch
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(saved.get("iter", -1)) != args.iterations - 1:
            raise RuntimeError("Final checkpoint iteration is incorrect")
        if not saved.get("optimizer_state_dict", {}).get("state"):
            raise RuntimeError("Final checkpoint has no optimizer history")
        for name, value in saved["model_state_dict"].items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite final model tensor: " + name)
        update("RECORDING", checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
               checkpoint_iteration=int(saved["iter"]))
        del saved
        run("scripts/rsl_rl/play.py", ["--task", "Robotics-Wheelbipe-Own-Flat-Play-v0",
            "--num_envs", 1, "--checkpoint", checkpoint, "--seed", args.seed, "--device", "cuda:0",
            "--headless", "--video", "--follow_camera", "--telemetry", "--video_length", args.video_steps,
            "--max_steps", args.video_steps], "play")
        candidates = list((job / "run/logs/audit").glob("*/videos/play/*.mp4"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one recorded video, found {candidates}")
        video = job / "own_after_3000.mp4"
        shutil.copy2(candidates[0], video)
        play_audit = candidates[0].parents[2]
        for name in ("rollout.jsonl", "rollout_summary.json"):
            shutil.copy2(play_audit / name, job / name)
        if video.stat().st_size < 1024:
            raise RuntimeError("Recorded video is empty")
        metadata = {"source": str(candidates[0]), "bytes": video.stat().st_size, "sha256": digest(video)}
        if shutil.which("ffprobe"):
            r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration", "-of", "json", str(video)],
                capture_output=True, text=True, check=True)
            metadata["ffprobe"] = json.loads(r.stdout)
            streams = metadata["ffprobe"].get("streams", [])
            if not streams or int(streams[0].get("nb_read_frames", 0)) < args.video_steps - 2:
                raise RuntimeError("Video contains fewer frames than requested")
        # Decode representative frames even when ffprobe is unavailable.
        import imageio.v2 as imageio
        reader = imageio.get_reader(str(video), format="ffmpeg")
        try:
            frame_count = reader.count_frames()
            if frame_count < args.video_steps - 2:
                raise RuntimeError("Recorded video is truncated")
            metadata["decoded_frames"] = []
            for index in (0, frame_count // 2, frame_count - 1):
                frame = reader.get_data(index)
                if frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    raise RuntimeError("Invalid decoded video frame")
                metadata["decoded_frames"].append({"index": index, "shape": list(frame.shape),
                                                   "mean": float(frame.mean()), "std": float(frame.std())})
            metadata["frame_count"] = frame_count
        finally:
            reader.close()
        (job / "video.json").write_text(json.dumps(metadata, indent=2))
        update("COMPLETE", video=str(video), video_metadata=metadata,
               performance_assessment="Review training metrics and actual recorded behavior; completion alone is not successful walking")
    except BaseException as error:
        update("FAILED", error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
