#!/usr/bin/env python3
"""Lightweight persistent training watchdog with heartbeat, PID tracking, and failure classification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Sequence


ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
NONFINITE_RE = re.compile(
    r"(FloatingPointError:\s*Nonfinite|nan\b|inf\b.*loss|Nonfinite .* before observation)",
    re.IGNORECASE,
)
OOM_RE = re.compile(r"(CUDA out of memory|OutOfMemoryError|Killed\s*$)", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def classify_failure(log_text: str, exit_code: int) -> str:
    """Classify non-zero exit into nonfinite, oom, or generic error."""
    if exit_code == 0:
        return "completed"
    if NONFINITE_RE.search(log_text):
        return "failed_nonfinite"
    if OOM_RE.search(log_text):
        return "failed_oom"
    return "failed_error"


def parse_last_iteration(log_text: str) -> tuple[int | None, int | None]:
    matches = ITER_RE.findall(log_text)
    if not matches:
        return None, None
    cur, total = matches[-1]
    return int(cur), int(total)


def write_state(state_path: Path, payload: dict) -> None:
    tmp_path = state_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(state_path)


def run_watchdog(
    cmd: Sequence[str],
    run_dir: Path,
    *,
    name: str,
    gpu: str = "",
    poll_interval_s: float = 10.0,
    init_timeout_s: float = 900.0,
    stall_timeout_s: float = 300.0,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    state_path = run_dir / "watchdog_state.json"

    started_at = _utc_now()
    start_mono = time.monotonic()

    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            list(cmd),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(run_dir),
            start_new_session=True,
        )

    state = {
        "name": name,
        "gpu": gpu,
        "cmd": list(cmd),
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "watcher_pid": os.getpid(),
        "child_pid": proc.pid,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "status": "running",
        "exit_code": None,
        "last_iter": None,
        "total_iters": None,
        "last_log_mtime": None,
    }
    write_state(state_path, state)

    last_log_size = 0
    last_progress_mono = start_mono
    aborted_reason: str | None = None

    while True:
        rc = proc.poll()
        now_mono = time.monotonic()

        log_text = ""
        if log_path.exists():
            stat = log_path.stat()
            state["last_log_mtime"] = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            if stat.st_size != last_log_size:
                last_log_size = stat.st_size
                last_progress_mono = now_mono
            log_text = log_path.read_text(encoding="utf-8", errors="replace")[-32768:]
            cur_iter, total_iters = parse_last_iteration(log_text)
            if cur_iter is not None:
                state["last_iter"] = cur_iter
                state["total_iters"] = total_iters

        if rc is not None:
            state["exit_code"] = rc
            state["updated_at"] = _utc_now()
            state["status"] = aborted_reason or classify_failure(log_text, rc)
            write_state(state_path, state)
            return rc

        timeout = init_timeout_s if state["last_iter"] is None else stall_timeout_s
        if (now_mono - last_progress_mono) > timeout:
            aborted_reason = "aborted_stall_timeout"
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            state["exit_code"] = proc.returncode if proc.returncode is not None else -15
            state["status"] = aborted_reason
            state["updated_at"] = _utc_now()
            write_state(state_path, state)
            return int(state["exit_code"])

        state["updated_at"] = _utc_now()
        write_state(state_path, state)
        time.sleep(poll_interval_s)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent training watchdog")
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--init-timeout", type=float, default=900.0)
    parser.add_argument("--stall-timeout", type=float, default=300.0)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("Command after -- is required")
    return run_watchdog(
        cmd,
        Path(args.run_dir),
        name=args.name,
        gpu=args.gpu,
        poll_interval_s=args.poll_interval,
        init_timeout_s=args.init_timeout,
        stall_timeout_s=args.stall_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
