#!/usr/bin/env python3
"""Unit tests for tools/train_watchdog.py using dummy subprocesses."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_watchdog  # noqa: E402


class TrainWatchdogTests(unittest.TestCase):
    def test_normal_completion_and_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "ok_run"
            rc = train_watchdog.run_watchdog(
                [
                    sys.executable,
                    "-c",
                    "import time; print('Learning iteration 0/2', flush=True); time.sleep(0.15); "
                    "print('Learning iteration 1/2', flush=True)",
                ],
                run_dir,
                name="dummy_ok",
                gpu="0",
                poll_interval_s=0.05,
                init_timeout_s=2.0,
                stall_timeout_s=2.0,
            )
            self.assertEqual(rc, 0)
            state = json.loads((run_dir / "watchdog_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["exit_code"], 0)
            self.assertEqual(state["last_iter"], 1)
            self.assertEqual(state["total_iters"], 2)
            self.assertIsNotNone(state["updated_at"])

    def test_nonfinite_failure_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "nan_run"
            rc = train_watchdog.run_watchdog(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('Learning iteration 5/10', flush=True); "
                    "print('FloatingPointError: Nonfinite root_state_w', flush=True); sys.exit(1)",
                ],
                run_dir,
                name="dummy_nan",
                gpu="1",
                poll_interval_s=0.05,
            )
            self.assertEqual(rc, 1)
            state = json.loads((run_dir / "watchdog_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed_nonfinite")
            self.assertEqual(state["last_iter"], 5)

    def test_stall_timeout_abort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "stall_run"
            rc = train_watchdog.run_watchdog(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                run_dir,
                name="dummy_stall",
                gpu="2",
                poll_interval_s=0.05,
                init_timeout_s=0.25,
                stall_timeout_s=0.25,
            )
            self.assertNotEqual(rc, 0)
            state = json.loads((run_dir / "watchdog_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "aborted_stall_timeout")


if __name__ == "__main__":
    unittest.main(verbosity=2)
