#!/usr/bin/env python3
"""Checks for the standalone recording review page."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/video_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "video_review_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_module()

    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "route_demo"
        run_dir.mkdir()
        (run_dir / "policy.mp4").write_bytes(b"not-a-real-video")
        summary = {
            "status": "PASS",
            "steps": 6000,
            "sim_seconds": 120.0,
            "resets": 0,
            "max_closure_m": 0.001164,
            "mean_abs_vx_error_m_s": 0.0152,
            "route": {
                "waypoints_total": 9,
                "waypoints_reached": 9,
                "completed": True,
            },
            "error": "private traceback that should not enter the viewer",
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))

        output = module.write_viewer(run_dir)
        page = output.read_text()
        assert output.name == "review.html"
        assert 'src="policy.mp4"' in page
        assert 'id="stage"' in page
        assert 'id="zoom-in"' in page
        assert 'id="timeline"' in page
        assert '"steps":6000' in page
        assert '"waypoints_reached":9' in page
        assert "private traceback" not in page
        assert "http://" not in page and "https://" not in page

        try:
            module.render_viewer(
                run_dir,
                video_filename="missing.mp4",
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing video must be rejected")

    print("video review HTML contract: PASS")


if __name__ == "__main__":
    main()
