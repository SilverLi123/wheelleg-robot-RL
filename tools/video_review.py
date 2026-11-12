#!/usr/bin/env python3
"""Create a standalone pan-and-zoom review page for a recording directory."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _summary_metadata(run_dir: Path) -> dict:
    """Read only the small set of run fields needed by the review page."""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    fields = (
        "status",
        "steps",
        "sim_seconds",
        "resets",
        "reset_causes",
        "max_closure_m",
        "checkpoint_iteration",
        "checkpoint_sha256",
        "profile",
        "experiment_profile",
        "mean_abs_vx_error_m_s",
    )
    metadata = {
        key: summary[key]
        for key in fields
        if key in summary
    }
    if "error" in summary:
        metadata["has_error"] = True
    if "route" in summary:
        metadata["route"] = summary["route"]
    return metadata


def render_viewer(
    run_dir: Path,
    *,
    video_filename: str = "policy.mp4",
) -> str:
    """Return the complete HTML document for one recording directory."""
    run_dir = run_dir.resolve()
    video_path = run_dir / video_filename
    if not video_path.is_file():
        raise FileNotFoundError(f"recording video not found: {video_path}")

    metadata = _summary_metadata(run_dir)
    metadata["run_directory"] = str(run_dir)
    metadata["video_file"] = video_filename
    metadata_json = json.dumps(
        metadata,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    title = html.escape(run_dir.name)
    video_source = html.escape(video_filename, quote=True)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} - Wheelbipe replay</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b0f14;
      --panel: #121923;
      --line: #283445;
      --text: #edf4fb;
      --muted: #93a5b8;
      --accent: #5fd1c7;
      --danger: #ff7b7b;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; background: var(--bg); color: var(--text); }}
    body {{
      font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      padding: 22px;
    }}
    .shell {{ width: min(1500px, 100%); margin: 0 auto; }}
    header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 14px;
    }}
    h1 {{ margin: 0 0 5px; font-size: clamp(20px, 3vw, 30px); font-weight: 650; }}
    .path {{ color: var(--muted); overflow-wrap: anywhere; }}
    .badge {{
      flex: 0 0 auto;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .badge[data-status="PASS"] {{ color: var(--accent); border-color: #2b716b; }}
    .badge[data-status="FAIL"], .badge[data-status="ERROR"] {{
      color: var(--danger);
      border-color: #743f47;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .metric {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 9px;
      background: var(--panel);
    }}
    .metric strong {{ display: block; margin-top: 2px; font-size: 17px; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .viewer {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #05080c;
      box-shadow: 0 24px 80px rgb(0 0 0 / 35%);
    }}
    #stage {{
      position: relative;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      cursor: grab;
      touch-action: none;
      user-select: none;
    }}
    #stage.dragging {{ cursor: grabbing; }}
    #surface {{
      position: absolute;
      inset: 0;
      transform-origin: 0 0;
      will-change: transform;
    }}
    video {{
      width: 100%;
      height: 100%;
      display: block;
      object-fit: contain;
      pointer-events: none;
    }}
    .hint {{
      position: absolute;
      top: 12px;
      left: 12px;
      padding: 7px 10px;
      border: 1px solid rgb(255 255 255 / 12%);
      border-radius: 8px;
      background: rgb(8 12 18 / 78%);
      color: #d5e1ed;
      pointer-events: none;
      font-size: 12px;
    }}
    .video-error {{
      position: absolute;
      inset: 0;
      display: none;
      place-items: center;
      padding: 28px;
      color: var(--danger);
      text-align: center;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto auto minmax(130px, 1fr) auto auto;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }}
    button, select {{
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #182231;
      color: var(--text);
      min-height: 34px;
      padding: 5px 10px;
      font: inherit;
    }}
    button:hover, select:hover {{ border-color: #4d647d; }}
    button:focus-visible, select:focus-visible, input:focus-visible {{
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    .time {{ min-width: 118px; color: var(--muted); text-align: right; }}
    .zoom-controls {{
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 12px 0 0;
    }}
    .zoom-controls button {{ min-width: 40px; }}
    .zoom-value {{ min-width: 54px; color: var(--muted); text-align: center; }}
    .note {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 760px) {{
      body {{ padding: 12px; }}
      header {{ display: block; }}
      .badge {{ display: inline-block; margin-top: 9px; }}
      .controls {{
        grid-template-columns: auto minmax(0, 1fr) auto;
      }}
      .controls select {{ grid-column: 1 / -1; }}
      .hint {{ display: none; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>Wheelbipe replay</h1>
        <div class="path" id="run-path">{title}</div>
      </div>
      <div class="badge" id="status">UNKNOWN</div>
    </header>

    <section class="metrics" id="metrics" aria-label="Run metrics"></section>

    <section class="viewer" aria-label="Video review player">
      <div id="stage">
        <div id="surface">
          <video id="video" src="{video_source}" preload="metadata" playsinline></video>
        </div>
        <div class="hint">滚轮缩放 · 拖拽平移 · 双击复位</div>
        <div class="video-error" id="video-error">
          无法加载 {video_source}。请保持本页面与视频文件在同一目录。
        </div>
      </div>
      <div class="controls">
        <button id="play" type="button" aria-label="播放或暂停">播放</button>
        <span class="time" id="current-time">0:00 / 0:00</span>
        <input id="timeline" type="range" min="0" max="1000" value="0" aria-label="视频进度">
        <select id="speed" aria-label="播放速度">
          <option value="0.25">0.25×</option>
          <option value="0.5">0.5×</option>
          <option value="1" selected>1×</option>
          <option value="2">2×</option>
        </select>
        <button id="fullscreen" type="button">全屏</button>
      </div>
    </section>

    <div class="zoom-controls" aria-label="缩放控制">
      <button id="zoom-out" type="button" aria-label="缩小">−</button>
      <span class="zoom-value" id="zoom-value">100%</span>
      <button id="zoom-in" type="button" aria-label="放大">+</button>
      <button id="reset-view" type="button">复位视角</button>
    </div>
    <p class="note">
      此页面可缩放和移动已经录制的画面，不会重新计算 3D 相机。
      快捷键：空格播放/暂停，+/- 缩放，0 复位。
    </p>
  </main>

  <script id="run-metadata" type="application/json">{metadata_json}</script>
  <script>
    (() => {{
      const meta = JSON.parse(document.getElementById("run-metadata").textContent);
      const video = document.getElementById("video");
      const stage = document.getElementById("stage");
      const surface = document.getElementById("surface");
      const playButton = document.getElementById("play");
      const timeline = document.getElementById("timeline");
      const currentTime = document.getElementById("current-time");
      const speed = document.getElementById("speed");
      const zoomValue = document.getElementById("zoom-value");
      const status = document.getElementById("status");

      const state = {{
        scale: 1,
        x: 0,
        y: 0,
        dragging: false,
        pointerId: null,
        lastX: 0,
        lastY: 0,
      }};

      const formatTime = (seconds) => {{
        if (!Number.isFinite(seconds)) return "0:00";
        const whole = Math.max(0, Math.floor(seconds));
        const minutes = Math.floor(whole / 60);
        return `${{minutes}}:${{String(whole % 60).padStart(2, "0")}}`;
      }};

      const formatValue = (value, digits = 3) => {{
        if (typeof value === "number" && Number.isFinite(value)) {{
          return Number(value).toFixed(digits);
        }}
        return String(value);
      }};

      const metric = (label, value) => {{
        const item = document.createElement("div");
        item.className = "metric";
        const caption = document.createElement("span");
        caption.textContent = label;
        const content = document.createElement("strong");
        content.textContent = value;
        item.append(caption, content);
        return item;
      }};

      const metrics = document.getElementById("metrics");
      if (meta.steps !== undefined) metrics.append(metric("策略步数", meta.steps));
      if (meta.sim_seconds !== undefined) {{
        metrics.append(metric("仿真时长", `${{formatValue(meta.sim_seconds, 2)}} s`));
      }}
      if (meta.resets !== undefined) metrics.append(metric("重置次数", meta.resets));
      if (meta.max_closure_m !== undefined) {{
        metrics.append(metric("最大闭合误差", `${{formatValue(meta.max_closure_m * 1000)}} mm`));
      }}
      if (meta.mean_abs_vx_error_m_s !== undefined) {{
        metrics.append(metric(
          "平均 vx 误差",
          `${{formatValue(meta.mean_abs_vx_error_m_s, 4)}} m/s`,
        ));
      }}
      if (meta.route?.waypoints_reached !== undefined) {{
        metrics.append(metric(
          "路线完成",
          `${{meta.route.waypoints_reached}}/${{meta.route.waypoints_total}}`,
        ));
      }}
      const runStatus = meta.status || "UNKNOWN";
      status.textContent = runStatus;
      status.dataset.status = runStatus;
      document.getElementById("run-path").textContent = meta.run_directory || meta.video_file;

      const renderTransform = () => {{
        surface.style.transform =
          `translate3d(${{state.x}}px, ${{state.y}}px, 0) scale(${{state.scale}})`;
        zoomValue.textContent = `${{Math.round(state.scale * 100)}}%`;
      }};

      const setScale = (nextScale, anchorX = stage.clientWidth / 2, anchorY = stage.clientHeight / 2) => {{
        const clamped = Math.min(8, Math.max(1, nextScale));
        const originX = (anchorX - state.x) / state.scale;
        const originY = (anchorY - state.y) / state.scale;
        state.x = anchorX - originX * clamped;
        state.y = anchorY - originY * clamped;
        state.scale = clamped;
        if (state.scale === 1) {{
          state.x = 0;
          state.y = 0;
        }}
        renderTransform();
      }};

      const resetView = () => {{
        state.scale = 1;
        state.x = 0;
        state.y = 0;
        renderTransform();
      }};

      stage.addEventListener("wheel", (event) => {{
        event.preventDefault();
        const rect = stage.getBoundingClientRect();
        const factor = Math.exp(-event.deltaY * 0.0012);
        setScale(
          state.scale * factor,
          event.clientX - rect.left,
          event.clientY - rect.top,
        );
      }}, {{ passive: false }});

      stage.addEventListener("pointerdown", (event) => {{
        if (state.scale <= 1) return;
        state.dragging = true;
        state.pointerId = event.pointerId;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        stage.classList.add("dragging");
        stage.setPointerCapture(event.pointerId);
      }});

      stage.addEventListener("pointermove", (event) => {{
        if (!state.dragging || event.pointerId !== state.pointerId) return;
        state.x += event.clientX - state.lastX;
        state.y += event.clientY - state.lastY;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        renderTransform();
      }});

      const endDrag = () => {{
        state.dragging = false;
        state.pointerId = null;
        stage.classList.remove("dragging");
      }};
      stage.addEventListener("pointerup", endDrag);
      stage.addEventListener("pointercancel", endDrag);
      stage.addEventListener("dblclick", resetView);

      document.getElementById("zoom-in").addEventListener("click", () => {{
        setScale(state.scale * 1.35);
      }});
      document.getElementById("zoom-out").addEventListener("click", () => {{
        setScale(state.scale / 1.35);
      }});
      document.getElementById("reset-view").addEventListener("click", resetView);

      const togglePlayback = async () => {{
        if (video.paused) {{
          await video.play();
        }} else {{
          video.pause();
        }}
      }};
      playButton.addEventListener("click", togglePlayback);
      video.addEventListener("play", () => {{ playButton.textContent = "暂停"; }});
      video.addEventListener("pause", () => {{ playButton.textContent = "播放"; }});
      video.addEventListener("timeupdate", () => {{
        const duration = Number.isFinite(video.duration) ? video.duration : 0;
        const progress = duration > 0 ? (video.currentTime / duration) * 1000 : 0;
        timeline.value = String(progress);
        currentTime.textContent =
          `${{formatTime(video.currentTime)}} / ${{formatTime(duration)}}`;
      }});
      video.addEventListener("loadedmetadata", () => {{
        currentTime.textContent = `0:00 / ${{formatTime(video.duration)}}`;
      }});
      video.addEventListener("error", () => {{
        document.getElementById("video-error").style.display = "grid";
      }});
      video.addEventListener("click", togglePlayback);

      timeline.addEventListener("input", () => {{
        if (Number.isFinite(video.duration)) {{
          video.currentTime = (Number(timeline.value) / 1000) * video.duration;
        }}
      }});
      speed.addEventListener("change", () => {{
        video.playbackRate = Number(speed.value);
      }});
      document.getElementById("fullscreen").addEventListener("click", async () => {{
        if (!document.fullscreenElement) {{
          await document.querySelector(".viewer").requestFullscreen();
        }} else {{
          await document.exitFullscreen();
        }}
      }});

      window.addEventListener("keydown", (event) => {{
        const tag = event.target?.tagName;
        if (tag === "INPUT" || tag === "SELECT" || tag === "BUTTON") return;
        if (event.key === " ") {{
          event.preventDefault();
          togglePlayback();
        }} else if (event.key === "+" || event.key === "=") {{
          setScale(state.scale * 1.2);
        }} else if (event.key === "-") {{
          setScale(state.scale / 1.2);
        }} else if (event.key === "0") {{
          resetView();
        }}
      }});

      renderTransform();
    }})();
  </script>
</body>
</html>
"""


def write_viewer(
    run_dir: Path,
    *,
    video_filename: str = "policy.mp4",
    output_filename: str = "review.html",
) -> Path:
    """Write ``review.html`` next to a recording and return its path."""
    run_dir = Path(run_dir)
    output_path = run_dir / output_filename
    output_path.write_text(
        render_viewer(run_dir, video_filename=video_filename)
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing policy.mp4 and optionally summary.json.",
    )
    parser.add_argument("--video", default="policy.mp4")
    parser.add_argument("--output", default="review.html")
    args = parser.parse_args()
    output = write_viewer(
        args.run_dir,
        video_filename=args.video,
        output_filename=args.output,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
