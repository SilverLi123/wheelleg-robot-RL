#!/usr/bin/env python3
"""Assemble nine terrain recordings into grid and sequential overview videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


TILES = (
    ("flat", "Flat"),
    ("up06", "Uphill 6%"),
    ("up10", "Uphill 10%"),
    ("up14", "Uphill 14%"),
    ("up18", "Uphill 18%"),
    ("down06", "Downhill 6%"),
    ("down10", "Downhill 10%"),
    ("down14", "Downhill 14%"),
    ("down18", "Downhill 18%"),
)
GRID = (
    ("flat", "up06", "up10"),
    ("up14", "up18", "down06"),
    ("down10", "down14", "down18"),
)
OUTPUT_FPS = 25
TILE_SIZE = (640, 360)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int) -> ImageFont.FreeTypeFont:
    try:
        path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def label_tile(frame: np.ndarray, name: str) -> Image.Image:
    image = Image.fromarray(frame).resize(TILE_SIZE, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((10, 10, 225, 48), radius=7, fill=(9, 18, 29, 210))
    draw.text((20, 15), name, font=font(20), fill=(245, 249, 252, 255))
    return image


def write_grid(root: Path, destination: Path) -> None:
    readers = {
        name: imageio.get_reader(root / name / "policy.mp4")
        for name, _ in TILES
    }
    writer = imageio.get_writer(
        destination,
        fps=OUTPUT_FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
    )
    labels = dict(TILES)
    frame_index = 0
    try:
        while True:
            canvas = Image.new("RGB", (1920, 1080), "#0b1118")
            try:
                for row, names in enumerate(GRID):
                    for column, name in enumerate(names):
                        frame = readers[name].get_next_data()
                        # Source clips are 50 FPS. Keep one frame in two so the
                        # resulting comparison remains an 8-second, 25 FPS clip.
                        if frame_index % 2:
                            continue
                        tile = label_tile(frame, labels[name])
                        canvas.paste(
                            tile,
                            (column * TILE_SIZE[0], row * TILE_SIZE[1]),
                        )
                if frame_index % 2 == 0:
                    writer.append_data(np.asarray(canvas))
                frame_index += 1
            except (IndexError, RuntimeError):
                break
    finally:
        writer.close()
        for reader in readers.values():
            reader.close()


def write_sequence(root: Path, destination: Path) -> None:
    writer = imageio.get_writer(
        destination,
        fps=OUTPUT_FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        output_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
    )
    title = Image.new("RGB", (1280, 720), "#0b1118")
    draw = ImageDraw.Draw(title)
    draw.text(
        (60, 245),
        "Wheelbipe V14-6 baseline",
        font=font(54),
        fill=(244, 248, 252),
    )
    draw.text(
        (60, 330),
        "Mixed terrain replay: flat, 6-18% uphill and downhill",
        font=font(30),
        fill=(160, 201, 234),
    )
    draw.text(
        (60, 388),
        "Command: +0.5 m/s, zero yaw | domain randomization enabled",
        font=font(24),
        fill=(185, 198, 211),
    )
    for _ in range(OUTPUT_FPS):
        writer.append_data(np.asarray(title))

    try:
        for name, label in TILES:
            section = Image.new("RGB", (1280, 720), "#0b1118")
            draw = ImageDraw.Draw(section)
            draw.text((42, 38), label, font=font(44), fill=(244, 248, 252))
            draw.text(
                (44, 100),
                "Same checkpoint, seed, command and randomization settings",
                font=font(22),
                fill=(160, 201, 234),
            )
            for _ in range(OUTPUT_FPS // 2):
                writer.append_data(np.asarray(section))

            reader = imageio.get_reader(root / name / "policy.mp4")
            try:
                for index, frame in enumerate(reader):
                    if index % 2 == 0:
                        writer.append_data(frame)
            finally:
                reader.close()
    finally:
        writer.close()


def write_contact_sheet(root: Path, destination: Path) -> None:
    """Show one late rollout frame from every terrain in the same 3x3 grid."""
    canvas = Image.new("RGB", (1920, 1080), "#0b1118")
    labels = dict(TILES)
    for row, names in enumerate(GRID):
        for column, name in enumerate(names):
            reader = imageio.get_reader(root / name / "policy.mp4")
            try:
                frame_count = reader.count_frames()
                # Use a late frame so the robot has already travelled onto the
                # slope. The fallback handles streams whose frame count is not
                # exposed by their backend.
                try:
                    frame = reader.get_data(max(0, int(frame_count * 0.70)))
                except (IndexError, RuntimeError, ValueError):
                    frame = next(iter(reader))
            finally:
                reader.close()
            canvas.paste(
                label_tile(frame, labels[name]),
                (column * TILE_SIZE[0], row * TILE_SIZE[1]),
            )
    canvas.save(destination, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    missing = [
        str(root / name / "policy.mp4")
        for name, _ in TILES
        if not (root / name / "policy.mp4").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing terrain recordings: " + ", ".join(missing))

    grid = root / "all_terrains_grid_1920x1080.mp4"
    sequence = root / "all_terrains_sequence_1280x720.mp4"
    contact_sheet = root / "overview_contact_sheet.png"
    write_grid(root, grid)
    write_sequence(root, sequence)
    write_contact_sheet(root, contact_sheet)

    manifest = {
        "policy": "selected baseline model_50.pt",
        "command_vx_m_s": 0.5,
        "command_yaw_rad_s": 0.0,
        "seed": 20261111,
        "domain_randomization": True,
        "terrain_order": [name for name, _ in TILES],
        "videos": [
            {
                "name": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in (grid, sequence)
        ],
        "contact_sheet": {
            "name": contact_sheet.name,
            "path": str(contact_sheet),
            "bytes": contact_sheet.stat().st_size,
            "sha256": sha256(contact_sheet),
        },
    }
    (root / "overview_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
