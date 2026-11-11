#!/usr/bin/env python3
"""Summarize repeated mixed-slope evaluations without hiding seed variation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def summarize_group(paths: list[Path]) -> dict:
    reports = [json.loads(path.read_text()) for path in paths]
    seeds = [int(report["seed"]) for report in reports]
    tile_order = [row["tile"] for row in reports[0]["results"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds in {paths}")

    common = {
        "steps",
        "policy_dt_s",
        "checkpoint_sha256",
        "terrain_types",
    }
    for key in common:
        values = {json.dumps(report.get(key), sort_keys=True) for report in reports}
        if len(values) != 1:
            raise ValueError(f"Reports disagree on {key}: {sorted(paths)}")

    tile_rows: dict[str, list[dict]] = {}
    for report in reports:
        for row in report["results"]:
            tile_rows.setdefault(row["tile"], []).append(row)

    tiles = []
    for tile, rows in tile_rows.items():
        if len(rows) != len(reports):
            raise ValueError(f"Tile {tile} appears {len(rows)} times, expected {len(reports)}")
        success = sum(row["resets"] == 0 and not row["nonwheel_contact_peak_n"] for row in rows)
        tiles.append(
            {
                "tile": tile,
                "target_grade": rows[0]["target_grade"],
                "seeds": len(rows),
                "vx_mae": mean_std([float(row["vx_mae"]) for row in rows]),
                "abs_lateral_m": mean_std(
                    [abs(float(row["lateral_m"])) for row in rows]
                ),
                "forward_m": mean_std([float(row["forward_m"]) for row in rows]),
                "mean_forward_speed_m_s": mean_std(
                    [float(row["mean_forward_speed_m_s"]) for row in rows]
                ),
                "pitch_min_deg": mean_std(
                    [float(row["pitch_range_deg"][0]) for row in rows]
                ),
                "pitch_max_deg": mean_std(
                    [float(row["pitch_range_deg"][1]) for row in rows]
                ),
                "resets": sum(int(row["resets"]) for row in rows),
                "tiles_with_nonwheel_contact": sum(
                    bool(row["nonwheel_contact_peak_n"]) for row in rows
                ),
                "successful_seed_rows": success,
                "success_rate": success / len(rows),
            }
        )

    return {
        "checkpoint_sha256": reports[0]["checkpoint_sha256"],
        "checkpoint_iteration": reports[0]["checkpoint_iteration"],
        "seeds": sorted(seeds),
        "steps": reports[0]["steps"],
        "policy_dt_s": reports[0]["policy_dt_s"],
        "command": reports[0]["command"],
        "tiles": sorted(tiles, key=lambda row: tile_order.index(row["tile"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    groups = {}
    for model in ("baseline", "model59"):
        paths = sorted(args.raw_dir.glob(f"{model}_seed*.json"))
        if len(paths) < 2:
            raise ValueError(f"Need repeated-seed reports for {model}")
        groups[model] = summarize_group(paths)

    baseline = {row["tile"]: row for row in groups["baseline"]["tiles"]}
    candidate = {row["tile"]: row for row in groups["model59"]["tiles"]}
    comparisons = []
    for tile in baseline:
        comparisons.append(
            {
                "tile": tile,
                "vx_mae_delta_candidate_minus_baseline": (
                    candidate[tile]["vx_mae"]["mean"]
                    - baseline[tile]["vx_mae"]["mean"]
                ),
                "abs_lateral_delta_candidate_minus_baseline_m": (
                    candidate[tile]["abs_lateral_m"]["mean"]
                    - baseline[tile]["abs_lateral_m"]["mean"]
                ),
            }
        )

    all_baseline_mae = [
        row["vx_mae"]["mean"] for row in groups["baseline"]["tiles"]
    ]
    all_candidate_mae = [
        row["vx_mae"]["mean"] for row in groups["model59"]["tiles"]
    ]
    all_baseline_lateral = [
        row["abs_lateral_m"]["mean"] for row in groups["baseline"]["tiles"]
    ]
    all_candidate_lateral = [
        row["abs_lateral_m"]["mean"] for row in groups["model59"]["tiles"]
    ]
    summary = {
        "schema_version": 1,
        "evaluation_scope": {
            "command_vx_m_s": 0.5,
            "command_yaw_rad_s": 0.0,
            "steps_per_seed": groups["baseline"]["steps"],
            "duration_s": groups["baseline"]["steps"] * groups["baseline"]["policy_dt_s"],
            "seeds": groups["baseline"]["seeds"],
            "terrain": "flat, 6/10/14/18% upslope, 6/10/14/18% downslope",
        },
        "models": groups,
        "comparison": {
            "mean_vx_mae_over_all_tiles": {
                "baseline": mean_std(all_baseline_mae),
                "model59": mean_std(all_candidate_mae),
                "candidate_minus_baseline": (
                    statistics.fmean(all_candidate_mae)
                    - statistics.fmean(all_baseline_mae)
                ),
            },
            "mean_abs_lateral_over_all_tiles_m": {
                "baseline": mean_std(all_baseline_lateral),
                "model59": mean_std(all_candidate_lateral),
                "candidate_minus_baseline": (
                    statistics.fmean(all_candidate_lateral)
                    - statistics.fmean(all_baseline_lateral)
                ),
            },
            "per_tile": comparisons,
        },
        "selection": {
            "recommended": "baseline",
            "reason": (
                "The mixed-slope fine-tune completed without resets or body contact, "
                "but its mean velocity error increased on every terrain and its lateral "
                "result was not consistently better across seeds."
            ),
        },
    }
    if not all(math.isfinite(v) for row in summary["models"]["baseline"]["tiles"]
               for value in (row["vx_mae"]["mean"], row["abs_lateral_m"]["mean"])
               for v in [value]):
        raise ValueError("Nonfinite aggregate")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "tile",
                "target_grade",
                "vx_mae_mean",
                "vx_mae_std",
                "abs_lateral_m_mean",
                "abs_lateral_m_std",
                "forward_m_mean",
                "resets",
                "nonwheel_contact_rows",
                "success_rate",
            ]
        )
        for model, group in groups.items():
            for row in group["tiles"]:
                writer.writerow(
                    [
                        model,
                        row["tile"],
                        row["target_grade"],
                        f'{row["vx_mae"]["mean"]:.8f}',
                        f'{row["vx_mae"]["std"]:.8f}',
                        f'{row["abs_lateral_m"]["mean"]:.8f}',
                        f'{row["abs_lateral_m"]["std"]:.8f}',
                        f'{row["forward_m"]["mean"]:.8f}',
                        row["resets"],
                        row["tiles_with_nonwheel_contact"],
                        f'{row["success_rate"]:.3f}',
                    ]
                )

    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()
