"""Export checkpoint and scalar evidence from an actual RSL-RL run."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    import yaml
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    checkpoints = sorted(args.run.glob("model_*.pt"), key=lambda x: int(x.stem.split("_")[-1]))
    if not checkpoints:
        raise FileNotFoundError("No actual training checkpoint")
    checkpoint = checkpoints[-1]
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_finite = all(not torch.is_tensor(x) or bool(torch.isfinite(x).all())
                       for x in data["model_state_dict"].values())
    optimizer_finite = all(not torch.is_tensor(x) or bool(torch.isfinite(x).all())
        for state in data["optimizer_state_dict"]["state"].values() for x in state.values())
    events = EventAccumulator(str(args.run), size_guidance={"scalars": 0}).Reload()
    scalars, series = {}, {}
    for tag in events.Tags()["scalars"]:
        values = events.Scalars(tag)
        y = [x.value for x in values]
        scalars[tag] = {"first": y[0], "last": y[-1], "min": min(y), "max": max(y),
                        "points": len(y), "finite": all(math.isfinite(v) for v in y)}
        if not tag.startswith("Episode/"):
            series[tag] = [[x.step, x.value] for x in values]
    env = yaml.load((args.run / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    enabled = {k: float(v) for k, v in env["rewards"].items() if float(v) != 0.0}
    no_tag = [k for k in enabled if "Episode/Reward/" + k not in scalars]
    zero_terms = [k for k in enabled if "Episode/Reward/" + k in scalars and
                  scalars["Episode/Reward/" + k]["min"] == scalars["Episode/Reward/" + k]["max"] == 0]
    report = {"run": str(args.run), "checkpoint": str(checkpoint), "checkpoint_iter": data["iter"],
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "model_finite": model_finite, "optimizer_finite": optimizer_finite,
              "optimizer_has_history": bool(data["optimizer_state_dict"]["state"]),
              "enabled_rewards": enabled, "enabled_rewards_without_scalar_tag": no_tag,
              "enabled_rewards_zero_in_this_run": zero_terms, "scalars": scalars, "series": series}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("series", "scalars", "enabled_rewards")}, indent=2))
    if not model_finite or not optimizer_finite or any(not x["finite"] for x in scalars.values()) or no_tag:
        raise RuntimeError("Nonfinite state/scalar or enabled reward missing from real run logs")


if __name__ == "__main__":
    main()
