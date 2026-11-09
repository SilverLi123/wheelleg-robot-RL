"""Hash source and assets, and optionally create a portable cloud upload bundle."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def files():
    prefixes = ("source", "scripts", "docs/own_robot", "tests", "audit/2026-10-28_own_robot",
                "audit/2026-10-28_model_migration", "audit/2026-10-28_cloud_training")
    result = [ROOT / n for n in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md")]
    for prefix in prefixes:
        result += [p for p in (ROOT / prefix).rglob("*") if p.is_file()
                   and not any(x == "__pycache__" or x.endswith(".egg-info") for x in p.parts)
                   and p.suffix not in (".pyc", ".pyo", ".tar", ".gz")
                   and p.name != "delivery_manifest.json"]
    return sorted(set(result))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    args = ap.parse_args()
    selected = files()
    data = {"project": "wheeled-legged_RL-main", "identity": "file hashes; repository has no .git",
            "files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in selected}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    if args.bundle:
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.bundle, "w:gz") as tar:
            for p in selected:
                tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
            tar.add(args.output, arcname="delivery_manifest.json", recursive=False)
        print(args.bundle, hashlib.sha256(args.bundle.read_bytes()).hexdigest())
    print(f"Recorded {len(selected)} files in {args.output}")


if __name__ == "__main__":
    main()
