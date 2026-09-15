"""Content-level split and historical-overlap audit for the frozen dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scene_key(row: dict[str, object]) -> str:
    scene = row["instance"] if "instance" in row else row
    scene = {key: value for key, value in scene.items() if key != "instance_id"}
    return digest(scene)


def load_rows(path: Path) -> list[dict[str, object]]:
    return [*json.loads(path.read_text(encoding="utf-8"))["splits"]["train"]["records"], *json.loads(path.read_text(encoding="utf-8"))["splits"]["validation"]["records"], *json.loads(path.read_text(encoding="utf-8"))["splits"]["test"]["records"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--old-instances", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    split_keys = {split: [scene_key(row) for row in data["splits"][split]["records"]] for split in ("train", "validation", "test")}
    old = json.loads(args.old_instances.read_text(encoding="utf-8"))["instances"]
    old_keys = [scene_key(row) for row in old]
    all_keys = [key for values in split_keys.values() for key in values]
    duplicate_keys = sorted(key for key in set(all_keys) if all_keys.count(key) > 1)
    overlap = sorted(set(all_keys) & set(old_keys))
    result = {"schema": "m10-minimal-consequence-content-audit-v1", "dataset": str(args.dataset.resolve()), "old_instances": str(args.old_instances.resolve()), "split_counts": {k: len(v) for k, v in split_keys.items()}, "within_and_cross_split_duplicate_count": len(duplicate_keys), "historical_32_content_overlap_count": len(overlap), "dataset_content_hashes": {k: digest(v) for k, v in split_keys.items()}, "status": "passed" if not duplicate_keys and not overlap and all(len(set(v)) == len(v) for v in split_keys.values()) else "failed", "method": "canonical full scene content including UAV positions and all subtask fields; labels/candidates excluded from scene identity"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
