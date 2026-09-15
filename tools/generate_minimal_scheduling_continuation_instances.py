"""Freeze 128 fresh continuation-validation instances before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.minimal_scheduling import Instance, Subtask, UAV, instance_to_dict


def canonical_scene(scene: dict[str, object]) -> str:
    uavs = sorted(
        [{"uav_id": int(x["uav_id"]), "position": list(x["position"])} for x in scene["uavs"]],
        key=lambda x: x["uav_id"],
    )
    tasks = []
    for task in scene["subtasks"]:
        tasks.append({
            "task_id": task["task_id"], "chain_id": task["chain_id"], "index": int(task["index"]),
            "position": list(task["position"]), "process_time": int(task["process_time"]),
            "eligible_uavs": sorted(int(u) for u in task["eligible_uavs"]),
            "predecessors": sorted(task["predecessors"]),
        })
    tasks.sort(key=lambda x: (x["chain_id"], x["index"], x["task_id"]))
    return json.dumps({"uavs": uavs, "subtasks": tasks}, sort_keys=True, separators=(",", ":"))


def scene_digest(scene: dict[str, object]) -> str:
    return hashlib.sha256(canonical_scene(scene).encode("utf-8")).hexdigest()


def read_scenes(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenes: list[dict[str, object]] = []
    if "splits" in payload:
        for split in payload["splits"].values():
            for row in split.get("records", []):
                scenes.append(row.get("instance", row))
    elif "instances" in payload:
        scenes.extend(payload["instances"])
    else:
        raise ValueError(f"unrecognized historical instance file: {path}")
    return scenes


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_instance(instance_id: str, group: str, rng: random.Random) -> Instance:
    uavs = tuple(UAV(i, (rng.randint(0, 10), rng.randint(0, 10))) for i in range(2))
    tasks = []
    for chain in range(2):
        for step in range(2):
            eligible = (0, 1) if group == "homogeneous" else rng.choice(((0,), (1,), (0, 1)))
            task_id = f"chain-{chain}-subtask-{step}"
            tasks.append(Subtask(task_id, f"chain-{chain}", step,
                (rng.randint(0, 10), rng.randint(0, 10)), rng.randint(1, 5), eligible,
                (f"chain-{chain}-subtask-{step - 1}",) if step else ()))
    return Instance(instance_id, group, uavs, tuple(tasks))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--known-data", type=Path, required=True)
    ap.add_argument("--known-instances", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=731001)
    args = ap.parse_args()
    if args.seed != 731001:
        raise SystemExit("registered generation seed is fixed at 731001")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {args.out}")
    start = time.monotonic()
    old_scenes = read_scenes(args.known_data) + read_scenes(args.known_instances)
    known = {scene_digest(scene) for scene in old_scenes}
    accepted: list[dict[str, object]] = []
    duplicate_attempts: list[dict[str, object]] = []
    rng = random.Random(args.seed)
    generation_index = 0
    for group, count in (("homogeneous", 64), ("heterogeneous", 64)):
        accepted_group = 0
        while accepted_group < count:
            scene = instance_to_dict(make_instance(
                f"m10-minsched-continuation-{group}-{accepted_group:03d}", group, rng
            ))
            digest = scene_digest(scene)
            current_index = generation_index
            generation_index += 1
            if digest in known:
                duplicate_attempts.append({"generation_index": current_index, "group": group, "content_sha256": digest, "action": "replace_next_rng_draw"})
                continue
            known.add(digest)
            accepted.append({"generation_index": current_index, "group_index": accepted_group, "content_sha256": digest, "instance": scene})
            accepted_group += 1
    output = {
        "schema": "m10-minimal-scheduling-continuation-instances-v1",
        "seed": args.seed,
        "count": len(accepted),
        "records": accepted,
    }
    dump(args.out / "instances.json", output)
    identity = {
        "schema": "m10-minimal-scheduling-continuation-generation-identity-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "seed": args.seed,
        "generation_order": ["homogeneous:64", "heterogeneous:64"],
        "candidate_distribution": "same field generation order and support as minimal consequence train/test generator",
        "historical_sources": [
            {"path": str(args.known_data.resolve()), "sha256": sha256(args.known_data), "scenes": len(read_scenes(args.known_data))},
            {"path": str(args.known_instances.resolve()), "sha256": sha256(args.known_instances), "scenes": len(read_scenes(args.known_instances))},
        ],
        "duplicate_replacements": duplicate_attempts,
        "duplicate_replacement_count": len(duplicate_attempts),
        "accepted_count_by_group": {g: sum(row["group"] == g for row in accepted) for g in ("homogeneous", "heterogeneous")},
        "accepted_content_sha256": {row["instance"]["instance_id"]: row["content_sha256"] for row in accepted},
        "elapsed_seconds": time.monotonic() - start,
        "test_outcomes_unseen": True,
    }
    dump(args.out / "run-identity.json", identity)
    files = []
    for path in sorted(args.out.iterdir()):
        if path.is_file(): files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    dump(args.out / "manifest.json", {"schema": "m10-minimal-scheduling-continuation-instance-manifest-v1", "files": files})
    print(json.dumps({"status": "frozen", "count": len(accepted), "groups": identity["accepted_count_by_group"], "duplicate_replacements": len(duplicate_attempts), "elapsed_seconds": identity["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
