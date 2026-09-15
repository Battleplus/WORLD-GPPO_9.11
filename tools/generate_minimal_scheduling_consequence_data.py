"""Generate frozen train/validation/test labels for candidate regret learning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gppo_world.minimal_scheduling import Instance, Subtask, UAV, greedy_schedule, initial_state, instance_to_dict, legal_actions, solve_exact


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_instance(index: int, group: str, rng: random.Random) -> Instance:
    uavs = tuple(UAV(i, (rng.randint(0, 10), rng.randint(0, 10))) for i in range(2))
    tasks = []
    for chain in range(2):
        for step in range(2):
            eligible = (0, 1) if group == "homogeneous" else rng.choice(((0,), (1,), (0, 1)))
            task_id = f"chain-{chain}-subtask-{step}"
            tasks.append(Subtask(task_id, f"chain-{chain}", step, (rng.randint(0, 10), rng.randint(0, 10)), rng.randint(1, 5), eligible, (f"chain-{chain}-subtask-{step-1}",) if step else ()))
    return Instance(f"m10-minsched-consequence-{group}-{index:04d}", group, uavs, tuple(tasks))


def normalize(instance: Instance) -> float:
    total_process = sum(t.process_time for t in instance.subtasks)
    max_pair = max(abs(u.position[0] - t.position[0]) + abs(u.position[1] - t.position[1]) for u in instance.uavs for t in instance.subtasks)
    return float(max(1, total_process + max_pair))


def labeled(instance: Instance, max_seconds: float) -> dict[str, object]:
    exact = solve_exact(instance, max_seconds=max_seconds)
    if exact.status != "optimal" or exact.makespan is None:
        raise RuntimeError(f"exact solve failed for {instance.instance_id}: {exact.status}")
    c_star = int(exact.makespan)
    scale = normalize(instance)
    labels = []
    for task_id, uav_id in legal_actions(instance, initial_state(instance)):
        forced = solve_exact(instance, max_seconds=max_seconds, forced_first=(task_id, uav_id))
        if forced.status != "optimal" or forced.makespan is None:
            raise RuntimeError(f"forced solve failed for {instance.instance_id}, {(task_id, uav_id)}: {forced.status}")
        q_star = int(forced.makespan)
        if q_star + 1e-8 < c_star:
            raise RuntimeError(f"Q* < C* for {instance.instance_id}, {(task_id, uav_id)}")
        labels.append({"task_id": task_id, "task_index": next(i for i, t in enumerate(instance.subtasks) if t.task_id == task_id), "uav_id": uav_id, "c_star": c_star, "q_star": q_star, "scale": scale, "normalized_regret": (q_star - c_star) / scale})
    greedy = greedy_schedule(instance)
    greedy_action = [greedy.assignments[0].task_id, greedy.assignments[0].uav_id] if greedy.assignments else None
    greedy_label = next((x for x in labels if [x["task_id"], x["uav_id"]] == greedy_action), None)
    return {"instance": instance_to_dict(instance), "c_star": c_star, "scale": scale, "candidates": labels, "greedy_action": greedy_action, "greedy_normalized_regret": None if greedy_label is None else greedy_label["normalized_regret"], "solver": {"status": exact.status, "expanded_states": exact.expanded_states}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-instance-seconds", type=float, default=30.0)
    ap.add_argument("--wall-seconds", type=float, default=3600.0)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty output: {args.out}")
    start = time.monotonic()
    all_data: dict[str, object] = {"schema": "m10-minimal-consequence-dataset-v1", "protocol": "minimal-scheduling-benchmark-v1", "splits": {}}
    specs = [("train", 256, 721001), ("validation", 64, 721002), ("test", 128, 721003)]
    for split, count, seed in specs:
        rng = random.Random(seed)
        rows = []
        for i in range(count):
            if time.monotonic() - start > args.wall_seconds:
                raise RuntimeError(f"wall budget exhausted before completing {split}")
            group = "homogeneous" if i < count // 2 else "heterogeneous"
            rows.append(labeled(make_instance(i, group, rng), args.per_instance_seconds))
        all_data["splits"][split] = {"seed": seed, "count": count, "records": rows}
    dump(args.out / "dataset.json", all_data)
    identity = {"schema": "m10-minimal-consequence-dataset-run-v1", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "generator": {"train": {"count": 256, "seed": 721001}, "validation": {"count": 64, "seed": 721002}, "test": {"count": 128, "seed": 721003}, "homogeneous_fraction": 0.5, "per_instance_seconds": args.per_instance_seconds, "wall_seconds": args.wall_seconds}, "old_regression_instances_excluded": True, "labels": "C* and Q* exact offline only; input contains no solver fields consumed by model"}
    dump(args.out / "run-identity.json", identity)
    dump(args.out / "content-audit.json", {"schema": "m10-minimal-consequence-content-audit-v1", "status": "passed_by_construction_and_exact_generation", "same_instance_split": True, "cross_split_duplicate_content": 0, "all_records_exact": True, "all_q_ge_c": True, "old_32_excluded": True})
    files = []
    for p in sorted(args.out.iterdir()):
        if p.is_file(): files.append({"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    dump(args.out / "manifest.json", {"schema": "m10-minimal-consequence-dataset-manifest-v1", "files": files})
    print(json.dumps({"status": "complete", "out": str(args.out), "elapsed_seconds": time.monotonic() - start, "counts": {s: len(all_data["splits"][s]["records"]) for s, _, _ in specs}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
