"""Generate and solve the fixed 32-instance minimum scheduling benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from gppo_world.minimal_scheduling import Instance, Subtask, UAV, greedy_schedule, instance_to_dict, legal_actions, solve_exact


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def generate_instance(index: int, group: str, rng: random.Random) -> Instance:
    uavs = tuple(UAV(uav_id=i, position=(rng.randint(0, 10), rng.randint(0, 10))) for i in range(2))
    subtasks: list[Subtask] = []
    for chain in range(2):
        for step in range(2):
            if group == "homogeneous":
                eligible = (0, 1)
            else:
                eligible = rng.choice(((0,), (1,), (0, 1)))
            task_id = f"chain-{chain}-subtask-{step}"
            subtasks.append(Subtask(task_id, f"chain-{chain}", step, (rng.randint(0, 10), rng.randint(0, 10)), rng.randint(1, 5), eligible, (f"chain-{chain}-subtask-{step - 1}",) if step else ()))
    return Instance(f"m10-minsched-{group}-{index:02d}", group, uavs, tuple(subtasks))


def assignment_dict(item: object) -> dict[str, object]:
    return asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item)  # type: ignore[arg-type]


def solve_record(instance: Instance, limit: float, deadline: float) -> dict[str, object]:
    remaining = max(0.001, min(limit, deadline - time.monotonic()))
    greedy = greedy_schedule(instance)
    exact = solve_exact(instance, max_seconds=remaining)
    first = []
    for action in legal_actions(instance, __import__("gppo_world.minimal_scheduling", fromlist=["initial_state"]).initial_state(instance)):
        if time.monotonic() >= deadline:
            break
        forced = solve_exact(instance, max_seconds=max(0.001, min(limit, deadline - time.monotonic())), forced_first=action)
        first.append({"action": list(action), "status": forced.status, "total_makespan": forced.makespan, "assignments": [assignment_dict(x) for x in forced.assignments], "proof": forced.proof})
    greedy_ms = greedy.makespan
    exact_ms = exact.makespan
    return {"instance": instance_to_dict(instance), "greedy": {"status": greedy.status, "makespan": greedy_ms, "assignments": [assignment_dict(x) for x in greedy.assignments], "waits": greedy.waits}, "exact": {"status": exact.status, "makespan": exact_ms, "assignments": [assignment_dict(x) for x in exact.assignments], "waits": exact.waits, "expanded_states": exact.expanded_states, "proof": exact.proof}, "makespan_gap": None if greedy_ms is None or exact_ms is None else greedy_ms - exact_ms, "relative_gap": None if greedy_ms is None or exact_ms in (None, 0) else (greedy_ms - exact_ms) / exact_ms, "legal_first_action_count": len(legal_actions(instance, __import__("gppo_world.minimal_scheduling", fromlist=["initial_state"]).initial_state(instance))), "forced_first_action_results": first}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=615001)
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--per-instance-seconds", type=float, default=30.0)
    ap.add_argument("--wall-seconds", type=float, default=30 * 60)
    args = ap.parse_args()
    if args.count != 32:
        raise SystemExit("this registered run is fixed at 32 instances")
    started = time.monotonic()
    deadline = started + args.wall_seconds
    rng = random.Random(args.seed)
    instances = [generate_instance(i, "homogeneous" if i < 16 else "heterogeneous", rng) for i in range(args.count)]
    dump(args.out / "instances.json", {"schema": "m10-minimal-scheduling-instances-v1", "seed": args.seed, "instances": [instance_to_dict(x) for x in instances]})
    records = []
    for instance in instances:
        if time.monotonic() >= deadline:
            records.append({"instance": instance_to_dict(instance), "status": "budget_not_started"})
            continue
        records.append(solve_record(instance, args.per_instance_seconds, deadline))
    dump(args.out / "results.json", {"schema": "m10-minimal-scheduling-results-v1", "records": records})
    completed = [r for r in records if r.get("exact", {}).get("status") in ("optimal", "timeout_feasible_not_proven")]
    optimal = [r for r in records if r.get("exact", {}).get("status") == "optimal"]
    gaps = [r["makespan_gap"] for r in optimal if r.get("makespan_gap") is not None]
    differing = [r for r in optimal if (r.get("makespan_gap") or 0) > 0]
    summary = {"schema": "m10-minimal-scheduling-summary-v1", "seed": args.seed, "scheduled_instances": args.count, "completed_instances": len(completed), "optimal_proven": len(optimal), "timeout_or_unproven": sum(r.get("exact", {}).get("status") == "timeout_feasible_not_proven" for r in records), "test_failures": sum("exact" not in r for r in records), "different_greedy_vs_optimal": len(differing), "greedy_optimal": sum(r.get("makespan_gap") == 0 for r in optimal), "greedy_suboptimal": len(differing), "mean_absolute_gap": sum(gaps) / len(gaps) if gaps else None, "by_group": {group: {"instances": sum(r.get("instance", {}).get("group") == group for r in optimal), "different": sum(r.get("instance", {}).get("group") == group and (r.get("makespan_gap") or 0) > 0 for r in optimal), "greedy_optimal": sum(r.get("instance", {}).get("group") == group and r.get("makespan_gap") == 0 for r in optimal)} for group in ("homogeneous", "heterogeneous")}, "elapsed_seconds": time.monotonic() - started, "wall_budget_seconds": args.wall_seconds}
    dump(args.out / "summary.json", summary)
    identity = {"schema": "m10-minimal-scheduling-run-identity-v1", "source_commit": git("rev-parse", "HEAD"), "source_branch": git("branch", "--show-current"), "seed": args.seed, "generator": {"count": 32, "homogeneous": 16, "heterogeneous": 16, "grid": [0, 10], "flight": "Manhattan", "process_time": [1, 5], "eligibility_choices": [[0], [1], [0, 1]], "choice_probability": "uniform for heterogeneous"}, "solver": {"task_bound": 4, "uav_bound": 2, "per_instance_seconds": args.per_instance_seconds, "wall_seconds": args.wall_seconds, "wait": "advance to next completion event", "training": False}, "runtime": {"python": sys.version, "platform": platform.platform()}, "elapsed_seconds": time.monotonic() - started}
    dump(args.out / "run-identity.json", identity)
    manifest = []
    for path in sorted(args.out.iterdir()):
        if path.is_file():
            manifest.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    dump(args.out / "manifest.json", {"schema": "m10-minimal-scheduling-manifest-v1", "files": manifest})
    print(json.dumps({"status": "complete", "out": str(args.out), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

