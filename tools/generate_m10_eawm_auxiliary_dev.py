"""Generate the bounded public-transition development data for EAWM auxiliary GPPO-History.

This is a no-update development collector.  It records labels derived from
adjacent received/public snapshots and never serializes hidden simulator
state into the model input or label payload.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.eawm_auxiliary import (  # noqa: E402
    AUXILIARY_METHOD,
    AUXILIARY_SCHEMA,
    POSITION_CLASSES,
    STATE_CLASSES,
    build_public_event_targets,
)
from gppo_world.m10_environment import (  # noqa: E402
    M10Config,
    M10Environment,
    default_scenario,
    scenario_to_dict,
    weak_communication_profile,
)
from tools.run_m10_baseline_comparison import traditional_action  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def public_payload(observation: dict[str, Any]) -> dict[str, Any]:
    """Return only auditable public fields for the trace identity hash."""
    return {
        "flat": np.asarray(observation["flat"], dtype=np.float32).tolist(),
        "mask": np.asarray(observation["mask"], dtype=bool).tolist(),
        "uavs": np.asarray(observation["uavs"], dtype=np.float64).tolist(),
        "tasks": np.asarray(observation["tasks"], dtype=np.float64).tolist(),
        "entity_ids": observation["entity_ids"],
        "time": float(observation["time"]),
    }


def finite_public_count(observation: dict[str, Any]) -> int:
    count = 0
    for key in ("uavs", "tasks"):
        array = np.asarray(observation[key], dtype=np.float64)
        if array.ndim == 2:
            count += int((~np.isfinite(array)).sum())
    return count


def class_counts(values: np.ndarray, mask: np.ndarray, labels: tuple[str, ...]) -> dict[str, int]:
    return {name: int(((values == index) & mask).sum()) for index, name in enumerate(labels)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=471001)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1024)
    args = parser.parse_args()
    if args.count != 8 or args.seed != 471001 or args.max_steps != 1024:
        raise SystemExit("this registered development run is fixed at seed=471001, count=8, max-steps=1024")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to reuse non-empty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    scenarios = []
    for index in range(args.count):
        scenario = default_scenario("mixed", seed=args.seed + index, split="eawm-aux-dev")
        scenarios.append(replace(scenario, communication=weak_communication_profile("composite")))

    source_files = [ROOT / "gppo_world" / "eawm_auxiliary.py", ROOT / "gppo_world" / "m10_environment.py", ROOT / "gppo_world" / "task_policy_view.py"]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files}
    dump(args.out / "scenarios.json", [scenario_to_dict(item) for item in scenarios])
    config_payload = asdict(config)
    run_id = f"eawm-aux-dev-seed{args.seed}"
    dump(args.out / "run-identity.json", {
        "run_id": run_id,
        "status": "running",
        "method": AUXILIARY_METHOD,
        "auxiliary_schema": AUXILIARY_SCHEMA,
        "source_commit": git_commit(),
        "source_hashes": source_hashes,
        "protocol": {"task_completion_mode": config.task_completion_mode, "deadline_basis": config.deadline_basis},
        "config": config_payload,
        "seed": args.seed,
        "parent_episode_budget": args.count,
        "environment_step_budget": args.max_steps,
        "parameter_updates": 0,
        "controller": "traditional_action using public observation only",
        "communication_profile": "composite weak communication, fixed for development",
        "formal_test_or_ood_used": False,
        "started_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    labels_path = args.out / "public-event-labels.jsonl"
    stats = {
        "schema": AUXILIARY_SCHEMA,
        "parent_episodes": 0,
        "environment_steps": 0,
        "records": 0,
        "effective_records": {"position": 0, "energy": 0, "task_state": 0},
        "class_counts": {
            "position_x": {name: 0 for name in POSITION_CLASSES},
            "position_y": {name: 0 for name in POSITION_CLASSES},
            "energy": {name: 0 for name in POSITION_CLASSES},
            "task_state": {name: 0 for name in STATE_CLASSES},
        },
        "mask_reasons": {"episode_boundary": 0, "identity_mismatch": 0, "missing_or_invalid": 0, "nonfinite": 0},
        "identity_matches": {"uav": 0, "task": 0},
        "nonfinite_public_values": 0,
        "parent_record_counts": {},
        "wall_seconds": None,
    }
    with labels_path.open("w", encoding="utf-8", newline="\n") as output:
        for parent_index, scenario in enumerate(scenarios):
            env = M10Environment(config, scenario)
            observation = env.reset()
            done = False
            local_steps = 0
            while not done and stats["environment_steps"] < args.max_steps:
                action = int(traditional_action(observation))
                next_observation, reward, done, info = env.step(action, submit_command=True)
                targets = build_public_event_targets(observation, next_observation)
                stats["records"] += 1
                stats["environment_steps"] += 1
                stats["effective_records"]["position"] += targets.valid_counts["position"]
                stats["effective_records"]["energy"] += targets.valid_counts["energy"]
                stats["effective_records"]["task_state"] += targets.valid_counts["task_state"]
                for key, value in targets.mask_reasons.items():
                    stats["mask_reasons"][key] += int(value)
                for key, value in targets.identity_matches.items():
                    stats["identity_matches"][key] += int(value)
                stats["nonfinite_public_values"] += finite_public_count(observation) + finite_public_count(next_observation)
                for axis, axis_name in enumerate(("x", "y")):
                    counts = class_counts(targets.position_class[:, axis], targets.position_mask[:, axis], POSITION_CLASSES)
                    for name, value in counts.items():
                        stats["class_counts"][f"position_{axis_name}"][name] += value
                counts = class_counts(targets.energy_class, targets.energy_mask, POSITION_CLASSES)
                for name, value in counts.items():
                    stats["class_counts"]["energy"][name] += value
                state_mask = targets.task_state_mask
                stats["class_counts"]["task_state"]["unchanged"] += int(((targets.task_state_changed == 0) & state_mask).sum())
                stats["class_counts"]["task_state"]["changed"] += int(((targets.task_state_changed == 1) & state_mask).sum())
                record = {
                    "run_id": run_id,
                    "parent_episode": parent_index,
                    "tape_id": scenario.tape_id,
                    "step": local_steps,
                    "action": action,
                    "reward": float(reward),
                    "done": bool(done),
                    "terminated": bool(info.get("terminated", done)),
                    "truncated": bool(info.get("truncated", False)),
                    "public_observation_sha256": canonical_sha(public_payload(observation)),
                    "next_public_observation_sha256": canonical_sha(public_payload(next_observation)),
                    "entity_ids": observation["entity_ids"],
                    "targets": targets.as_dict(),
                }
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                local_steps += 1
                observation = next_observation
                if stats["environment_steps"] >= args.max_steps:
                    break
            stats["parent_episodes"] += 1
            stats["parent_record_counts"][scenario.tape_id] = local_steps

    stats["wall_seconds"] = time.perf_counter() - started
    dump(args.out / "stats.json", stats)
    run_identity = {
        "run_id": run_id,
        "status": "complete" if stats["parent_episodes"] == args.count and stats["environment_steps"] < args.max_steps else "budget_exhausted_or_incomplete",
        "method": AUXILIARY_METHOD,
        "auxiliary_schema": AUXILIARY_SCHEMA,
        "source_commit": git_commit(),
        "source_hashes": source_hashes,
        "protocol": {"task_completion_mode": config.task_completion_mode, "deadline_basis": config.deadline_basis},
        "config": config_payload,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "collection_device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "seed": args.seed,
        "parent_episode_budget": args.count,
        "environment_step_budget": args.max_steps,
        "parent_episodes": stats["parent_episodes"],
        "environment_steps": stats["environment_steps"],
        "parameter_updates": 0,
        "controller": "traditional_action using public observation only",
        "communication_profile": "composite weak communication, fixed for development",
        "formal_test_or_ood_used": False,
        "wall_seconds": stats["wall_seconds"],
    }
    dump(args.out / "run-identity.json", run_identity)
    manifest = {
        "schema": AUXILIARY_SCHEMA,
        "run_id": run_id,
        "files": {
            "scenarios.json": {"path": "scenarios.json", "sha256": sha256_file(args.out / "scenarios.json")},
            "public-event-labels.jsonl": {"path": "public-event-labels.jsonl", "sha256": sha256_file(labels_path), "records": stats["records"]},
            "stats.json": {"path": "stats.json", "sha256": sha256_file(args.out / "stats.json")},
        },
    }
    dump(args.out / "manifest.json", manifest)
    dump(args.out / "run-status.json", {
        "run_id": run_id,
        "status": "complete" if stats["parent_episodes"] == args.count and stats["environment_steps"] < args.max_steps else "budget_exhausted_or_incomplete",
        "parameter_updates": 0,
        "environment_steps": stats["environment_steps"],
        "wall_seconds": stats["wall_seconds"],
        "manifest_sha256": sha256_file(args.out / "manifest.json"),
    })
    print(json.dumps({"run_id": run_id, "status": "complete", "records": stats["records"], "environment_steps": stats["environment_steps"], "wall_seconds": stats["wall_seconds"], "manifest": str(args.out / "manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
