"""Extend frozen H runs from 8192 to 32768 steps and evaluate a new test tape.

This entry is intentionally H-only. It resumes only at the existing runner's
rollout boundary, where the environment and History state are recreated for
each batch. No E, auxiliary loss, candidate prior, or new training route is
used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.m10_environment import M10Config, scenario_from_dict, scenario_to_dict, weak_communication_tape  # noqa: E402
from tools.run_m10_eawm_auxiliary_matrix import (  # noqa: E402
    M10Environment,
    arrival_config,
    evaluate_variant,
    make_policy,
    run_rules,
    train_one,
)


SEEDS = (1101, 2203, 3307)
START_STEPS = 8192
TARGETS = (16384, 24576, 32768)
UPDATES = {16384: 256, 24576: 384, 32768: 512}
NEW_TEST_BASE_SEED = 992001
NEW_TEST_COUNT = 64
TOTAL_WALL_SECONDS = 4 * 60 * 60


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_existing_tapes(source_root: Path) -> tuple[list, list]:
    payload = json.loads((source_root / "frozen-tapes.json").read_text(encoding="utf-8"))
    return ([scenario_from_dict(item) for item in payload["train"]],
            [scenario_from_dict(item) for item in payload["validation"]])


def new_test_tape(config: M10Config) -> list:
    return list(weak_communication_tape("test", count=NEW_TEST_COUNT, base_seed=NEW_TEST_BASE_SEED, level="composite"))


def verify_source_checkpoint(path: Path, seed: int, source_root: Path, out: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    recovery = payload.get("recovery_state") or {}
    metadata = payload.get("metadata") or {}
    state = metadata.get("state") or {}
    optimizer = payload.get("optimizer_state_dict")
    rng = recovery.get("rng_state") or {}
    checks = {
        "format": payload.get("format"),
        "format_expected": "gppo-history-arrival-training-v1",
        "format_match": payload.get("format") == "gppo-history-arrival-training-v1",
        "state_dict_present": isinstance(payload.get("state_dict"), dict) and bool(payload.get("state_dict")),
        "optimizer_present": isinstance(optimizer, dict) and bool(optimizer.get("param_groups")),
        "optimizer_state_entries": len((optimizer or {}).get("state", {})),
        "rng_keys": sorted(rng),
        "rng_complete": all(key in rng for key in ("python", "numpy", "torch", "cuda")),
        "environment_steps": recovery.get("environment_steps", state.get("environment_steps")),
        "optimizer_updates": recovery.get("optimizer_updates", state.get("optimizer_updates")),
        "seed": recovery.get("seed", seed),
        "config_present": bool(metadata.get("config")),
        "ppo_present": bool(metadata.get("ppo")),
        "data_order": "frozen train tape order; no shuffle; next rollout starts at cumulative step boundary",
        "rollout_boundary": "runner resets M10Environment and History at each rollout batch; no in-progress env state is required",
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": sha256(path),
        "source_root": str(source_root),
    }
    checks["ready_for_safe_boundary_resume"] = bool(
        checks["format_match"] and checks["state_dict_present"] and checks["optimizer_present"]
        and checks["rng_complete"] and checks["environment_steps"] == START_STEPS
        and checks["optimizer_updates"] == 128 and checks["config_present"] and checks["ppo_present"]
    )
    dump(out / f"recovery-preflight-{seed}.json", checks)
    if not checks["ready_for_safe_boundary_resume"]:
        raise RuntimeError(f"recovery preflight failed for seed {seed}: {checks}")
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.threads < 1 or args.threads > 4:
        raise SystemExit("threads must be 1..4")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent fallback")
    source_root = args.source_root.resolve()
    root = args.out.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise SystemExit(f"refusing non-empty extension root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    config = arrival_config()
    train_tape, validation_tape = load_existing_tapes(source_root)
    test_tape = new_test_tape(config)

    old_ids = {scenario.tape_id for key in ("train", "validation", "final_test")
               for scenario in [scenario_from_dict(x) for x in json.loads((source_root / "frozen-tapes.json").read_text(encoding="utf-8"))[key]]}
    new_ids = [scenario.tape_id for scenario in test_tape]
    if len(new_ids) != len(set(new_ids)) or old_ids.intersection(new_ids):
        raise RuntimeError("new final-test tape identity overlaps an existing tape")
    dump(root / "frozen-tapes.json", {
        "train": [scenario_to_dict(x) for x in train_tape],
        "validation": [scenario_to_dict(x) for x in validation_tape],
        "final_test": [scenario_to_dict(x) for x in test_tape],
    })
    dump(root / "tape-manifest.json", {
        "schema": "gppo-history-budget-extension-tape-manifest-v1",
        "source_tape_manifest": str(source_root / "tape-manifest.json"),
        "source_tape_manifest_sha256": sha256(source_root / "tape-manifest.json"),
        "train_source": "reused frozen train tape from prior H matrix; no scenario changes",
        "validation_source": "reused frozen validation tape from prior H matrix; curve only",
        "final_test_generation": "same weak composite generator, new base seed, no result filtering",
        "final_test_base_seed": NEW_TEST_BASE_SEED,
        "final_test_count": NEW_TEST_COUNT,
        "task_count_per_scenario": config.task_capacity,
        "task_completion_mode": config.task_completion_mode,
        "deadline_basis": config.deadline_basis,
        "communication": "composite weak communication",
        "final_test_not_read_during_training": True,
        "frozen_tapes_sha256": sha256(root / "frozen-tapes.json"),
        "old_tape_id_overlap": 0,
    })
    dump(root / "run-identity.json", {
        "schema": "gppo-history-budget-extension-v1",
        "source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_root": str(source_root),
        "source_matrix_commit": "93b6cf5ff98687cd43a2675d3fceebb08048205",
        "device": str(device),
        "threads": torch.get_num_threads() if device.type == "cpu" else None,
        "config": asdict(config),
        "budgets": {"starting_steps": START_STEPS, "targets": list(TARGETS), "updates": UPDATES, "new_steps_total": 73728, "wall_seconds": TOTAL_WALL_SECONDS},
        "recovery_boundary": "only after a completed rollout batch; runner resets env and History at each batch and derives next rollout seed from cumulative environment_steps",
        "variants": ["H only"],
    })

    preflight = []
    for seed in SEEDS:
        old_dir = source_root / "runs" / f"m10-eawm-aux-h-{seed}-formal"
        preflight.append(verify_source_checkpoint(old_dir / "last.pt", seed, source_root, root))
    dump(root / "recovery-preflight.json", {"status": "passed", "checks": preflight})

    deadline = time.perf_counter() + TOTAL_WALL_SECONDS
    for seed in SEEDS:
        old_dir = source_root / "runs" / f"m10-eawm-aux-h-{seed}-formal"
        source_last = old_dir / "last.pt"
        if not source_last.exists():
            raise RuntimeError(f"missing source recovery checkpoint: {source_last}")
        run_dir = root / "runs" / f"m10-eawm-aux-h-{seed}-budget-extension"
        if run_dir.exists() and any(run_dir.iterdir()):
            if not args.resume:
                raise RuntimeError(f"refusing non-empty extension run: {run_dir}")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_last, run_dir / "last.pt")
            shutil.copy2(source_last, run_dir / "checkpoint-8192.pt")
            dump(run_dir / "source-8192-checkpoint.json", {"path": str(source_last), "sha256": sha256(source_last), "source_run_id": f"m10-eawm-aux-h-{seed}-formal", "environment_steps": 8192, "optimizer_updates": 128})
        for target in TARGETS:
            if time.perf_counter() >= deadline:
                raise RuntimeError("global wall-clock budget exhausted before all seeds completed")
            result = train_one("H", seed, target, max_updates=UPDATES[target], root=root,
                               config=config, train_tape=train_tape, validation_tape=validation_tape,
                               device=device, wall_deadline=deadline, run_label="budget-extension", pilot=False,
                               resume=True)
            stage = run_dir / "stages" / str(target)
            stage.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / "last.pt", stage / "last.pt")
            shutil.copy2(run_dir / "last.pt", run_dir / f"checkpoint-{target}.pt")
            for name in ("run-status.json", "run-identity.json", "checkpoint-roundtrip.json", "validation-curve.json"):
                path = run_dir / name
                if path.exists():
                    shutil.copy2(path, stage / name)
            dump(stage / "stage-result.json", {"target_environment_steps": target, "target_optimizer_updates": UPDATES[target], "runner_result": result, "checkpoint_sha256": sha256(run_dir / f"checkpoint-{target}.pt")})
        final_status = json.loads((run_dir / "run-status.json").read_text(encoding="utf-8"))
        if final_status.get("environment_steps") != 32768 or final_status.get("optimizer_updates") != 512:
            raise RuntimeError(f"incomplete final extension for seed {seed}: {final_status}")

    # Evaluate old 8192 and new 32768 checkpoints on the same new tape only
    # after every extension run is complete.
    evaluations = []
    for seed in SEEDS:
        old_dir = source_root / "runs" / f"m10-eawm-aux-h-{seed}-formal"
        new_dir = root / "runs" / f"m10-eawm-aux-h-{seed}-budget-extension"
        for label, checkpoint in (("8192", old_dir / "last.pt"), ("32768", new_dir / "checkpoint-32768.pt")):
            policy = make_policy(config, "H", seed, device)
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if payload.get("format") != "gppo-history-arrival-training-v1":
                raise RuntimeError(f"checkpoint format mismatch: {checkpoint}")
            policy.load_state_dict(payload["state_dict"])
            out = root / "evaluations" / f"h-{seed}-{label}"
            existing = out / "summary.json"
            if existing.exists():
                result = json.loads(existing.read_text(encoding="utf-8"))
            else:
                if out.exists() and any(out.iterdir()):
                    raise RuntimeError(f"evaluation directory exists without summary: {out}")
                out.mkdir(parents=True, exist_ok=False)
                result = evaluate_variant(policy, "H", test_tape, config, device, f"history-budget-h-{seed}-{label}", out)
                result["seed"] = seed
                result["checkpoint_steps"] = int(label)
                result["checkpoint_sha256"] = sha256(checkpoint)
                dump(out / "summary.json", result)
            evaluations.append(result)
    rule_out = root / "evaluations" / "rule"
    rule = run_rules(test_tape, config, device, rule_out)
    dump(root / "results.json", {"schema": "gppo-history-budget-extension-results-v1", "evaluations": evaluations, "rule": rule, "new_test_tape_sha256": sha256(root / "frozen-tapes.json"), "source_checkpoint_steps": [8192, 32768]})
    print(json.dumps({"status": "complete", "root": str(root), "new_test_count": len(test_tape), "evaluations": len(evaluations)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
