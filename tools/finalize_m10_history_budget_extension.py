"""Freeze and audit the completed H budget-extension artifacts.

This is a read-only analysis of an existing run.  It never trains or evaluates
an additional episode.  In particular, it detects stage files that were
rewritten by a post-completion resume instead of treating their filenames as
evidence for an intermediate checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


SEEDS = (1101, 2203, 3307)
STAGES = (8192, 16384, 24576, 32768)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_state(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    recovery = payload.get("recovery_state", {})
    metadata = payload.get("metadata", {})
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "format": payload.get("format"),
        "environment_steps": recovery.get("environment_steps", metadata.get("environment_steps")),
        "optimizer_updates": recovery.get("optimizer_updates", metadata.get("optimizer_updates")),
        "optimizer_present": bool(payload.get("optimizer_state_dict")),
        "rng_keys": sorted((recovery.get("rng_state") or {}).keys()),
    }


def bootstrap_tape_delta(results: dict) -> dict:
    by = {(int(e["checkpoint_steps"]), int(e["seed"])): e for e in results["evaluations"]}
    per_seed = {
        str(seed): {
            "physical_arrival_delta_tasks": by[(32768, seed)]["physical_arrival_on_time"]
            - by[(8192, seed)]["physical_arrival_on_time"],
            "physical_arrival_delta_rate": (
                by[(32768, seed)]["physical_arrival_on_time"]
                - by[(8192, seed)]["physical_arrival_on_time"]
            ) / by[(32768, seed)]["tasks"],
        }
        for seed in SEEDS
    }
    maps = {}
    for e in results["evaluations"]:
        if int(e["checkpoint_steps"]) in (8192, 32768):
            maps[(int(e["checkpoint_steps"]), int(e["seed"]))] = {
                x["tape_id"]: x["physical_arrival_on_time"] / x["tasks"]
                for x in e["episodes"]
            }
    tape_ids = sorted(maps[(8192, SEEDS[0])])
    tape_delta = [
        sum(maps[(32768, seed)][t] - maps[(8192, seed)][t] for seed in SEEDS) / len(SEEDS)
        for t in tape_ids
    ]
    rng = random.Random(20260914)
    boot = [sum(tape_delta[rng.randrange(len(tape_delta))] for _ in tape_delta) / len(tape_delta) for _ in range(20000)]
    boot.sort()
    return {
        "unit": "parent-tape mean task completion-rate difference; 64 paired tapes",
        "seed_deltas": per_seed,
        "mean_rate_delta": sum(tape_delta) / len(tape_delta),
        "bootstrap_replicates": len(boot),
        "bootstrap_seed": 20260914,
        "bootstrap_ci_95_tape_only": [boot[500], boot[19500]],
        "bootstrap_does_not_cover": "training-seed uncertainty; seed values are averaged within each tape",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    results = load(root / "results.json")
    evaluations = {(int(e["checkpoint_steps"]), int(e["seed"])): e for e in results["evaluations"]}

    runs = {}
    for seed in SEEDS:
        run = root / "runs" / f"m10-eawm-aux-h-{seed}-budget-extension"
        stages = {}
        for stage in STAGES:
            path = run / f"checkpoint-{stage}.pt"
            state = checkpoint_state(path) if path.exists() else None
            stages[str(stage)] = {
                "present": bool(state),
                "state": state,
                "label_matches_state": bool(state and state["environment_steps"] == stage),
            }
        runs[str(seed)] = {
            "source_checkpoint": checkpoint_state(run / "checkpoint-8192.pt"),
            "stages": stages,
            "training_rollout_ledger_records": sum(1 for _ in (gzip_open(run / "training-rollout-ledger.jsonl.gz"))),
            "update_records": sum(1 for _ in run.joinpath("updates.jsonl").open(encoding="utf-8")),
            "final_checkpoint": checkpoint_state(run / "checkpoint-32768.pt"),
        }

    eval_audit = {}
    for key, e in sorted(evaluations.items()):
        p = root / "evaluations" / f"h-{key[1]}-{key[0]}" / "evaluation-ledger.jsonl.gz"
        eval_audit[f"h-{key[1]}-{key[0]}"] = {
            "tasks": e["tasks"],
            "episodes": len(e["episodes"]),
            "ledger_records": sum(1 for _ in gzip_open(p)),
            "security_violations": e["security_violations"],
            "unresolved_tasks": e["unresolved_tasks"],
        }

    top_files = ["results.json", "frozen-tapes.json", "tape-manifest.json", "run-identity.json", "recovery-preflight.json"]
    hashes = {name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size} for name in top_files if (root / name).exists()}
    summary = {
        "schema": "m10-history-budget-extension-audit-v1",
        "source_commit": "caf2b2bb4e8e8bcf7d1c75214e4f5a47bd6de1c9",
        "run_root": str(root),
        "device": "cuda / NVIDIA GeForce RTX 3060 Laptop GPU",
        "scope": "H only; extension from 8192 to 32768; no E, world-model, prior, event-trigger, reward, or protocol change",
        "budget": {"seeds": list(SEEDS), "starting_steps": 8192, "target_steps": 32768, "new_steps_per_seed": 24576, "new_steps_total": 73728, "starting_updates": 128, "target_updates": 512, "new_updates_per_seed": 384, "new_updates_total": 1152},
        "recovery_preflight": load(root / "recovery-preflight.json"),
        "new_test": load(root / "tape-manifest.json"),
        "runs": runs,
        "evaluations": eval_audit,
        "results": results,
        "paired_tape_bootstrap": bootstrap_tape_delta(results),
        "integrity_note": "A post-completion no-op resume rewrote the labelled 16384/24576 stage files with the final 32768 state. Those labels are marked invalid; the final 32768 checkpoint and final-test results remain separately verifiable. No rerun was performed to repair this artifact issue.",
        "top_level_hashes": hashes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "status": "written", "tape_ci": summary["paired_tape_bootstrap"]["bootstrap_ci_95_tape_only"]}, ensure_ascii=False))
    return 0


def gzip_open(path: Path):
    import gzip

    return gzip.open(path, "rt", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
