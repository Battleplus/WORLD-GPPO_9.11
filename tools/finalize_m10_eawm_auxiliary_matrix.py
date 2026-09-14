"""Finalize the bounded EAWM auxiliary GPPO-History matrix without rerunning it."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (1101, 2203, 3307)
VARIANTS = ("H", "E")
N_BOOT = 20000


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def bootstrap_ci(values: list[float], seed: int = 9112026) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0, "method": "not_estimable"}
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(N_BOOT, arr.size), replace=True).mean(axis=1)
    return {
        "n": int(arr.size),
        "method": "percentile bootstrap over parent tapes; tape uncertainty only",
        "seed": seed,
        "replicates": N_BOOT,
        "estimate": float(arr.mean()),
        "ci95": [percentile(draws.tolist(), 2.5), percentile(draws.tolist(), 97.5)],
    }


def safe_rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-root", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    root = args.matrix_root.resolve()
    repo = args.repo_root.resolve()
    out_dir = repo / "artifacts" / "m10-eawm-auxiliary-history-matrix-20260914"
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = load(root / "matrix-results.json")
    identities = {}
    statuses = {}
    for p in sorted((root / "runs").glob("*/run-identity.json")):
        identities[p.parent.name] = load(p)
        statuses[p.parent.name] = load(p.parent / "run-status.json")

    final = {}
    for entry in matrix["final_test"]:
        run_id = entry["run_id"]
        identity = identities[run_id]
        final[(identity["variant"], int(identity["seed"]))] = entry

    rows = []
    for seed in SEEDS:
        h = final[("H", seed)]
        e = final[("E", seed)]
        h_return = sum(x["return"] for x in h["episodes"])
        e_return = sum(x["return"] for x in e["episodes"])
        rows.append({
            "seed": seed,
            "H": {k: h[k] for k in ("tasks", "physical_arrival_on_time", "host_confirmation_on_time", "deadline_failures", "unresolved_tasks", "correct_rejections", "security_violations")},
            "E": {k: e[k] for k in ("tasks", "physical_arrival_on_time", "host_confirmation_on_time", "deadline_failures", "unresolved_tasks", "correct_rejections", "security_violations")},
            "E_minus_H": {
                "physical_arrival_on_time": e["physical_arrival_on_time"] - h["physical_arrival_on_time"],
                "host_confirmation_on_time": e["host_confirmation_on_time"] - h["host_confirmation_on_time"],
                "deadline_failures": e["deadline_failures"] - h["deadline_failures"],
                "return": e_return - h_return,
                "latency_mean_ms": e["latency"]["mean_ms"] - h["latency"]["mean_ms"],
            },
            "return": {"H": h_return, "E": e_return},
            "latency_ms": {"H": h["latency"], "E": e["latency"]},
        })

    rule = load(root / "rules-final-test" / "summary.json")
    totals = {}
    for variant in VARIANTS:
        entries = [final[(variant, seed)] for seed in SEEDS]
        totals[variant] = {
            "tasks": sum(x["tasks"] for x in entries),
            "physical_arrival_on_time": sum(x["physical_arrival_on_time"] for x in entries),
            "host_confirmation_on_time": sum(x["host_confirmation_on_time"] for x in entries),
            "deadline_failures": sum(x["deadline_failures"] for x in entries),
            "unresolved_tasks": sum(x["unresolved_tasks"] for x in entries),
            "correct_rejections": sum(x["correct_rejections"] for x in entries),
            "security_violations": sum(x["security_violations"] for x in entries),
            "return_sum": sum(sum(ep["return"] for ep in x["episodes"]) for x in entries),
            "latency_samples": sum(x["latency"]["samples"] for x in entries),
            "latency_mean_ms_weighted": sum(x["latency"]["mean_ms"] * x["latency"]["samples"] for x in entries) / sum(x["latency"]["samples"] for x in entries),
        }
        totals[variant]["physical_rate"] = safe_rate(totals[variant]["physical_arrival_on_time"], totals[variant]["tasks"])
        totals[variant]["host_rate"] = safe_rate(totals[variant]["host_confirmation_on_time"], totals[variant]["tasks"])

    # One paired observation per test tape: average the three training-seed
    # rates for each tape, then bootstrap tapes. This is not seed uncertainty.
    tape_diffs = []
    h_tape = {x["tape_id"]: [] for x in final[("H", SEEDS[0])]["episodes"]}
    e_tape = {x["tape_id"]: [] for x in final[("E", SEEDS[0])]["episodes"]}
    for seed in SEEDS:
        for x in final[("H", seed)]["episodes"]:
            h_tape[x["tape_id"]].append(x["physical_arrival_on_time"] / x["tasks"])
        for x in final[("E", seed)]["episodes"]:
            e_tape[x["tape_id"]].append(x["physical_arrival_on_time"] / x["tasks"])
    for tape_id in sorted(h_tape):
        tape_diffs.append(float(statistics.mean(e_tape[tape_id]) - statistics.mean(h_tape[tape_id])))

    diagnostics = load(root / "auxiliary-prediction-diagnostics.json")
    diag_summary = {}
    for family in ("position_x", "position_y", "energy", "task_state"):
        matrices = []
        for run in diagnostics["runs"]:
            matrices.append(np.asarray(run["metrics"][family]["confusion"], dtype=int))
        cm = sum(matrices)
        support = cm.sum(axis=1)
        predicted = cm.sum(axis=0)
        total = int(cm.sum())
        accuracy = float(np.trace(cm) / total) if total else None
        majority = float(support.max() / total) if total else None
        recalls = []
        f1s = []
        for i, row in enumerate(cm):
            if row.sum() == 0:
                continue
            recall = float(cm[i, i] / row.sum())
            precision = float(cm[i, i] / predicted[i]) if predicted[i] else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            recalls.append({"class": i, "support": int(row.sum()), "recall": recall})
            f1s.append(f1)
        diag_summary[family] = {
            "valid_count": total,
            "confusion": cm.tolist(),
            "support": support.tolist(),
            "predicted": predicted.tolist(),
            "accuracy": accuracy,
            "majority_accuracy": majority,
            "accuracy_minus_majority": None if accuracy is None or majority is None else accuracy - majority,
            "macro_f1_supported_classes": float(statistics.mean(f1s)) if f1s else None,
            "supported_class_recall": recalls,
        }

    run_summary = []
    for name in sorted(statuses):
        s = statuses[name]
        i = identities[name]
        run_summary.append({
            "run_id": i["run_id"], "variant": i["variant"], "seed": i["seed"],
            "source_commit": i["source_commit"], "status": s["status"],
            "environment_steps": s["environment_steps"], "optimizer_updates": s["optimizer_updates"],
            "stop_reason": s["stop_reason"], "elapsed_seconds": s["elapsed_seconds"],
            "peak_memory_bytes": s.get("peak_memory_bytes"),
        })

    summary = {
        "schema": "gppo-eawm-auxiliary-matrix-final-v1",
        "matrix_root": str(root),
        "matrix_identity": load(root / "matrix-identity.json"),
        "tape_manifest": load(root / "tape-manifest.json"),
        "runs": run_summary,
        "final_test_by_seed": rows,
        "totals": totals,
        "shared_rule_final_test": {k: rule[k] for k in ("tasks", "physical_arrival_on_time", "host_confirmation_on_time", "variant")},
        "E_minus_H_total": {
            "physical_arrival_on_time": totals["E"]["physical_arrival_on_time"] - totals["H"]["physical_arrival_on_time"],
            "host_confirmation_on_time": totals["E"]["host_confirmation_on_time"] - totals["H"]["host_confirmation_on_time"],
            "deadline_failures": totals["E"]["deadline_failures"] - totals["H"]["deadline_failures"],
            "physical_rate_points": totals["E"]["physical_rate"] - totals["H"]["physical_rate"],
        },
        "parent_tape_bootstrap": {"physical_rate_difference": bootstrap_ci(tape_diffs), "parent_tape_count": len(tape_diffs)},
        "auxiliary_prediction_train_replay_only": {
            "parameter_updates": diagnostics["parameter_updates"],
            "records_per_run": {x["run_id"]: x["records"] for x in diagnostics["runs"]},
            "replay_public_mismatches": {x["run_id"]: x["replay_public_mismatches"] for x in diagnostics["runs"]},
            "pooled": diag_summary,
            "scope": "read-only replay of formal E training ledgers; not validation/test and not independent generalization evidence",
        },
        "notes": [
            "Final test uses fixed last checkpoint at 8192 environment steps; validation curves were not used for checkpoint selection.",
            "Rule result is one shared final-test evaluation, not three pseudo-seeds.",
            "Parent-tape bootstrap covers only the 64 shared final-test tapes; it does not cover training-seed uncertainty.",
            "Communication bytes are serialized simulation proxies, not measured network traffic.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("matrix-identity.json", "matrix-results.json", "frozen-tapes.json", "tape-manifest.json", "auxiliary-prediction-diagnostics.json"):
        (out_dir / name).write_bytes((root / name).read_bytes())

    manifest = {"schema": "gppo-eawm-auxiliary-matrix-artifact-index-v1", "source_root": str(root), "files": []}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            manifest["files"].append({"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    manifest["file_count"] = len(manifest["files"])
    (out_dir / "artifact-index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "summary.json"), "artifact_index": str(out_dir / "artifact-index.json"), "files": len(manifest["files"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
