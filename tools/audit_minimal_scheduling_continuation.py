"""Independently replay frozen continuation ledgers and summarize outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np

from generate_minimal_scheduling_continuation_instances import read_scenes, scene_digest
from run_minimal_scheduling_continuation import EXPECTED_CHECKPOINTS, load_instance, replay_validate


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def timing_stats(values: list[float]) -> dict[str, float | int]:
    return {"mean": mean(values), "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)), "samples": len(values)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--generation-identity", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    frozen = json.loads(args.instances.read_text(encoding="utf-8"))
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))["records"]
    prior = json.loads(args.results.read_text(encoding="utf-8"))
    rows = frozen["records"]
    if len(rows) != 128 or len(ledger) != 128 or len({x["instance"]["instance_id"] for x in rows}) != 128:
        raise SystemExit("instance/ledger count or identity uniqueness failure")
    row_by_id = {x["instance"]["instance_id"]: x for x in rows}
    if {x["instance_id"] for x in ledger} != set(row_by_id):
        raise SystemExit("ledger identities differ from frozen instance set")
    generation_identity = json.loads(args.generation_identity.read_text(encoding="utf-8"))
    known_hashes: set[str] = set()
    historical_counts = {}
    for source in generation_identity["historical_sources"]:
        path = Path(source["path"])
        if sha256(path) != source["sha256"]:
            raise SystemExit(f"historical source hash changed: {path}")
        scenes = read_scenes(path)
        historical_counts[str(path)] = len(scenes)
        known_hashes.update(scene_digest(scene) for scene in scenes)
    content_hashes = [x["content_sha256"] for x in rows]
    if len(set(content_hashes)) != 128 or any(value in known_hashes for value in content_hashes):
        raise SystemExit("fresh instance content duplicates historical or within-run scene")
    if len(generation_identity["duplicate_replacements"]) != generation_identity["duplicate_replacement_count"]:
        raise SystemExit("duplicate replacement ledger inconsistency")
    if generation_identity["duplicate_replacement_count"] != 0:
        raise SystemExit("this registered audit expected no duplicate replacements; inspect before interpreting")

    errors = []
    exact_statuses: dict[str, int] = {}
    group_rows = {"homogeneous": [], "heterogeneous": []}
    per_seed_rows = {seed: [] for seed in EXPECTED_CHECKPOINTS}
    inference_ms = {seed: [] for seed in EXPECTED_CHECKPOINTS}
    full_compute_ms = {seed: [] for seed in EXPECTED_CHECKPOINTS}
    greedy_compute_ms: list[float] = []
    solver_seconds: list[float] = []
    all_makespans: dict[str, list[int]] = {"G": []}
    all_makespans.update({f"M{s}": [] for s in EXPECTED_CHECKPOINTS})
    for entry in ledger:
        instance_id = entry["instance_id"]
        if entry.get("status") != "complete":
            errors.append(f"incomplete:{instance_id}:{entry.get('status')}")
            continue
        instance = load_instance(row_by_id[instance_id])
        exact_status = entry["exact_reference"]["status"]
        exact_statuses[exact_status] = exact_statuses.get(exact_status, 0) + 1
        solver_seconds.append(float(entry["exact_reference"]["elapsed_seconds"]))
        if entry["exact_reference"].get("controller_used") is not False:
            errors.append(f"solver_controller_flag:{instance_id}")
        for method_name, schedule in entry["methods"].items():
            try:
                replay = replay_validate(instance, schedule)
                if replay["replayed_makespan"] != schedule["makespan"]:
                    errors.append(f"makespan_replay:{instance_id}:{method_name}")
            except Exception as exc:
                errors.append(f"ledger_replay:{instance_id}:{method_name}:{type(exc).__name__}:{exc}")
            if method_name == "G":
                greedy_compute_ms.append(float(schedule["compute_wall_ms"]))
                if any(x["controller"] != "frozen_greedy_continuation" for x in schedule["decisions"]):
                    errors.append(f"G_not_greedy:{instance_id}")
            else:
                seed = int(method_name[1:])
                if schedule.get("model_forward_calls") != 1:
                    errors.append(f"model_call_count:{instance_id}:{seed}")
                if len(schedule["decisions"]) < 1 or schedule["decisions"][0]["controller"] != "frozen_model_first_action":
                    errors.append(f"missing_model_first_action:{instance_id}:{seed}")
                if any(x["controller"] != "frozen_greedy_continuation" for x in schedule["decisions"][1:]):
                    errors.append(f"non_greedy_suffix:{instance_id}:{seed}")
                evidence = entry["model_selection_evidence"][str(seed)]
                if evidence["selected_action"] != schedule["decisions"][0]["selected_action"]:
                    errors.append(f"model_action_identity:{instance_id}:{seed}")
                if evidence["legal_actions"] != schedule["decisions"][0]["legal_actions"]:
                    errors.append(f"model_initial_mask:{instance_id}:{seed}")
                inference_ms[seed].append(float(schedule["initial_model_inference_ms"]))
                full_compute_ms[seed].append(float(schedule["complete_method_compute_wall_ms"]))
            all_makespans[method_name].append(int(schedule["makespan"]))
            if exact_status == "optimal":
                if schedule["optimal_regret"] != int(schedule["makespan"]) - int(entry["exact_reference"]["makespan"]):
                    errors.append(f"optimal_regret_mismatch:{instance_id}:{method_name}")
            elif schedule["optimal_regret"] is not None:
                errors.append(f"unproven_optimal_gap_present:{instance_id}:{method_name}")
        group_rows[instance.group].append(entry)
        for seed in EXPECTED_CHECKPOINTS:
            delta = int(entry["methods"]["G"]["makespan"]) - int(entry["methods"][f"M{seed}"]["makespan"])
            per_seed_rows[seed].append((entry, delta))
    expected_params = prior["execution_integrity"]["model_parameter_state_sha256_before"]
    if expected_params != prior["execution_integrity"]["model_parameter_state_sha256_after"]:
        errors.append("model_parameter_state_hash_changed")
    if prior["execution_integrity"]["checkpoint_file_sha256_before"] != prior["execution_integrity"]["checkpoint_file_sha256_after"]:
        errors.append("checkpoint_file_hash_changed")
    if prior["execution_integrity"]["exact_solver_controller_calls"] != 0:
        errors.append("solver_controller_calls_nonzero")

    metrics: dict[str, object] = {}
    for group in ("overall", "homogeneous", "heterogeneous"):
        subset = ledger if group == "overall" else [x for x in ledger if x.get("group") == group]
        method_metrics = {}
        for method_name in ["G", "M1101", "M2203", "M3307"]:
            times = [int(x["methods"][method_name]["makespan"]) for x in subset if x.get("status") == "complete"]
            method_metrics[method_name] = {
                "instances": len(times), "mean_makespan": mean(times) if times else None,
                "optimal_regret_mean": mean([x["methods"][method_name]["optimal_regret"] for x in subset]) if times and all(x["methods"][method_name]["optimal_regret"] is not None for x in subset) else None,
            }
        for seed in EXPECTED_CHECKPOINTS:
            sr = [x for x in subset if x.get("status") == "complete"]
            diffs = [int(x["methods"]["G"]["makespan"]) - int(x["methods"][f"M{seed}"]["makespan"]) for x in sr]
            method_metrics[f"M{seed}"].update({
                "G_minus_M_mean": mean(diffs) if diffs else None,
                "relative_reduction": mean(diffs) / method_metrics["G"]["mean_makespan"] if diffs and method_metrics["G"]["mean_makespan"] else None,
                "wins_ties_losses": [sum(v > 0 for v in diffs), sum(v == 0 for v in diffs), sum(v < 0 for v in diffs)],
            })
        metrics[group] = method_metrics
    per_instance_mean_delta = {}
    for row in ledger:
        if row.get("status") == "complete":
            per_instance_mean_delta[row["instance_id"]] = mean(
                int(row["methods"]["G"]["makespan"]) - int(row["methods"][f"M{s}"]["makespan"])
                for s in EXPECTED_CHECKPOINTS
            )
    rng = np.random.default_rng(20260916)
    samples = np.empty(10000)
    strata = {g: [x for x in ledger if x.get("status") == "complete" and x["group"] == g] for g in ("homogeneous", "heterogeneous")}
    for i in range(len(samples)):
        values = []
        for rows_g in strata.values():
            draw = rng.integers(0, len(rows_g), size=len(rows_g))
            values.extend(per_instance_mean_delta[rows_g[j]["instance_id"]] for j in draw)
        samples[i] = mean(values) if values else float("nan")
    timing = {
        "G_complete_schedule_compute_wall_ms": timing_stats(greedy_compute_ms),
        "initial_model_forward_ms_by_seed": {str(seed): timing_stats(v) for seed, v in inference_ms.items()},
        "complete_model_first_plus_greedy_wall_ms_by_seed": {str(seed): timing_stats(v) for seed, v in full_compute_ms.items()},
        "exact_solver_wall_seconds": timing_stats(solver_seconds),
        "note": "No model warmup; CUDA synchronize brackets forward timing. These are compute wall times, not simulated makespan. File output excluded.",
    }
    output = {
        "schema": "m10-minimal-scheduling-continuation-independent-audit-v1",
        "source": {"instances_sha256": sha256(args.instances), "ledger_sha256": sha256(args.ledger), "results_sha256": sha256(args.results), "historical_scene_counts": historical_counts},
        "freshness": {"new_instances": len(rows), "within_new_duplicate_content": len(content_hashes) - len(set(content_hashes)), "historical_content_overlap": sum(x in known_hashes for x in content_hashes), "duplicate_replacements": generation_identity["duplicate_replacement_count"]},
        "exact_status_counts": exact_statuses,
        "completed_instances": sum(x.get("status") == "complete" for x in ledger),
        "methods_by_group": metrics,
        "bootstrap": {"unit": "within instance mean of G-M across three seeds, stratified instance bootstrap", "strata": {k: len(v) for k, v in strata.items()}, "replicates": len(samples), "seed": 20260916, "mean_time_reduction": mean(per_instance_mean_delta.values()) if per_instance_mean_delta else None, "ci95_time_units": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))] if len(per_instance_mean_delta) == 128 else None, "does_not_cover_training_seed_uncertainty": True},
        "timing": timing,
        "integrity_errors": errors,
        "integrity_pass": not errors,
        "frozen_checkpoint_sha256": EXPECTED_CHECKPOINTS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_root = args.instances.parent.parent
    manifest_rows = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            manifest_rows.append({
                "path": path.relative_to(run_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest = {
        "schema": "m10-minimal-scheduling-continuation-run-artifact-manifest-v1",
        "files": manifest_rows,
        "file_count": len(manifest_rows),
    }
    (run_root / "artifact-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"integrity_pass": output["integrity_pass"], "completed": output["completed_instances"], "errors": len(errors), "exact": exact_statuses, "bootstrap": output["bootstrap"]["ci95_time_units"]}, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
