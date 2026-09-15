"""Recompute frozen three-seed candidate regret metrics from saved test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_bootstrap(
    differences_by_id: dict[str, list[float]],
    groups_by_id: dict[str, str],
    reps: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {"homogeneous": [], "heterogeneous": []}
    for instance_id, group in groups_by_id.items():
        groups[group].append(instance_id)
    samples: list[float] = []
    for _ in range(reps):
        picked: list[str] = []
        for instance_ids in groups.values():
            picked.extend(rng.choice(instance_ids) for _ in instance_ids)
        samples.append(mean(mean(differences_by_id[k]) for k in picked))
    samples.sort()
    return [samples[int(0.025 * reps)], samples[int(0.975 * reps)]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260915)
    args = ap.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    seed_rows = payload["per_seed"]
    if len(seed_rows) != 3:
        raise SystemExit(f"expected 3 frozen model seeds, got {len(seed_rows)}")
    differences_by_id: dict[str, list[float]] = {}
    normalized_by_id: dict[str, list[float]] = {}
    groups_by_id: dict[str, str] = {}
    summaries = []
    reference_ids: set[str] | None = None
    for seed_result in seed_rows:
        rows = seed_result["rows"]
        model_path = Path(seed_result["model"])
        model_hash = sha256(model_path) if model_path.is_file() else "unavailable"
        if model_hash == "unavailable":
            raise SystemExit(f"frozen checkpoint unavailable: {model_path}")
        ids = {row["instance_id"] for row in rows}
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise SystemExit("seed test instance identities differ")
        raw_model = [float(row["selected_raw_regret"]) for row in rows]
        raw_greedy = [float(row["greedy_raw_regret"]) for row in rows]
        norm_model = [float(row["selected_normalized_regret"]) for row in rows]
        norm_greedy = [float(row["greedy_normalized_regret"]) for row in rows]
        raw_diff = []
        for row in rows:
            instance_id = row["instance_id"]
            differences_by_id.setdefault(instance_id, []).append(
                float(row["greedy_raw_regret"]) - float(row["selected_raw_regret"])
            )
            normalized_by_id.setdefault(instance_id, []).append(
                float(row["greedy_normalized_regret"]) - float(row["selected_normalized_regret"])
            )
            groups_by_id[instance_id] = row["group"]
            raw_diff.append(float(row["greedy_raw_regret"]) - float(row["selected_raw_regret"]))
        summaries.append(
            {
                "seed": int(model_path.parent.name.rsplit("seed", 1)[-1]),
                "checkpoint_sha256": model_hash,
                "instances": len(rows),
                "mean_model_raw_regret": mean(raw_model),
                "mean_shared_greedy_raw_regret": mean(raw_greedy),
                "mean_raw_regret_reduction": mean(raw_diff),
                "relative_raw_regret_reduction": mean(raw_diff) / mean(raw_greedy) if mean(raw_greedy) else None,
                "model_mean_normalized_regret": mean(norm_model),
                "shared_greedy_mean_normalized_regret": mean(norm_greedy),
                "mean_normalized_regret_reduction": mean(norm_greedy) - mean(norm_model),
                "instances_better_equal_worse": [sum(x > 0 for x in raw_diff), sum(x == 0 for x in raw_diff), sum(x < 0 for x in raw_diff)],
            }
        )
    raw_ci = stratified_bootstrap(differences_by_id, groups_by_id, args.reps, args.bootstrap_seed)
    normalized_ci = stratified_bootstrap(normalized_by_id, groups_by_id, args.reps, args.bootstrap_seed)
    greedy_raw = mean(summaries[0]["mean_shared_greedy_raw_regret"] for _ in summaries)
    mean_model_raw = mean(item["mean_model_raw_regret"] for item in summaries)
    relative_reduction = (greedy_raw - mean_model_raw) / greedy_raw if greedy_raw else None
    result = {
        "schema": "m10-minimal-consequence-raw-regret-recomputation-v1",
        "source_results": str(args.results.resolve()),
        "source_results_sha256": sha256(args.results),
        "checkpoint_sha256": [item["checkpoint_sha256"] for item in summaries],
        "paired_instances": len(differences_by_id),
        "bootstrap": {
            "unit": "instance; average three seed paired differences within instance, then stratified resample instances",
            "groups": {group: sum(value == group for value in groups_by_id.values()) for group in ("homogeneous", "heterogeneous")},
            "replicates": args.reps,
            "seed": args.bootstrap_seed,
            "raw_regret_reduction_ci95_time_units": raw_ci,
            "normalized_regret_reduction_ci95": normalized_ci,
            "does_not_cover_training_seed_population_uncertainty": True,
        },
        "shared_greedy_mean_raw_regret": greedy_raw,
        "three_seed_mean_model_raw_regret": mean_model_raw,
        "three_seed_relative_raw_regret_reduction": relative_reduction,
        "per_seed": summaries,
        "preregistered_raw_gate": {
            "relative_reduction_at_least_10_percent": bool(relative_reduction is not None and relative_reduction >= 0.10),
            "at_least_two_of_three_seed_better": sum(x["mean_raw_regret_reduction"] > 0 for x in summaries) >= 2,
            "paired_ci_lower_bound_above_zero": raw_ci[0] > 0,
            "pass": bool(relative_reduction is not None and relative_reduction >= 0.10 and sum(x["mean_raw_regret_reduction"] > 0 for x in summaries) >= 2 and raw_ci[0] > 0),
        },
        "prior_reported_normalized_ci95": [0.008021811159172406, 0.039343537811267495],
        "prior_normalized_relative_reduction_claim_percent": 57.44,
        "interpretation": "Normalized regret is secondary. The primary outcome is unnormalized simulation-time regret.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pass": result["preregistered_raw_gate"]["pass"], "raw_relative_reduction": relative_reduction, "raw_ci95": raw_ci, "normalized_ci95": normalized_ci}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
