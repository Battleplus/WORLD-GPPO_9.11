"""One-shot frozen test evaluation for candidate regret models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gppo_world.minimal_consequence_model import CandidateRegretNet, graph_sample


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(model: CandidateRegretNet, rows: list[dict[str, object]], device: torch.device, timing: bool = False) -> tuple[dict[str, object], list[dict[str, object]]]:
    out = []
    elapsed = []
    for row in rows:
        sample = graph_sample(row, device)
        if timing:
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.no_grad(): pred = model(sample)
        if timing:
            if device.type == "cuda": torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - t0) * 1000.0)
        if not torch.isfinite(pred).all(): raise FloatingPointError("non-finite test prediction")
        p = pred.detach().cpu().numpy()
        labels = sample.labels.detach().cpu().numpy()
        chosen = int(np.argmin(p))
        greedy_action = row["greedy_action"]
        greedy_index = next(i for i, c in enumerate(row["candidates"]) if [c["task_id"], c["uav_id"]] == greedy_action)
        best = np.isclose(labels, 0.0, atol=1e-8)
        pair_total = 0; pair_correct = 0
        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[i] < labels[j]: pair_total += 1; pair_correct += int(p[i] < p[j])
        out.append({"instance_id": row["instance"]["instance_id"], "group": row["instance"]["group"], "selected_action": row["candidates"][chosen], "greedy_action": greedy_action, "selected_normalized_regret": float(labels[chosen]), "greedy_normalized_regret": float(labels[greedy_index]), "selected_raw_regret": float(row["candidates"][chosen]["q_star"] - row["c_star"]), "greedy_raw_regret": float(row["candidates"][greedy_index]["q_star"] - row["c_star"]), "regret_reduction_vs_greedy": float(labels[greedy_index] - labels[chosen]), "selected_is_optimal": bool(best[chosen]), "greedy_is_optimal": bool(best[greedy_index]), "non_tie_pairs": pair_total, "non_tie_pair_correct": pair_correct})
    def aggregate(subset: list[dict[str, object]]) -> dict[str, object]:
        diffs = np.asarray([x["regret_reduction_vs_greedy"] for x in subset], dtype=float)
        model_reg = np.asarray([x["selected_normalized_regret"] for x in subset], dtype=float)
        return {"instances": len(subset), "mean_normalized_regret": float(np.mean(model_reg)), "mean_regret_reduction_vs_greedy": float(np.mean(diffs)), "optimal_first_action_rate": float(np.mean([x["selected_is_optimal"] for x in subset])), "greedy_mean_normalized_regret": float(np.mean([x["greedy_normalized_regret"] for x in subset])), "greedy_optimal_first_action_rate": float(np.mean([x["greedy_is_optimal"] for x in subset])), "label_mae_selected": float(np.mean([abs(x["selected_normalized_regret"]) for x in subset])), "non_tie_pair_accuracy": float(sum(x["non_tie_pair_correct"] for x in subset) / sum(x["non_tie_pairs"] for x in subset)) if sum(x["non_tie_pairs"] for x in subset) else None}
    result = {"overall": aggregate(out), "by_group": {g: aggregate([x for x in out if x["group"] == g]) for g in ("homogeneous", "heterogeneous")}}
    if elapsed: result["inference_ms"] = {"mean": float(np.mean(elapsed)), "p95": float(np.percentile(elapsed, 95)), "p99": float(np.percentile(elapsed, 99)), "samples": len(elapsed), "warmup": "10 test instances before timed pass", "scope": "model forward only, no file IO"}
    return result, out


def bootstrap(per_seed: list[list[dict[str, object]]], reps: int, seed: int) -> dict[str, object]:
    by_id = {x["instance_id"]: [] for x in per_seed[0]}
    for rows in per_seed:
        for x in rows: by_id[x["instance_id"]].append(float(x["regret_reduction_vs_greedy"]))
    grouped = {"homogeneous": [], "heterogeneous": []}
    for x in per_seed[0]: grouped[x["group"]].append(x["instance_id"])
    rng = random.Random(seed); values = []
    for _ in range(reps):
        picked = []
        for group_ids in grouped.values():
            picked.extend(rng.choice(group_ids) for _ in group_ids)
        values.append(float(np.mean([np.mean(by_id[i]) for i in picked])))
    values.sort()
    return {"unit": "per-instance mean across three frozen training seeds of normalized regret reduction vs greedy", "groups": {k: len(v) for k, v in grouped.items()}, "replicates": reps, "seed": seed, "ci95": [values[int(0.025 * reps)], values[int(0.975 * reps)]], "does_not_cover_training_seed_uncertainty": True}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--models", nargs=3, type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    data = json.loads(args.data.read_text(encoding="utf-8")); rows = list(data["splits"]["test"]["records"])
    device = torch.device(args.device); per_seed = []; seed_results = []
    for model_path in args.models:
        payload = torch.load(model_path, map_location=device, weights_only=False)
        if payload.get("format") != "m10-minimal-scheduling-candidate-regret-training-v1": raise SystemExit(f"invalid checkpoint {model_path}")
        model = CandidateRegretNet().to(device); model.load_state_dict(payload["model_state_dict"], strict=True); model.eval()
        # Warm-up is separate from the reported timed pass.
        for row in rows[:10]:
            with torch.no_grad(): _ = model(graph_sample(row, device))
        if device.type == "cuda": torch.cuda.synchronize()
        summary, details = metrics(model, rows, device, timing=True)
        seed_results.append({"model": str(model_path.resolve()), "model_sha256": sha256(model_path), "checkpoint_seed": payload["seed"], "summary": summary})
        per_seed.append(details)
    diffs = []
    for rows_seed in per_seed:
        diffs.extend(x["regret_reduction_vs_greedy"] for x in rows_seed)
    final = {"schema": "m10-minimal-scheduling-candidate-regret-test-results-v1", "data": str(args.data.resolve()), "data_sha256": sha256(args.data), "test_instances": len(rows), "rule_baseline": {"overall": {"instances": len(rows), "mean_normalized_regret": float(np.mean([x["greedy_normalized_regret"] for x in per_seed[0]])), "optimal_first_action_rate": float(np.mean([x["greedy_is_optimal"] for x in per_seed[0]]))}, "note": "one shared rule result; not copied into three seeds"}, "per_seed": seed_results, "bootstrap": bootstrap(per_seed, 10000, 20260915), "selection_protocol": "minimum predicted normalized regret; ties follow frozen earliest-finish task-ID/UAV-ID rule", "scope": "first action only with exact optimal continuation; not end-to-end scheduling"}
    dump(args.out, final)
    details_path = args.out.with_name(args.out.stem + "-per-instance.json")
    dump(details_path, {"schema": "m10-minimal-scheduling-candidate-regret-per-instance-v1", "per_seed": [{"model": str(p.resolve()), "rows": r} for p, r in zip(args.models, per_seed)]})
    print(json.dumps({"status": "complete", "out": str(args.out), "instances": len(rows), "bootstrap_ci95": final["bootstrap"]["ci95"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())

