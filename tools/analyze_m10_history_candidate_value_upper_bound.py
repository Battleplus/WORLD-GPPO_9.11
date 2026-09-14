"""Post-hoc analysis for the frozen-H candidate-value development run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def proxy_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.run.joinpath("selections.jsonl").read_text(encoding="utf-8").splitlines()]
    branches: dict[tuple[str, int, int], dict[str, Any]] = {}
    total_steps = 0
    safety = 0
    with gzip.open(args.run / "branches.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["prefix"]["prefix_id"], int(row["repeat"]), int(row["candidate"]["action"]))
            branches[key] = row
            total_steps += len(row["branch"]["steps"])
            safety += int(row["branch"]["safety"]["violations"])

    def outcome(row: dict[str, Any], metric: str) -> float:
        return float(row["branch"][metric])

    def branch_for(prefix: str, repeat: int, action: int) -> dict[str, Any]:
        return branches[(prefix, repeat, action)]

    selected_physical: list[float] = []
    selected_host: list[float] = []
    h_physical: list[float] = []
    h_host: list[float] = []
    selected_bytes: list[float] = []
    h_bytes: list[float] = []
    selected_actions = []
    h_actions = []
    rule_actions = []
    discovery_gaps = []
    direction_consistent = 0
    complete = []
    for row in rows:
        prefix = row["prefix_id"]
        selected = int(row["selected_action"])
        h = int(row["h_action"])
        rule = int(row["rule_action"])
        selected_actions.append(selected)
        h_actions.append(h)
        rule_actions.append(rule)
        discovery_gaps.append(float(row["discovery_best_minus_h"]))
        conf_selected = [branch_for(prefix, repeat, selected) for repeat in (2, 3)]
        conf_h = [branch_for(prefix, repeat, h) for repeat in (2, 3)]
        sp = float(np.mean([outcome(x, "physical_arrival_on_time") for x in conf_selected]))
        hp = float(np.mean([outcome(x, "physical_arrival_on_time") for x in conf_h]))
        sh = float(np.mean([outcome(x, "host_confirmation_on_time") for x in conf_selected]))
        hh = float(np.mean([outcome(x, "host_confirmation_on_time") for x in conf_h]))
        sb = float(np.mean([proxy_bytes(x["branch"]["communication_log"]) for x in conf_selected]))
        hb = float(np.mean([proxy_bytes(x["branch"]["communication_log"]) for x in conf_h]))
        selected_physical.append(sp); h_physical.append(hp); selected_host.append(sh); h_host.append(hh)
        selected_bytes.append(sb); h_bytes.append(hb)
        if sp - hp > 0: relation = "better"
        elif sp - hp < 0: relation = "worse"
        else: relation = "tie"
        complete.append({"parent_tape_id": row["parent_tape_id"], "prefix_id": prefix, "selected_action": selected, "h_action": h, "rule_action": rule, "physical_delta": sp - hp, "host_delta": sh - hh, "communication_proxy_bytes_delta": sb - hb, "relation": relation})
        candidate_values = row["candidate_direction_consistent"]["per_action"]
        if all(len(set(v["values"])) <= 1 for v in candidate_values.values()):
            direction_consistent += 1

    parent_means = {}
    for item in complete:
        parent_means.setdefault(item["parent_tape_id"], []).append(item["physical_delta"])
    parent_values = [float(np.mean(v)) for v in parent_means.values()]
    rng = random.Random(20260915)
    boot = sorted(float(np.mean([parent_values[rng.randrange(len(parent_values))] for _ in parent_values])) for _ in range(10000)) if parent_values else []
    summary = {
        "schema": "m10-history-candidate-value-upper-bound-analysis-v1",
        "run": str(args.run.resolve()),
        "parent_count": len({r["parent_tape_id"] for r in rows}), "prefix_count": len(rows),
        "candidate_count_distribution": {str(k): sum(len(r["candidate_actions"]) == k for r in rows) for k in sorted({len(r["candidate_actions"]) for r in rows})},
        "prefixes_with_at_least_two_candidates": sum(len(r["candidate_actions"]) >= 2 for r in rows),
        "branches": len(branches), "branch_environment_steps": total_steps, "branch_budget_check": total_steps <= 50000,
        "selected_equals_h_fraction": float(np.mean([a == b for a, b in zip(selected_actions, h_actions)])),
        "selected_equals_rule_fraction": float(np.mean([a == b for a, b in zip(selected_actions, rule_actions)])),
        "discovery_best_minus_h_mean": float(np.mean(discovery_gaps)),
        "confirmation_physical": {"mean_selected_minus_h": float(np.mean(np.asarray(selected_physical) - np.asarray(h_physical))), "better_prefixes": int(np.sum(np.asarray(selected_physical) - np.asarray(h_physical) > 0)), "tie_prefixes": int(np.sum(np.asarray(selected_physical) - np.asarray(h_physical) == 0)), "worse_prefixes": int(np.sum(np.asarray(selected_physical) - np.asarray(h_physical) < 0))},
        "confirmation_host": {"mean_selected_minus_h": float(np.mean(np.asarray(selected_host) - np.asarray(h_host))), "better_prefixes": int(np.sum(np.asarray(selected_host) - np.asarray(h_host) > 0)), "tie_prefixes": int(np.sum(np.asarray(selected_host) - np.asarray(h_host) == 0)), "worse_prefixes": int(np.sum(np.asarray(selected_host) - np.asarray(h_host) < 0))},
        "confirmation_communication_proxy": {"unit": "serialized JSON bytes of branch communication log; not real network traffic", "mean_selected_minus_h_bytes": float(np.mean(np.asarray(selected_bytes) - np.asarray(h_bytes)))},
        "parent_bootstrap_physical": {"unit": "parent-tape mean of confirmation selected-minus-H physical on-time task count", "parents": len(parent_values), "replicates": 10000, "seed": 20260915, "ci95": [boot[500], boot[9500]] if boot else [None, None]},
        "safety_violations": safety,
        "direction_constant_across_four_repeats": direction_consistent,
        "prefix_results": complete,
        "predeclared_signal": {"complete_parent_at_least_12": len(parent_values) >= 12, "better_parent_at_least_6": len({x["parent_tape_id"] for x in complete if x["physical_delta"] > 0}) >= 6, "overall_mean_gt_zero": bool(complete and np.mean(np.asarray(selected_physical) - np.asarray(h_physical)) > 0), "no_safety_or_ledger_gap": safety == 0 and total_steps <= 50000, "all_conditions": bool(len(parent_values) >= 12 and len({x["parent_tape_id"] for x in complete if x["physical_delta"] > 0}) >= 6 and complete and np.mean(np.asarray(selected_physical) - np.asarray(h_physical)) > 0 and safety == 0 and total_steps <= 50000)},
        "interpretation": "All results are fixed-H, finite-candidate, development evidence. Discovery repeats selected the candidate; confirmation repeats evaluated it. No branch truth entered H input.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump(args.out, summary)
    print(json.dumps({"status": "written", "out": str(args.out), "summary": summary["confirmation_physical"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
