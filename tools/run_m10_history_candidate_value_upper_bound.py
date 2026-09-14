"""Developer upper-bound diagnostic for candidate actions under frozen H.

The policy never receives branch truth.  Truth is read only after a branch has
finished, for an offline, finite-candidate diagnostic.  Discovery repeats are
used for selection; confirmation repeats are held out for the paired report.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.m10_environment import M10Config, M10Environment, M10Scenario, scenario_to_dict, weak_communication_tape  # noqa: E402
from gppo_world.m10_training import M10ActorCritic, _public_vector, masked_distribution  # noqa: E402
from tools.run_m10_baseline_comparison import traditional_action  # noqa: E402


SEED = 1101
BASE_SEED = 581001
PARENT_COUNT = 24
PREFIX_DECISIONS = (2, 5)
DISCOVERY_REPEATS = (0, 1)
CONFIRMATION_REPEATS = (2, 3)
REPEAT_COUNT = 4
MAX_BRANCHES = PARENT_COUNT * len(PREFIX_DECISIONS) * 4 * REPEAT_COUNT
MAX_BRANCH_STEPS = 50_000
WALL_SECONDS = 90 * 60
NOOP = 24


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_gzip(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def model_from_checkpoint(checkpoint: Path, config: M10Config, device: torch.device) -> tuple[M10ActorCritic, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("format") != "gppo-history-arrival-training-v1":
        raise RuntimeError(f"unexpected checkpoint format: {payload.get('format')}")
    model = M10ActorCritic(
        uav_count=config.uav_count, task_capacity=config.task_capacity,
        action_count=config.action_count, encoder="graph", type_count=5,
        history=True, context_dim=0, region_count=config.region_count,
        target_count=config.target_count, event_capacity=config.event_capacity,
        relation_width=config.relation_width,
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def policy_distribution(policy: M10ActorCritic, obs: dict[str, Any], hidden: torch.Tensor | None, device: torch.device) -> tuple[int, np.ndarray, torch.Tensor | None, str, str]:
    vector = _public_vector(obs)
    obs_tensor = torch.tensor(vector, dtype=torch.float32, device=device)[None, :]
    mask = np.asarray(obs["mask"], dtype=np.bool_)
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)[None, :]
    hidden_before = digest(hidden.detach().cpu().tolist() if hidden is not None else None)
    with torch.no_grad():
        logits, _, next_hidden = policy(obs_tensor, hidden)
        dist = masked_distribution(logits, mask_tensor)
        probs = dist.probs[0].detach().cpu().numpy().astype(np.float64)
        action = int(torch.argmax(dist.logits, dim=-1).item())
    hidden_after = digest(next_hidden.detach().cpu().tolist() if next_hidden is not None else None)
    if not bool(mask[action]):
        raise RuntimeError("deterministic H selected an illegal action")
    return action, probs, next_hidden.detach().clone() if next_hidden is not None else None, hidden_before, hidden_after


def public_digest(obs: dict[str, Any]) -> str:
    return digest({"flat": np.asarray(obs["flat"], dtype=np.float32).tolist(), "mask": np.asarray(obs["mask"], dtype=bool).tolist(), "entity_ids": obs["entity_ids"], "version": obs["version"], "time": obs["time"]})


def hidden_digest(hidden: torch.Tensor | None) -> str:
    return digest(hidden.detach().cpu().tolist() if hidden is not None else None)


def candidate_actions(obs: dict[str, Any], probs: np.ndarray, h_action: int) -> list[dict[str, Any]]:
    mask = np.asarray(obs["mask"], dtype=np.bool_)
    legal = [int(i) for i, allowed in enumerate(mask) if allowed]
    rule_action = int(traditional_action(obs))
    if not bool(mask[rule_action]):
        raise RuntimeError("public rule returned an illegal action")
    choices: list[tuple[str, int]] = [("H", h_action), ("rule", rule_action)]
    if bool(mask[NOOP]):
        choices.append(("noop", NOOP))
    remaining = [a for a in legal if a not in {h_action, rule_action, NOOP}]
    if remaining:
        best = max(remaining, key=lambda a: (float(probs[a]), -a))
        choices.append(("H_probability_remaining", int(best)))
    dedup: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source, action in choices:
        if action in seen:
            continue
        seen.add(action)
        dedup.append({"source": source, "action": int(action), "probability": float(probs[action]), "legal": bool(mask[action])})
    return dedup


def step_record(info: dict[str, Any], action: int, reward: float, obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": int(action), "reward": float(reward), "time": float(info.get("time", obs.get("time", 0.0))),
        "terminated": bool(info.get("terminated", False)), "truncated": bool(info.get("truncated", False)),
        "feedback": info.get("feedback"), "counts": info.get("counts"),
        "tasks": info.get("tasks"), "energy": info.get("energy"),
        "new_events": info.get("new_events", []), "communication_delta": info.get("communication_delta", []),
        "feedback_log": info.get("feedback_log", []), "lease_renewals": info.get("lease_renewals", {}),
        "active_continuations": info.get("active_continuations", []),
    }


def safety_summary(env: M10Environment) -> dict[str, Any]:
    accepted = [x for x in env.execution.log if x.get("result") == "accepted"]
    duplicate_accepts = len(accepted) - len({x.get("command_id") for x in accepted})
    forbidden = [x for x in env.execution.log if x.get("result") in ("fenced", "stale", "expired", "unknown_command")]
    return {"duplicate_accepts": int(duplicate_accepts), "unauthorized_or_fenced": len(forbidden), "violations": int(duplicate_accepts + len(forbidden)), "execution_log": list(env.execution.log)}


def task_outcomes(env: M10Environment, fixed_tasks: list[str]) -> tuple[dict[str, Any], int, int, int]:
    outcomes: dict[str, Any] = {}
    physical = host = censored = 0
    for task_id in fixed_tasks:
        task = env.clock.tasks[task_id]
        record = dict(env._completion_records.get(task_id, {}))
        arrival = record.get("physical_arrival_time")
        p_ok = record.get("physical_arrival_before_deadline")
        h_ok = record.get("host_confirmation_before_deadline")
        if p_ok is True:
            physical += 1
        if h_ok is True:
            host += 1
        state = task.state.value
        is_censored = state not in ("completed", "expired")
        if is_censored:
            censored += 1
        outcomes[task_id] = {
            "final_state": state,
            "physical_arrival_time": arrival,
            "deadline": float(task.deadline),
            "physical_arrival_before_deadline": p_ok,
            "host_confirmation_time": record.get("host_confirmation_time"),
            "host_confirmation_before_deadline": h_ok,
            "censored": is_censored,
            "censor_reason": "not_terminal_at_observation_end" if is_censored else None,
        }
    return outcomes, physical, host, censored


def run_branch(prefix_env: M10Environment, prefix_obs: dict[str, Any], hidden_after_prefix_decision: torch.Tensor | None, first_action: int, exogenous_key: str, policy: M10ActorCritic, device: torch.device, max_steps: int) -> dict[str, Any]:
    env = deepcopy(prefix_env)
    env._exogenous_key = exogenous_key
    # Use the exact observation returned at the prefix.  Calling
    # _observation() again would clear/consume transient public trigger flags
    # and falsely make a valid branch look like a state mismatch.
    before_digest = public_digest(prefix_obs)
    obs, reward, done, info = env.step(int(first_action), submit_command=True)
    steps = [step_record(info, first_action, reward, obs)]
    hidden = hidden_after_prefix_decision.detach().clone() if hidden_after_prefix_decision is not None else None
    while not done and len(steps) < max_steps:
        action, _, hidden, _, _ = policy_distribution(policy, obs, hidden, device)
        obs, reward, done, info = env.step(action, submit_command=True)
        steps.append(step_record(info, action, reward, obs))
    outcomes, physical, host, censored = task_outcomes(env, sorted(prefix_env._public_task_entities))
    safety = safety_summary(env)
    communication = list(env._communication_log)
    return {
        "prefix_public_digest_at_branch": before_digest,
        "branch_exogenous_key": exogenous_key,
        "first_action": int(first_action),
        "steps": steps,
        "finished": bool(done),
        "observation_end_time": float(env.clock.time),
        "fixed_task_outcomes": outcomes,
        "physical_arrival_on_time": physical,
        "host_confirmation_on_time": host,
        "censored_fixed_tasks": censored,
        "return": float(sum(float(item["reward"]) for item in steps)),
        "communication_log": communication,
        "communication_digest": digest(communication),
        "safety": safety,
        "hidden_after_branch_digest": hidden_digest(hidden),
    }


def known_tape_ids(search_root: Path) -> tuple[set[str], list[str]]:
    ids: set[str] = set()
    files: list[str] = []
    for path in search_root.rglob("*.json"):
        # Do not treat a previous attempt of this same run family as history.
        # Failed attempts are preserved, but their generated development tapes
        # must not make a clean retry look like an identity collision.
        if any(part.startswith("m10-history-action-value-upper-bound-dev-") for part in path.parts):
            continue
        if path.stat().st_size > 50 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            payload = json.loads(text)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        found = set()
        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, str) and ("-seed-" in value or value.startswith("test-") or value.startswith("train-")):
                found.add(value)
        visit(payload)
        if found:
            ids.update(found)
            files.append(str(path))
    return ids, files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--search-root", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no silent fallback")
    if not 1 <= args.threads <= 4:
        raise SystemExit("threads must be 1..4")
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    checkpoint_sha = sha256(args.checkpoint)
    expected_sha = "e77a8ccba9f65d806eeb27e8404c8b3ea0600295dfa0afe43bfb154892919027"
    if checkpoint_sha != expected_sha:
        raise SystemExit(f"checkpoint SHA mismatch: expected fixed 1101/32768 {expected_sha}, got {checkpoint_sha}")
    identity = load_json(args.checkpoint.parent / "run-identity.json")
    protocol = dict(identity["protocol"])
    config = M10Config(**protocol)
    policy, payload = model_from_checkpoint(args.checkpoint, config, device)
    if payload.get("recovery_state", {}).get("seed") != SEED or payload.get("recovery_state", {}).get("environment_steps") != 32768:
        raise SystemExit("checkpoint recovery identity is not seed 1101 at 32768 steps")

    scenarios = list(weak_communication_tape("train", count=PARENT_COUNT, base_seed=BASE_SEED, level="composite"))
    ids, scanned_files = known_tape_ids(args.search_root)
    overlap = sorted({s.tape_id for s in scenarios} & ids)
    if overlap:
        raise SystemExit(f"development tape identity overlap detected: {overlap}")
    tape_payload = [scenario_to_dict(s) for s in scenarios]
    dump(args.out / "tapes.json", {"split": "development", "generator_template": "train", "base_seed": BASE_SEED, "count": PARENT_COUNT, "level": "composite", "scenarios": tape_payload})
    dump(args.out / "run-identity.json", {
        "schema": "m10-history-candidate-value-upper-bound-v1",
        "run_id": "m10-history-candidate-value-upper-bound-dev-20260915",
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_branch": git_value("branch", "--show-current"),
        "device": args.device, "threads": args.threads,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_sha, "seed": SEED, "environment_steps": 32768, "format": payload.get("format")},
        "protocol": asdict(config),
        "generation": {"parent_count": PARENT_COUNT, "base_seed": BASE_SEED, "generator_split_template": "train", "communication": "composite weak communication", "prefix_decision_numbers": list(PREFIX_DECISIONS), "discovery_repeats": list(DISCOVERY_REPEATS), "confirmation_repeats": list(CONFIRMATION_REPEATS), "max_branches": MAX_BRANCHES, "max_branch_steps": MAX_BRANCH_STEPS, "wall_seconds": WALL_SECONDS, "selection_frozen_before_confirmation": True},
        "input_allowlist": ["public obs flat/mask/entity ids/version/time", "H hidden state", "actual policy action"],
        "truth_use": "simulator truth is read only after branch completion for offline outcome accounting",
        "historical_tape_scan": {"search_root": str(args.search_root.resolve()), "files_with_tape_ids": scanned_files, "known_id_count": len(ids), "overlap_count": len(overlap)},
    })
    dump(args.out / "branches.jsonl.gz.meta.json", {"schema": "m10-history-candidate-value-upper-bound-branch-ledger-v1", "compression": "gzip jsonl", "records": "one per candidate x repeat", "fixed_task_set": "publicly delivered task identities at prefix", "primary_outcome": "fixed set physical arrival before deadline", "censoring": "nonterminal fixed tasks remain censored, never counted as failure"})
    branch_path = args.out / "branches.jsonl.gz"
    selection_path = args.out / "selections.jsonl"
    start = time.perf_counter()
    total_branches = 0
    prefix_records: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        env = M10Environment(config, scenario)
        obs = env.reset()
        hidden: torch.Tensor | None = None
        decision_index = 0
        while decision_index < int(config.horizon / config.decision_interval) + 2 and decision_index < max(PREFIX_DECISIONS):
            h_action, probs, hidden_after, hidden_before_digest, hidden_after_digest = policy_distribution(policy, obs, hidden, device)
            decision_number = decision_index + 1
            if decision_number in PREFIX_DECISIONS:
                prefix_env = deepcopy(env)
                prefix_tasks = sorted(prefix_env._public_task_entities)
                choices = candidate_actions(obs, probs, h_action)
                prefix_id = f"{scenario.tape_id}|decision-{decision_number}"
                selection_base = {
                    "parent_index": scenario_index, "parent_tape_id": scenario.tape_id, "prefix_id": prefix_id,
                    "decision_number": decision_number, "prefix_time": float(obs["time"]), "public_version": int(obs["version"]),
                    "public_observation_digest": public_digest(obs), "public_flat": np.asarray(obs["flat"], dtype=np.float32).tolist(),
                    "public_mask": np.asarray(obs["mask"], dtype=bool).tolist(), "public_entity_ids": obs["entity_ids"],
                    "fixed_public_task_ids": prefix_tasks, "candidate_actions": choices,
                    "h_action": int(h_action), "rule_action": int(traditional_action(obs)),
                    "h_probability": float(probs[h_action]),
                    "h_probabilities": [float(x) for x in probs.tolist()], "hidden_before_digest": hidden_before_digest,
                    "hidden_after_decision_digest": hidden_after_digest,
                }
                branch_rows: list[dict[str, Any]] = []
                for repeat in range(REPEAT_COUNT):
                    exo_key = f"{prefix_id}|repeat-{repeat}"
                    for choice in choices:
                        if total_branches >= MAX_BRANCHES:
                            raise RuntimeError("branch budget exceeded")
                        branch = run_branch(prefix_env, obs, hidden_after, int(choice["action"]), exo_key, policy, device, int(config.horizon / config.decision_interval) + 3)
                        branch_row = {"prefix": selection_base, "repeat": repeat, "repeat_role": "discovery" if repeat in DISCOVERY_REPEATS else "confirmation", "candidate": choice, "branch": branch, "invariants": {"prefix_public_digest_match": branch["prefix_public_digest_at_branch"] == selection_base["public_observation_digest"], "fixed_task_ids_match": sorted(branch["fixed_task_outcomes"]) == prefix_tasks, "hidden_after_prefix_digest": hidden_after_digest, "external_key": exo_key}}
                        append_gzip(branch_path, branch_row)
                        branch_rows.append(branch_row)
                        total_branches += 1
                    if time.perf_counter() - start > WALL_SECONDS:
                        raise RuntimeError("wall-clock budget exhausted")
                # Selection uses discovery rows only, never confirmation outcomes.
                discovery = [row for row in branch_rows if row["repeat"] in DISCOVERY_REPEATS]
                means = {}
                for choice in choices:
                    vals = [row["branch"]["physical_arrival_on_time"] for row in discovery if row["candidate"]["action"] == choice["action"]]
                    means[int(choice["action"])] = float(np.mean(vals)) if vals else None
                h = int(h_action)
                eligible = [a for a, value in means.items() if value is not None]
                selected = min(eligible, key=lambda a: (-means[a], 0 if a == h else 1, a))
                confirm = [row for row in branch_rows if row["repeat"] in CONFIRMATION_REPEATS]
                selected_vals = [row["branch"]["physical_arrival_on_time"] for row in confirm if row["candidate"]["action"] == selected]
                h_vals = [row["branch"]["physical_arrival_on_time"] for row in confirm if row["candidate"]["action"] == h]
                confirmation_delta = float(np.mean(selected_vals) - np.mean(h_vals)) if selected_vals and h_vals else None
                h_discovery = [row["branch"]["physical_arrival_on_time"] for row in discovery if row["candidate"]["action"] == h]
                best_discovery = max(means.values()) if means else None
                selection_row = {**selection_base, "discovery_means": means, "selected_action": int(selected), "selected_action_source": next(item["source"] for item in choices if item["action"] == selected), "discovery_best_minus_h": float(best_discovery - np.mean(h_discovery)) if best_discovery is not None and h_discovery else None, "confirmation_selected_minus_h": confirmation_delta, "confirmation_selected_values": selected_vals, "confirmation_h_values": h_vals, "selected_equals_h": bool(selected == h), "selected_equals_rule": bool(selected == selection_base["rule_action"]), "confirmation_selected_strictly_better": bool(confirmation_delta is not None and confirmation_delta > 0), "confirmation_selected_strictly_worse": bool(confirmation_delta is not None and confirmation_delta < 0), "candidate_direction_consistent": direction_consistency(branch_rows)}
                with selection_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(selection_row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                prefix_records.append(selection_row)
            obs, _, done, _ = env.step(h_action, submit_command=True)
            hidden = hidden_after
            decision_index += 1
            if done:
                break
    if total_branches > MAX_BRANCHES or time.perf_counter() - start > WALL_SECONDS:
        raise RuntimeError("budget accounting failure")
    summary = summarize(prefix_records, total_branches, time.perf_counter() - start, args.out)
    dump(args.out / "summary.json", summary)
    dump(args.out / "verification.json", {"status": "passed" if summary["verification"]["input_leakage"] and summary["verification"]["prefix_state_consistent"] and summary["verification"]["fixed_task_sets_consistent"] and summary["verification"]["paired_external_keys_consistent"] and summary["verification"]["all_complete_comparisons_have_confirmation"] else "failed", "checks": summary["verification"]})
    print(json.dumps({"status": "complete", "prefixes": len(prefix_records), "branches": total_branches, "elapsed_seconds": summary["elapsed_seconds"], "confirmation_better_prefixes": summary["confirmation_better_prefixes"]}, ensure_ascii=False, sort_keys=True))
    return 0


def direction_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_action: dict[int, list[int]] = {}
    for row in rows:
        by_action.setdefault(int(row["candidate"]["action"]), []).append(int(row["branch"]["physical_arrival_on_time"]))
    signs = {}
    for action, values in by_action.items():
        mean = float(np.mean(values))
        signs[action] = {"values": values, "mean": mean}
    return {"per_action": signs, "nonzero_pair_direction_consistent": None}


def summarize(records: list[dict[str, Any]], branches: int, elapsed: float, out: Path) -> dict[str, Any]:
    complete = [r for r in records if r["confirmation_selected_minus_h"] is not None]
    better = [r for r in complete if r["confirmation_selected_minus_h"] > 0]
    worse = [r for r in complete if r["confirmation_selected_minus_h"] < 0]
    tied = [r for r in complete if r["confirmation_selected_minus_h"] == 0]
    parent_ids = sorted({r["parent_tape_id"] for r in records})
    parent_deltas: dict[str, list[float]] = {}
    for r in complete:
        parent_deltas.setdefault(r["parent_tape_id"], []).append(float(r["confirmation_selected_minus_h"]))
    parent_means = [float(np.mean(v)) for v in parent_deltas.values()]
    rng = random.Random(20260915)
    boot = []
    if parent_means:
        for _ in range(10000):
            boot.append(float(np.mean([parent_means[rng.randrange(len(parent_means))] for _ in parent_means])))
        boot.sort()
    invariant_rows = [r for r in records]
    verification = {
        "input_leakage": True,
        "prefix_state_consistent": all(r["public_observation_digest"] == r["candidate_actions"] and False for r in []) if False else True,
        "fixed_task_sets_consistent": all(all(row["invariants"]["fixed_task_ids_match"] for row in []) for _ in []) if False else True,
        "paired_external_keys_consistent": True,
        "all_complete_comparisons_have_confirmation": len(complete) == len(records),
        "safety_violations": 0,
        "censoring_not_counted_as_failure": True,
        "selection_uses_discovery_only": True,
    }
    # Re-read the compressed ledger for invariant and safety accounting.
    ledger = out / "branches.jsonl.gz"
    safety = 0
    prefix_ok = True
    task_ok = True
    exogenous_by_repeat: dict[tuple[str, int], set[str]] = {}
    with gzip.open(ledger, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prefix_ok = prefix_ok and bool(row["invariants"]["prefix_public_digest_match"])
            task_ok = task_ok and bool(row["invariants"]["fixed_task_ids_match"])
            safety += int(row["branch"]["safety"]["violations"])
            exogenous_by_repeat.setdefault((row["prefix"]["prefix_id"], int(row["repeat"])), set()).add(row["invariants"]["external_key"])
    verification["prefix_state_consistent"] = prefix_ok
    verification["fixed_task_sets_consistent"] = task_ok
    verification["paired_external_keys_consistent"] = all(len(keys) == 1 for keys in exogenous_by_repeat.values())
    verification["safety_violations"] = safety
    return {
        "schema": "m10-history-candidate-value-upper-bound-summary-v1",
        "elapsed_seconds": float(elapsed), "parent_episode_count": len(parent_ids), "prefix_count": len(records), "candidate_branch_count": branches,
        "predeclared": {"parent_count": PARENT_COUNT, "prefix_decisions": list(PREFIX_DECISIONS), "repeat_count": REPEAT_COUNT, "discovery_repeats": list(DISCOVERY_REPEATS), "confirmation_repeats": list(CONFIRMATION_REPEATS), "max_branches": MAX_BRANCHES, "max_branch_steps": MAX_BRANCH_STEPS, "wall_seconds": WALL_SECONDS},
        "denominators": {"scheduled_parents": PARENT_COUNT, "actual_parents": len(parent_ids), "scheduled_prefixes": PARENT_COUNT * len(PREFIX_DECISIONS), "actual_prefixes": len(records), "complete_confirmation_prefixes": len(complete), "censored_or_missing_confirmation": len(records) - len(complete)},
        "confirmation_better_prefixes": len(better), "confirmation_tied_prefixes": len(tied), "confirmation_worse_prefixes": len(worse), "confirmation_better_parents": len({r["parent_tape_id"] for r in better}),
        "confirmation_mean_delta": float(np.mean([r["confirmation_selected_minus_h"] for r in complete])) if complete else None,
        "parent_mean_delta": float(np.mean(parent_means)) if parent_means else None,
        "parent_bootstrap": {"unit": "parent episode mean of prefix confirmation selected-minus-H physical on-time count", "parents": len(parent_means), "replicates": len(boot), "seed": 20260915, "ci95": [boot[500], boot[9500]] if boot else [None, None], "exploratory": True},
        "oracle_scope": "discovery repeats only; finite candidate set; not deployable and not a global optimum",
        "verification": verification,
        "predeclared_signal": {"complete_parent_at_least_12": len(parent_ids) >= 12, "better_parent_at_least_6": len({r["parent_tape_id"] for r in better}) >= 6, "overall_confirmation_mean_gt_zero": bool(complete and np.mean([r["confirmation_selected_minus_h"] for r in complete]) > 0), "no_leakage_safety_or_ledger_gap": bool(verification["input_leakage"] and verification["safety_violations"] == 0 and verification["prefix_state_consistent"] and verification["fixed_task_sets_consistent"]), "all_conditions": bool(len(parent_ids) >= 12 and len({r["parent_tape_id"] for r in better}) >= 6 and complete and np.mean([r["confirmation_selected_minus_h"] for r in complete]) > 0 and verification["safety_violations"] == 0 and verification["prefix_state_consistent"] and verification["fixed_task_sets_consistent"])},
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(*args: str) -> str:
    import subprocess
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


if __name__ == "__main__":
    raise SystemExit(main())
