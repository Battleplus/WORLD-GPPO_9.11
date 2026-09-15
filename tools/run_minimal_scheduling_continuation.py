"""Run frozen model first action plus greedy continuation on fresh instances."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.minimal_consequence_model import CandidateRegretNet, graph_sample
from gppo_world.minimal_scheduling import (
    Assignment, Instance, assign, initial_state, legal_actions, manhattan,
    solve_exact, wait_to_next_event,
)

EXPECTED_CHECKPOINTS = {
    1101: "a3306884f989d7307a25e2007c7fa2c46113ca3d9f08f2d34e61bc15234e664b",
    2203: "c572e2e907ccc2a84f343877ee4c4e0200e9972a344e8eb3e52ca89bbc71f903",
    3307: "7fbe399ed480c8192a7ccac844c34d0de2ec27de0e136bce62f74602b4c0562a",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_instance(payload: dict[str, object]) -> Instance:
    row = payload["instance"]
    from gppo_world.minimal_scheduling import Subtask, UAV
    return Instance(
        str(row["instance_id"]), str(row["group"]),
        tuple(UAV(int(x["uav_id"]), tuple(map(int, x["position"]))) for x in row["uavs"]),
        tuple(Subtask(str(x["task_id"]), str(x["chain_id"]), int(x["index"]),
            tuple(map(int, x["position"])), int(x["process_time"]),
            tuple(map(int, x["eligible_uavs"])), tuple(x["predecessors"])) for x in row["subtasks"]),
    )


def action_key(instance: Instance, action: tuple[str, int], positions: tuple[tuple[int, int], ...] | None = None) -> tuple[float, str, int]:
    task = instance.by_id[action[0]]
    origin = positions[action[1]] if positions is not None else instance.uavs[action[1]].position
    return (manhattan(origin, task.position) + task.process_time, action[0], action[1])


def greedy_action(instance: Instance, actions: tuple[tuple[str, int], ...], positions: tuple[tuple[int, int], ...] | None = None) -> tuple[str, int]:
    if not actions:
        raise ValueError("greedy action requested with empty legal mask")
    return min(actions, key=lambda action: action_key(instance, action, positions))


def model_input(instance: Instance) -> dict[str, object]:
    actions = legal_actions(instance, initial_state(instance))
    index = {task.task_id: i for i, task in enumerate(instance.subtasks)}
    return {
        "instance": {
            "instance_id": instance.instance_id,
            "group": instance.group,
            "uavs": [{"uav_id": u.uav_id, "position": list(u.position)} for u in instance.uavs],
            "subtasks": [{"task_id": t.task_id, "chain_id": t.chain_id, "index": t.index,
                "position": list(t.position), "process_time": t.process_time,
                "eligible_uavs": list(t.eligible_uavs), "predecessors": list(t.predecessors)} for t in instance.subtasks],
        },
        "candidates": [{"task_index": index[task_id], "task_id": task_id, "uav_id": uav_id,
            "normalized_regret": 0.0} for task_id, uav_id in actions],
    }


def choose_model_action(model: torch.nn.Module, instance: Instance, device: torch.device) -> tuple[tuple[str, int], dict[str, object], float]:
    actions = legal_actions(instance, initial_state(instance))
    if not actions:
        raise RuntimeError(f"no legal initial action for {instance.instance_id}")
    sample = graph_sample(model_input(instance), device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        scores = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if scores.ndim != 1 or len(scores) != len(actions) or not torch.isfinite(scores).all():
        raise FloatingPointError(f"invalid candidate scores for {instance.instance_id}")
    values = [float(x) for x in scores.detach().cpu().tolist()]
    minimum = min(values)
    tied = [i for i, value in enumerate(values) if abs(value - minimum) <= 1e-12]
    greedy = greedy_action(instance, actions)
    chosen_index = next((i for i in tied if actions[i] == greedy), None)
    if chosen_index is None:
        chosen_index = min(tied, key=lambda i: action_key(instance, (actions[i][0], actions[i][1])))
    evidence = {
        "legal_actions": [[task, uav] for task, uav in actions],
        "candidate_scores": [{"task_id": task, "uav_id": uav, "predicted_normalized_regret": score}
            for (task, uav), score in zip(actions, values)],
        "score_tie_tolerance": 1e-12,
        "tie_count": len(tied),
        "tie_resolution": "frozen greedy earliest-finish/task-ID/UAV-ID",
        "selected_action": [actions[chosen_index][0], actions[chosen_index][1]],
        "model_forward_calls": 1,
    }
    return actions[chosen_index], evidence, elapsed_ms


def assignment_record(assignment: Assignment, controller: str) -> dict[str, object]:
    return {
        "task_id": assignment.task_id, "uav_id": assignment.uav_id,
        "start_time": assignment.start_time, "finish_time": assignment.finish_time,
        "flight_time": assignment.flight_time, "process_time": assignment.process_time,
        "controller": controller,
    }


def execute_schedule(instance: Instance, forced_first: tuple[str, int] | None = None) -> dict[str, object]:
    """Execute a policy schedule without referring to the exact solver."""
    started = time.perf_counter()
    state = initial_state(instance)
    decisions: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    waits: list[dict[str, object]] = []
    first_pending = forced_first is not None
    while len(state.completed) < len(instance.subtasks):
        actions = legal_actions(instance, state)
        if actions:
            if first_pending:
                action = forced_first
                if action not in actions:
                    raise ValueError(f"forced first action is illegal: {action}")
                source = "frozen_model_first_action"
                first_pending = False
            else:
                action = greedy_action(instance, actions, state.positions)
                source = "frozen_greedy_continuation"
            before = state
            state, assignment = assign(instance, state, action)
            decisions.append({
                "time": before.time,
                "legal_actions": [[task, uav] for task, uav in actions],
                "selected_action": [action[0], action[1]],
                "controller": source,
            })
            assignments.append(assignment_record(assignment, source))
            continue
        if not any(r is not None for r in state.running):
            raise RuntimeError(f"deadlock before all tasks completed: {instance.instance_id}")
        from_time = state.time
        state, events = wait_to_next_event(instance, state)
        waits.append({"from_time": from_time, "to_time": state.time, "events": events})
        if state.time <= from_time:
            raise RuntimeError("wait did not advance simulation time")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    occupancy = [{"uav_id": x["uav_id"], "task_id": x["task_id"], "start_time": x["start_time"], "finish_time": x["finish_time"]} for x in assignments]
    return {
        "status": "complete", "makespan": state.time, "decisions": decisions,
        "assignments": assignments, "waits": waits, "uav_occupancy": occupancy,
        "simulation_time_unit": "frozen integer benchmark time", "compute_wall_ms": elapsed_ms,
    }


def replay_validate(instance: Instance, schedule: dict[str, object]) -> dict[str, object]:
    assignments = schedule["assignments"]
    by_task = {x["task_id"]: x for x in assignments}
    if len(by_task) != len(instance.subtasks) or set(by_task) != set(instance.by_id):
        raise AssertionError("ledger does not contain exactly one assignment per task")
    for task in instance.subtasks:
        rec = by_task[task.task_id]
        if rec["uav_id"] not in task.eligible_uavs:
            raise AssertionError(f"ineligible UAV in ledger for {task.task_id}")
        if any(by_task[p]["finish_time"] > rec["start_time"] for p in task.predecessors):
            raise AssertionError(f"precedence violated for {task.task_id}")
        if rec["finish_time"] - rec["start_time"] != rec["flight_time"] + task.process_time:
            raise AssertionError(f"duration mismatch for {task.task_id}")
        if rec["process_time"] != task.process_time:
            raise AssertionError(f"processing time mismatch for {task.task_id}")
    for uav in range(2):
        lane = sorted((x for x in assignments if x["uav_id"] == uav), key=lambda x: (x["start_time"], x["finish_time"]))
        previous_finish = 0
        previous_position = instance.uavs[uav].position
        for rec in lane:
            if rec["start_time"] < previous_finish:
                raise AssertionError(f"overlapping UAV occupation on {uav}")
            task = instance.by_id[rec["task_id"]]
            actual_flight = manhattan(previous_position, task.position)
            if actual_flight != rec["flight_time"]:
                raise AssertionError(f"flight path mismatch for {task.task_id}")
            previous_finish = rec["finish_time"]
            previous_position = task.position
    # Independently replay legal masks, deterministic greedy continuations,
    # event waits, and the exact assignment sequence from the initial state.
    replay_state = initial_state(instance)
    decision_index = 0
    wait_index = 0
    replayed_assignments: list[tuple[str, int, int, int]] = []
    while len(replay_state.completed) < len(instance.subtasks):
        actions = legal_actions(instance, replay_state)
        if actions:
            if decision_index >= len(schedule["decisions"]):
                raise AssertionError("decision ledger ended before the environment was complete")
            decision = schedule["decisions"][decision_index]
            if decision["time"] != replay_state.time:
                raise AssertionError("decision timestamp does not match replay state")
            expected_legal = [[task, uav] for task, uav in actions]
            if decision["legal_actions"] != expected_legal:
                raise AssertionError("recorded legal action set differs from replay mask")
            action = tuple(decision["selected_action"])
            if action not in actions:
                raise AssertionError("ledger selected an illegal action")
            controller = decision["controller"]
            if controller == "frozen_greedy_continuation":
                if action != greedy_action(instance, actions, replay_state.positions):
                    raise AssertionError("continuation action differs from frozen greedy rule")
            elif controller != "frozen_model_first_action" or decision_index != 0:
                raise AssertionError("model action may occur only as the first decision")
            replay_state, replay_assignment = assign(instance, replay_state, action)
            replayed_assignments.append((replay_assignment.task_id, replay_assignment.uav_id, replay_assignment.start_time, replay_assignment.finish_time))
            decision_index += 1
            continue
        if not any(r is not None for r in replay_state.running):
            raise AssertionError("replay deadlocked with no running task")
        if wait_index >= len(schedule["waits"]):
            raise AssertionError("wait ledger ended before task completion")
        wait = schedule["waits"][wait_index]
        if wait["from_time"] != replay_state.time:
            raise AssertionError("wait start time differs from replay state")
        replay_state, events = wait_to_next_event(instance, replay_state)
        if wait["to_time"] != replay_state.time or wait["events"] != events:
            raise AssertionError("wait event ledger differs from deterministic replay")
        wait_index += 1
    if decision_index != len(schedule["decisions"]) or wait_index != len(schedule["waits"]):
        raise AssertionError("ledger contains unused decision or wait events")
    recorded = [(x["task_id"], x["uav_id"], x["start_time"], x["finish_time"]) for x in assignments]
    if recorded != replayed_assignments:
        raise AssertionError("assignment rows do not match decision replay")
    expected_occupancy = [{"uav_id": x["uav_id"], "task_id": x["task_id"], "start_time": x["start_time"], "finish_time": x["finish_time"]} for x in assignments]
    if schedule["uav_occupancy"] != expected_occupancy:
        raise AssertionError("UAV occupancy ledger differs from assignments")
    replayed = max(x["finish_time"] for x in assignments)
    if replayed != schedule["makespan"]:
        raise AssertionError("ledger makespan does not replay")
    return {"status": "passed", "replayed_makespan": replayed, "task_count": len(assignments), "uav_count": 2}


def load_models(paths: list[Path], device: torch.device) -> tuple[dict[int, CandidateRegretNet], dict[int, str], dict[int, Path]]:
    if len(paths) != 3:
        raise ValueError("exactly three frozen checkpoints are required")
    models: dict[int, CandidateRegretNet] = {}
    hashes: dict[int, str] = {}
    model_paths: dict[int, Path] = {}
    for path in paths:
        checkpoint_hash = sha256(path)
        payload = torch.load(path, map_location=device, weights_only=False)
        seed = int(payload["seed"])
        if seed not in EXPECTED_CHECKPOINTS or checkpoint_hash != EXPECTED_CHECKPOINTS[seed]:
            raise ValueError(f"checkpoint identity/hash mismatch for {path}: seed={seed}, sha256={checkpoint_hash}")
        if seed in models:
            raise ValueError(f"duplicate model seed {seed}")
        model = CandidateRegretNet().to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        models[seed] = model
        hashes[seed] = checkpoint_hash
        model_paths[seed] = path
    if set(models) != set(EXPECTED_CHECKPOINTS):
        raise ValueError(f"checkpoint seed set mismatch: {sorted(models)}")
    return models, hashes, model_paths


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def evaluate_instance(
    index: int,
    row: dict[str, object],
    models: dict[int, CandidateRegretNet],
    device: torch.device,
    per_instance_seconds: float,
    wall_deadline: float,
) -> dict[str, object]:
    instance = load_instance(row)
    method_results: dict[str, object] = {}
    g_schedule = execute_schedule(instance)
    g_schedule["model_forward_calls"] = 0
    g_schedule["replay_validation"] = replay_validate(instance, g_schedule)
    method_results["G"] = g_schedule
    model_evidence: dict[str, object] = {}
    for seed in (1101, 2203, 3307):
        model = models[seed]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        method_start = time.perf_counter()
        first_action, evidence, inference_ms = choose_model_action(model, instance, device)
        schedule = execute_schedule(instance, forced_first=first_action)
        baseline_legal = method_results["G"]["decisions"][0]["legal_actions"]
        if evidence["legal_actions"] != baseline_legal:
            raise AssertionError("model and rule did not receive the same initial legal action identity set")
        if schedule["decisions"][0]["selected_action"] != evidence["selected_action"]:
            raise AssertionError("selected model action differs between scorer and executable schedule")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        complete_compute_ms = (time.perf_counter() - method_start) * 1000.0
        schedule["initial_model_inference_ms"] = inference_ms
        schedule["model_forward_calls"] = 1
        schedule["complete_method_compute_wall_ms"] = complete_compute_ms
        schedule["replay_validation"] = replay_validate(instance, schedule)
        schedule["continuation_controller"] = "frozen earliest-finish greedy after first action"
        method_results[f"M{seed}"] = schedule
        model_evidence[str(seed)] = evidence
    # Exact search is strictly post hoc: it cannot affect any executable controller.
    remaining = max(0.001, min(per_instance_seconds, wall_deadline - time.monotonic()))
    exact = solve_exact(instance, max_seconds=remaining)
    exact_status = exact.status
    exact_makespan = exact.makespan if exact_status == "optimal" else None
    for method in method_results.values():
        method["optimal_regret"] = int(method["makespan"]) - int(exact_makespan) if exact_makespan is not None else None
        method["optimal_reference_status"] = exact_status
    status = "complete" if all(x["status"] == "complete" for x in method_results.values()) else "execution_incomplete"
    return {
        "index": index, "instance_id": instance.instance_id, "group": instance.group,
        "content_sha256": row["content_sha256"], "status": status,
        "model_selection_evidence": model_evidence, "methods": method_results,
        "exact_reference": {"status": exact_status, "makespan": exact_makespan,
            "expanded_states": exact.expanded_states, "elapsed_seconds": exact.elapsed_seconds,
            "proof": exact.proof, "controller_used": False},
    }


def summarize(records: list[dict[str, object]], seed: int | None = None) -> dict[str, object]:
    group_names = ("homogeneous", "heterogeneous")
    result: dict[str, object] = {}
    for group in ("overall",) + group_names:
        subset = records if group == "overall" else [r for r in records if r["group"] == group]
        if seed is not None:
            method_rows = [r["methods"][f"M{seed}"] for r in subset]
        else:
            method_rows = [r["methods"] for r in subset]
        if not subset:
            result[group] = {"instances": 0}
            continue
        g_times = [int(r["methods"]["G"]["makespan"]) for r in subset]
        if seed is None:
            m_times = {s: [int(r["methods"][f"M{s}"]["makespan"]) for r in subset] for s in EXPECTED_CHECKPOINTS}
            result[group] = {
                "instances": len(subset), "G_mean_makespan": mean(g_times),
                "M_mean_makespan_by_seed": {str(s): mean(v) for s, v in m_times.items()},
            }
        else:
            m_times = [int(r["methods"][f"M{seed}"]["makespan"]) for r in subset]
            diffs = [g - m for g, m in zip(g_times, m_times)]
            full_compute_ms = [float(r["methods"][f"M{seed}"]["complete_method_compute_wall_ms"]) for r in subset]
            initial_inference_ms = [float(r["methods"][f"M{seed}"]["initial_model_inference_ms"]) for r in subset]
            result[group] = {
                "instances": len(subset), "G_mean_makespan": mean(g_times),
                "M_mean_makespan": mean(m_times), "mean_time_reduction_G_minus_M": mean(diffs),
                "relative_makespan_reduction": mean(diffs) / mean(g_times) if mean(g_times) else None,
                "wins_ties_losses": [sum(x > 0 for x in diffs), sum(x == 0 for x in diffs), sum(x < 0 for x in diffs)],
                "G_optimal_gap_mean": mean([r["methods"]["G"]["optimal_regret"] for r in subset if r["methods"]["G"]["optimal_regret"] is not None]) if all(r["methods"]["G"]["optimal_regret"] is not None for r in subset) else None,
                "M_optimal_gap_mean": mean([r["methods"][f"M{seed}"]["optimal_regret"] for r in subset if r["methods"][f"M{seed}"]["optimal_regret"] is not None]) if all(r["methods"][f"M{seed}"]["optimal_regret"] is not None for r in subset) else None,
                "all_complete": all(r["methods"][f"M{seed}"]["status"] == "complete" for r in subset),
                "execution_violations": sum(r["methods"][f"M{seed}"]["replay_validation"]["status"] != "passed" for r in subset),
                "initial_model_inference_wall_ms": {"mean": mean(initial_inference_ms), "p95": float(np.percentile(initial_inference_ms, 95)), "p99": float(np.percentile(initial_inference_ms, 99)), "samples": len(initial_inference_ms), "warmup": "none; first call included"},
                "complete_method_compute_wall_ms": {"mean": mean(full_compute_ms), "p95": float(np.percentile(full_compute_ms, 95)), "p99": float(np.percentile(full_compute_ms, 99)), "samples": len(full_compute_ms), "scope": "graph construction + one model forward + full greedy continuation; excludes exact solver and file output"},
            }
    return result


def stratified_bootstrap(records: list[dict[str, object]], reps: int = 10000, seed: int = 20260916) -> dict[str, object]:
    groups = {name: [r for r in records if r["group"] == name] for name in ("homogeneous", "heterogeneous")}
    per_instance: dict[str, float] = {}
    for row in records:
        differences = [row["methods"]["G"]["makespan"] - row["methods"][f"M{s}"]["makespan"] for s in EXPECTED_CHECKPOINTS]
        per_instance[row["instance_id"]] = mean(differences)
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        picked = []
        for rows in groups.values():
            indexes = rng.integers(0, len(rows), size=len(rows))
            picked.extend(per_instance[rows[j]["instance_id"]] for j in indexes)
        samples[i] = float(np.mean(picked))
    return {
        "unit": "per-instance mean over three frozen model seeds of G makespan minus M makespan, then instance resampling within homogeneous/heterogeneous strata",
        "strata_instances": {k: len(v) for k, v in groups.items()},
        "replicates": reps, "seed": seed,
        "mean_time_reduction": mean(per_instance.values()),
        "ci95_time_units": [float(np.quantile(samples, 0.025, method="linear")), float(np.quantile(samples, 0.975, method="linear"))],
        "does_not_cover_training_seed_population_uncertainty": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--models", nargs=3, type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--per-instance-seconds", type=float, default=30.0)
    ap.add_argument("--wall-seconds", type=float, default=1800.0)
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent device fallback")
    if args.per_instance_seconds > 30.0 or args.wall_seconds > 1800.0:
        raise SystemExit("registered solver/wall budget cannot be increased")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {args.out}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("schema") != "m10-minimal-scheduling-continuation-protocol-v1":
        raise SystemExit("unexpected evaluation protocol schema")
    device = torch.device(args.device)
    models, checkpoint_hashes, model_paths = load_models(args.models, device)
    state_hashes_before = {seed: state_dict_sha256(model) for seed, model in models.items()}
    inputs = json.loads(args.instances.read_text(encoding="utf-8"))
    instance_rows = inputs["records"]
    if len(instance_rows) != 128 or sum(row["group"] == "homogeneous" for row in instance_rows) != 64 or sum(row["group"] == "heterogeneous" for row in instance_rows) != 64:
        raise SystemExit("continuation validation set does not match frozen 64/64 contract")
    started = time.monotonic()
    wall_deadline = started + args.wall_seconds
    args.out.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, object]] = []
    run_identity = {
        "schema": "m10-minimal-scheduling-continuation-run-identity-v1",
        "run_id": args.out.name,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "protocol": str(args.protocol.resolve()), "protocol_sha256": sha256(args.protocol),
        "instances": str(args.instances.resolve()), "instances_sha256": sha256(args.instances),
        "checkpoints": {str(seed): {"path": str(model_paths[seed].resolve()), "sha256": checkpoint_hashes[seed]} for seed in (1101, 2203, 3307)},
        "device": args.device, "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_parameter_updates": 0,
    }
    dump(args.out / "run-identity.json", run_identity)
    dump(args.out / "ledger.json", {"schema": "m10-minimal-scheduling-continuation-ledger-v1", "records": results})
    for index, row in enumerate(instance_rows):
        if time.monotonic() >= wall_deadline:
            results.extend({"index": j, "instance_id": r["instance"]["instance_id"], "group": r["group"], "status": "not_started_wall_budget"} for j, r in enumerate(instance_rows[index:], start=index))
            dump(args.out / "ledger.json", {"schema": "m10-minimal-scheduling-continuation-ledger-v1", "records": results})
            break
        try:
            result_row = evaluate_instance(index, row, models, device, args.per_instance_seconds, wall_deadline)
        except Exception as exc:
            results.append({"index": index, "instance_id": row["instance"]["instance_id"], "group": row["group"], "status": "failed", "failure": f"{type(exc).__name__}: {exc}"})
            dump(args.out / "ledger.json", {"schema": "m10-minimal-scheduling-continuation-ledger-v1", "records": results})
            break
        results.append(result_row)
        dump(args.out / "ledger.json", {"schema": "m10-minimal-scheduling-continuation-ledger-v1", "records": results})
        if result_row["status"] != "complete":
            break
    # Hash the frozen parameter tensors again, and fail closed if inference changed any weight.
    state_hashes_after = {seed: state_dict_sha256(model) for seed, model in models.items()}
    if state_hashes_before != state_hashes_after:
        raise RuntimeError("frozen model parameters changed during evaluation")
    complete = [r for r in results if r.get("status") == "complete"]
    identities = {r["instance_id"] for r in complete}
    if len(identities) != len(complete):
        raise RuntimeError("duplicate instance identity in result rows")
    summary = {
        "schema": "m10-minimal-scheduling-continuation-results-v1",
        "scope": "learned first action followed by greedy continuation; no repeated model decisions",
        "instance_count": len(complete), "records_including_unstarted": len(results),
        "all_instances_complete": len(complete) == 128,
        "G_and_M_summary": {"G": summarize(complete), **{f"M{s}": summarize(complete, s) for s in (1101, 2203, 3307)}},
        "bootstrap": stratified_bootstrap(complete) if len(complete) == 128 else None,
        "execution_integrity": {
            "duplicate_result_id_count": len(complete) - len(identities),
            "illegal_or_replay_validation_violations": sum(1 for row in complete for method in row["methods"].values() if method["replay_validation"]["status"] != "passed"),
            "model_forward_calls_per_M": {str(seed): sorted(set(row["methods"][f"M{seed}"]["model_forward_calls"] for row in complete)) for seed in (1101, 2203, 3307)},
            "model_parameter_state_sha256_before": {str(k): v for k, v in state_hashes_before.items()},
            "model_parameter_state_sha256_after": {str(k): v for k, v in state_hashes_after.items()},
            "checkpoint_file_sha256_before": {str(k): v for k, v in checkpoint_hashes.items()},
            "checkpoint_file_sha256_after": {str(seed): sha256(model_paths[seed]) for seed in (1101, 2203, 3307)},
            "exact_solver_controller_calls": 0,
        },
        "budgets": {"per_instance_solver_seconds": args.per_instance_seconds, "wall_seconds": args.wall_seconds, "elapsed_wall_seconds": time.monotonic() - started},
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda if torch.cuda.is_available() else None, "device": args.device, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
    }
    if len(complete) == 128:
        mean_g = summary["G_and_M_summary"]["G"]["overall"]["G_mean_makespan"]
        mean_ms = {s: summary["G_and_M_summary"][f"M{s}"]["overall"]["M_mean_makespan"] for s in (1101, 2203, 3307)}
        mean_m = mean(mean_ms.values())
        ci_low = summary["bootstrap"]["ci95_time_units"][0]
        summary["preregistered_continuation_gate"] = {
            "mean_makespan_reduction_at_least_3_percent": bool(mean_g and (mean_g - mean_m) / mean_g >= 0.03),
            "at_least_two_of_three_seeds_improve": sum(mean_ms[s] < mean_g for s in mean_ms) >= 2,
            "paired_ci_lower_bound_above_zero": ci_low > 0,
            "all_instances_complete": len(complete) == 128,
            "no_execution_violations": summary["execution_integrity"]["illegal_or_replay_validation_violations"] == 0,
            "pass": bool(mean_g and (mean_g - mean_m) / mean_g >= 0.03 and sum(mean_ms[s] < mean_g for s in mean_ms) >= 2 and ci_low > 0 and len(complete) == 128 and summary["execution_integrity"]["illegal_or_replay_validation_violations"] == 0),
        }
    dump(args.out / "results.json", summary)
    manifest_files = []
    for path in sorted(args.out.iterdir()):
        if path.is_file(): manifest_files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (args.out / "manifest.json").write_text(json.dumps({"schema": "m10-minimal-scheduling-continuation-result-manifest-v1", "files": manifest_files}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete" if summary["all_instances_complete"] else "incomplete", "instances": len(complete), "gate": summary.get("preregistered_continuation_gate")}, ensure_ascii=False))
    return 0 if summary["all_instances_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
