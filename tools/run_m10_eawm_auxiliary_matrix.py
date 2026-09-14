"""Bounded H/E arrival-protocol training and evaluation matrix.

H is the original Graph-5 GPPO-History policy.  E is the same policy with the
training-only public-transition auxiliary loss.  This entry freezes tapes
before training, runs pilot first, then the registered formal seeds.  It never
uses the old candidate-consequence model or prior.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.eawm_auxiliary import (  # noqa: E402
    AUXILIARY_METHOD,
    AUXILIARY_SCHEMA,
    EventAwareHistoryPolicy,
    EventAwareTransition,
    build_public_event_targets,
    evaluate_event_aware_sequence,
    save_event_aware_checkpoint,
    update_event_aware_policy,
)
from gppo_world.m10_environment import (  # noqa: E402
    M10Config,
    M10Environment,
    M10Scenario,
    scenario_to_dict,
    weak_communication_tape,
)
from gppo_world.m10_training import (  # noqa: E402
    M10ActorCritic,
    PPOConfig,
    Transition,
    _evaluate_sequence,
    _gae,
    masked_distribution,
    seed_everything,
)
from tools.run_m10_baseline_comparison import traditional_action  # noqa: E402


PILOT_STEPS = 512
FORMAL_STEPS = 8192
MAX_UPDATES = 128
ROLLOUT_STEPS = 256
UPDATE_EPOCHS = 4
THREADS = 4
TOTAL_WALL_SECONDS = 4 * 60 * 60
PILOT_GROUP_WALL_SECONDS = 30 * 60
TAPE_BASE_SEED = 991001
TRAIN_TAPE_COUNT = 32
VALIDATION_TAPE_COUNT = 16
TEST_TAPE_COUNT = 64
SEEDS = (1101, 2203, 3307)


def arrival_config() -> M10Config:
    return M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")


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


def make_base_policy(config: M10Config) -> M10ActorCritic:
    return M10ActorCritic(
        uav_count=config.uav_count, task_capacity=config.task_capacity,
        action_count=config.action_count, encoder="graph", type_count=5,
        history=True, context_dim=0, region_count=config.region_count,
        target_count=config.target_count, event_capacity=config.event_capacity,
        relation_width=config.relation_width,
    )


def make_policy(config: M10Config, variant: str, seed: int, device: torch.device):
    seed_everything(seed)
    base = make_base_policy(config)
    if variant == "H":
        return base.to(device)
    if variant != "E":
        raise ValueError(f"unknown variant: {variant}")
    # The auxiliary heads use a forked RNG.  Creating them cannot advance the
    # global strategy-sampling RNG or alter the shared policy initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 7000003)
        policy = EventAwareHistoryPolicy(base)
    return policy.to(device)


def shared_initialization_audit(config: M10Config, seed: int) -> dict[str, Any]:
    history = make_policy(config, "H", seed, torch.device("cpu"))
    event = make_policy(config, "E", seed, torch.device("cpu"))
    shared = []
    for name, value in history.state_dict().items():
        other_name = f"base_policy.{name}"
        if other_name not in event.state_dict():
            raise RuntimeError(f"missing shared parameter in E: {other_name}")
        if not torch.equal(value, event.state_dict()[other_name]):
            raise RuntimeError(f"shared initialization mismatch: {name}")
        shared.append((name, hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()))
    return {"seed": seed, "shared_parameters_equal": True, "shared_parameter_hashes": dict(shared), "auxiliary_seed": seed + 7000003}


def freeze_tapes(root: Path, config: M10Config) -> dict[str, list[M10Scenario]]:
    path = root / "frozen-tapes.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        from gppo_world.m10_environment import scenario_from_dict
        return {key: [scenario_from_dict(item) for item in value] for key, value in payload.items()}
    tapes = {
        "train": list(weak_communication_tape("train", count=TRAIN_TAPE_COUNT, base_seed=TAPE_BASE_SEED, level="composite")),
        "validation": list(weak_communication_tape("validation", count=VALIDATION_TAPE_COUNT, base_seed=TAPE_BASE_SEED, level="composite")),
        "final_test": list(weak_communication_tape("test", count=TEST_TAPE_COUNT, base_seed=TAPE_BASE_SEED, level="composite")),
    }
    ids = [scenario.tape_id for group in tapes.values() for scenario in group]
    if len(ids) != len(set(ids)):
        raise RuntimeError("frozen tape identity overlap")
    dump(path, {key: [scenario_to_dict(item) for item in value] for key, value in tapes.items()})
    dump(root / "tape-manifest.json", {
        "source_commit": subprocess_git_commit(),
        "base_seed": TAPE_BASE_SEED,
        "split_counts": {key: len(value) for key, value in tapes.items()},
        "task_count_per_scenario": config.task_capacity,
        "communication": "composite weak communication",
        "task_completion_mode": config.task_completion_mode,
        "deadline_basis": config.deadline_basis,
        "no_formal_test_read_before_training_freeze": True,
        "frozen_tapes_sha256": sha256_file(path),
    })
    return tapes


def subprocess_git_commit() -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


@dataclass
class RolloutItem:
    transition: Transition
    public_observation: dict[str, Any]
    next_public_observation: dict[str, Any]
    targets: Any
    tape_id: str


def collect_history_rollout(policy, *, ppo: PPOConfig, config: M10Config, seed: int,
                            device: torch.device, scenarios: list[M10Scenario], event_aware: bool) -> list[RolloutItem]:
    if not scenarios:
        raise ValueError("training requires a non-empty frozen train tape")
    env_index = 0
    env = M10Environment(config, scenarios[env_index])
    obs = env.reset()
    hidden = None
    episode_start = True
    items: list[RolloutItem] = []
    for index in range(ppo.rollout_steps):
        with torch.no_grad():
            public = torch.as_tensor(obs["flat"], dtype=torch.float32, device=device)[None, :]
            logits, value, hidden = policy(public, hidden)
            mask = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device)[None, :]
            distribution = masked_distribution(logits, mask)
            action_tensor = distribution.sample()
            action = int(action_tensor.item())
            log_prob = float(distribution.log_prob(action_tensor).item())
            value_float = float(value.item())
        next_obs, reward, done, info = env.step(action, submit_command=True)
        info = dict(info)
        info["tape_id"] = scenarios[env_index].tape_id
        terminated = bool(info.get("terminated", done))
        truncated = bool(info.get("truncated", False)) or (not done and index == ppo.rollout_steps - 1)
        with torch.no_grad():
            if terminated:
                next_value = 0.0
            else:
                next_public = torch.as_tensor(next_obs["flat"], dtype=torch.float32, device=device)[None, :]
                next_value = float(policy(next_public, hidden)[1].item())
        transition = Transition(
            obs=np.asarray(obs["flat"], dtype=np.float32).copy(), context=np.zeros(0, dtype=np.float32),
            mask=np.asarray(obs["mask"], dtype=bool).copy(), action=action,
            log_prob=log_prob, value=value_float, reward=float(reward),
            done=bool(terminated or truncated), episode_start=episode_start,
            info=info, terminated=terminated, truncated=truncated, next_value=next_value,
            actor_decision=True,
        )
        targets = build_public_event_targets(obs, next_obs)
        items.append(RolloutItem(transition, public_json(obs), next_obs, targets, scenarios[env_index].tape_id))
        if done:
            env_index = (env_index + 1) % len(scenarios)
            env = M10Environment(config, scenarios[env_index])
            obs = env.reset()
            hidden = None
            episode_start = True
        else:
            obs = next_obs
            episode_start = False
    return items


def finite_optimizer(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError("non-finite optimizer state")


def update_history_policy(policy: M10ActorCritic, items: list[RolloutItem], *, ppo: PPOConfig,
                          device: torch.device, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    transitions = [item.transition for item in items]
    advantages, returns = _gae(transitions, ppo.gamma, ppo.gae_lambda)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    advantages, returns = advantages.to(device), returns.to(device)
    old_log_probs = torch.as_tensor([item.transition.log_prob for item in items], dtype=torch.float32, device=device)
    last: dict[str, Any] = {}
    for epoch in range(ppo.update_epochs):
        new_log_probs, values, entropies = _evaluate_sequence(policy, transitions, device)
        ratio = (new_log_probs - old_log_probs).exp()
        clipped = torch.clamp(ratio, 1.0 - ppo.clip_epsilon, 1.0 + ppo.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = entropies.mean()
        loss = policy_loss + ppo.value_weight * value_loss - ppo.entropy_weight * entropy
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite History loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo.grad_clip))
        if not np.isfinite(grad_norm):
            raise FloatingPointError("non-finite History gradient")
        optimizer.step()
        if not all(torch.isfinite(value).all() for value in policy.state_dict().values()):
            raise FloatingPointError("non-finite History parameter")
        finite_optimizer(optimizer)
        last = {
            "loss": float(loss.detach().cpu()), "ppo_loss": float(loss.detach().cpu()),
            "event_loss": 0.0, "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()), "entropy": float(entropy.detach().cpu()),
            "approx_kl": float((old_log_probs - new_log_probs).mean().detach().cpu()),
            "clip_fraction": float(((ratio - 1.0).abs() > ppo.clip_epsilon).float().mean().detach().cpu()),
            "grad_norm": grad_norm, "optimizer_step": 1, "update_epoch": epoch + 1,
            "valid_event_steps": 0,
        }
    last["optimizer_steps"] = ppo.update_epochs
    return last


def save_history_checkpoint(path: Path, policy: M10ActorCritic, metadata: dict[str, Any], *, inference: bool = False, optimizer=None) -> None:
    recovery = None if inference else {
        "seed": metadata.get("seed"), "environment_steps": metadata.get("environment_steps", 0),
        "optimizer_updates": metadata.get("optimizer_updates", 0), "state": metadata.get("state", {}),
        "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    torch.save({
        "format": "gppo-history-arrival-training-v1" if not inference else "gppo-history-arrival-inference-v1",
        "metadata": dict(metadata), "state_dict": policy.state_dict(),
        "optimizer_state_dict": None if inference or optimizer is None else optimizer.state_dict(),
        "recovery_state": recovery,
    }, path)


def public_json(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "uavs": np.asarray(observation["uavs"], dtype=np.float32).tolist(),
        "tasks": np.asarray(observation["tasks"], dtype=np.float32).tolist(),
        "entity_ids": observation["entity_ids"], "mask": np.asarray(observation["mask"], dtype=bool).tolist(),
        "time": float(observation["time"]),
    }


def write_rollout_ledger(path: Path, items: list[RolloutItem], run_id: str, seed: int, batch_index: int) -> dict[str, int]:
    counts = {"position": 0, "energy": 0, "task_state": 0}
    with gzip.open(path, "at", encoding="utf-8") as stream:
        for step_index, item in enumerate(items):
            transition = item.transition
            for key, value in item.targets.valid_counts.items():
                counts[key] += int(value)
            payload = {
                "run_id": run_id, "seed": seed, "batch": batch_index, "step_in_batch": step_index,
                "tape_id": item.tape_id, "action": transition.action, "old_log_prob": transition.log_prob,
                "value": transition.value, "reward": transition.reward, "terminated": transition.terminated,
                "truncated": transition.truncated, "bootstrap_value": transition.next_value,
                "episode_start": transition.episode_start, "public_observation": item.public_observation,
                "targets": item.targets.as_dict(),
                "next_public_observation": public_json(item.next_public_observation),
                "info": {
                    "feedback": transition.info.get("feedback"), "command_id": transition.info.get("command_id"),
                    "command_submitted": transition.info.get("command_submitted"), "counts": transition.info.get("counts"),
                    "communication_delta": transition.info.get("communication_delta", []),
                    "new_events": transition.info.get("new_events", []), "deadline_basis": transition.info.get("deadline_basis"),
                    "task_completion_mode": transition.info.get("task_completion_mode"),
                },
            }
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    return counts


def evaluate_variant(policy, variant: str, scenarios: list[M10Scenario], config: M10Config,
                     device: torch.device, run_id: str, output: Path) -> dict[str, Any]:
    policy.eval()
    records = []
    latencies: list[float] = []
    with gzip.open(output / "evaluation-ledger.jsonl.gz", "wt", encoding="utf-8") as ledger:
        with torch.inference_mode():
            for scenario in scenarios:
                env = M10Environment(config, scenario)
                obs = env.reset()
                hidden = None
                done = False
                total_reward = 0.0
                steps = 0
                actions = []
                last_info: dict[str, Any] = {}
                while not done and steps < int(config.horizon / config.decision_interval) + 2:
                    start = time.perf_counter()
                    public = torch.as_tensor(obs["flat"], dtype=torch.float32, device=device)[None, :]
                    logits, value, hidden = policy(public, hidden)
                    mask = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device)[None, :]
                    distribution = masked_distribution(logits, mask)
                    action_tensor = torch.argmax(distribution.logits, dim=-1)
                    action = int(action_tensor.item())
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    latencies.append(elapsed_ms)
                    next_obs, reward, done, info = env.step(action, submit_command=True)
                    total_reward += float(reward)
                    actions.append(action)
                    last_info = dict(info)
                    ledger.write(json.dumps({
                        "run_id": run_id, "tape_id": scenario.tape_id, "step": steps, "action": action,
                        "public_observation": public_json(obs), "next_public_observation": public_json(next_obs),
                        "info": {"counts": info.get("counts"), "communication_delta": info.get("communication_delta", []), "feedback": info.get("feedback")},
                    }, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                    obs = next_obs
                    steps += 1
                completion = last_info.get("completion_records", {})
                physical = sum(bool(item.get("physical_arrival_before_deadline")) for item in completion.values())
                host = sum(bool(item.get("host_confirmation_before_deadline")) for item in completion.values())
                final_states = last_info.get("tasks", {})
                communication = [entry for entry in last_info.get("communication_log", [])]
                correct_rejections = sum(1 for entry in last_info.get("feedback_log", []) if entry.get("result") not in ("accepted", "awaiting_ack", "noop", "reuse_existing"))
                security_violations = sum(1 for entry in communication if entry.get("security_violation", False))
                records.append({
                    "tape_id": scenario.tape_id, "tasks": len(scenario.tasks), "steps": steps,
                    "return": total_reward, "physical_arrival_on_time": physical,
                    "host_confirmation_on_time": host, "deadline_failures": max(0, len(scenario.tasks) - physical),
                    "unresolved_tasks": sum(state not in ("completed", "expired") for state in final_states.values()),
                    "correct_rejections": correct_rejections, "security_violations": security_violations,
                    "actions": actions,
                })
    latency = np.asarray(latencies, dtype=np.float64)
    return {
        "variant": variant, "run_id": run_id, "episodes": records,
        "tasks": int(sum(row["tasks"] for row in records)),
        "physical_arrival_on_time": int(sum(row["physical_arrival_on_time"] for row in records)),
        "host_confirmation_on_time": int(sum(row["host_confirmation_on_time"] for row in records)),
        "deadline_failures": int(sum(row["deadline_failures"] for row in records)),
        "unresolved_tasks": int(sum(row["unresolved_tasks"] for row in records)),
        "correct_rejections": int(sum(row["correct_rejections"] for row in records)),
        "security_violations": int(sum(row["security_violations"] for row in records)),
        "latency": {"samples": int(latency.size), "mean_ms": float(latency.mean()) if latency.size else None, "p95_ms": float(np.percentile(latency, 95)) if latency.size else None, "p99_ms": float(np.percentile(latency, 99)) if latency.size else None, "warmup_steps": 0, "includes": "public tensor construction, policy forward, masking and action selection; excludes environment progression and disk I/O", "raw_ms": latency.tolist()},
        "device": str(device), "threads": torch.get_num_threads() if device.type == "cpu" else None,
    }


def train_one(variant: str, seed: int, steps: int, max_updates: int, *, root: Path,
              config: M10Config, train_tape: list[M10Scenario], validation_tape: list[M10Scenario],
              device: torch.device, wall_deadline: float, run_label: str,
              pilot: bool = False, resume: bool = False) -> dict[str, Any]:
    run_id = f"m10-eawm-aux-{variant.lower()}-{seed}-{run_label}"
    output = root / "runs" / run_id
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError(f"refusing non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    policy = make_policy(config, variant, seed, device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    ppo = PPOConfig(rollout_steps=ROLLOUT_STEPS, update_epochs=UPDATE_EPOCHS, minibatch_size=ROLLOUT_STEPS)
    state = {"environment_steps": 0, "optimizer_updates": 0, "batches": 0, "best_loss": None, "stop_reason": None}
    if resume:
        last_path = output / "last.pt"
        if not last_path.exists():
            raise RuntimeError(f"resume requested but checkpoint is missing: {last_path}")
        payload = torch.load(last_path, map_location=device, weights_only=False)
        expected_format = "gppo-history-arrival-training-v1" if variant == "H" else EventAwareHistoryPolicy.format_version
        if payload.get("format") != expected_format:
            raise RuntimeError(f"resume format mismatch for {run_id}: {payload.get('format')}")
        policy.load_state_dict(payload["state_dict"])
        if payload.get("optimizer_state_dict") is None:
            raise RuntimeError("resume checkpoint lacks optimizer state")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state.update(payload.get("metadata", {}).get("state", {}))
        recovery = payload.get("recovery_state") or {}
        rng_state = recovery.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"].cpu())
            if device.type == "cuda" and rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state_all([item.cpu() for item in rng_state["cuda"]])
    started = time.perf_counter()
    updates_path = output / "updates.jsonl"
    ledger_path = output / "training-rollout-ledger.jsonl.gz"
    validation_curve: list[dict[str, Any]] = []
    while state["environment_steps"] < steps and state["optimizer_updates"] < max_updates:
        if time.perf_counter() >= wall_deadline:
            state["stop_reason"] = "global_wall_clock_budget"
            break
        batch_steps = min(ROLLOUT_STEPS, steps - state["environment_steps"])
        batch_ppo = PPOConfig(**{**asdict(ppo), "rollout_steps": batch_steps})
        items = collect_history_rollout(policy, ppo=batch_ppo, config=config, seed=seed + state["environment_steps"], device=device, scenarios=train_tape, event_aware=(variant == "E"))
        ledger_counts = write_rollout_ledger(ledger_path, items, run_id, seed, state["batches"])
        if variant == "E":
            event_items = [EventAwareTransition(item.transition, item.next_public_observation, item.targets) for item in items]
            metrics = update_event_aware_policy(policy, event_items, ppo=batch_ppo, device=device, optimizer=optimizer, coefficient=0.1)
        else:
            metrics = update_history_policy(policy, items, ppo=batch_ppo, device=device, optimizer=optimizer)
        state["environment_steps"] += len(items)
        state["optimizer_updates"] += int(metrics["optimizer_steps"])
        state["batches"] += 1
        metrics.update({"run_id": run_id, "seed": seed, "environment_steps": state["environment_steps"], "optimizer_updates": state["optimizer_updates"], "elapsed_seconds": time.perf_counter() - started, "label_counts": ledger_counts})
        with updates_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        if state["environment_steps"] % 2048 == 0:
            validation_root = output / "validation" / str(state["environment_steps"])
            validation_root.mkdir(parents=True, exist_ok=False)
            validation = evaluate_variant(policy, variant, validation_tape, config, device, run_id, validation_root)
            validation_curve.append({
                "environment_steps": state["environment_steps"],
                "physical_arrival_on_time": validation["physical_arrival_on_time"],
                "host_confirmation_on_time": validation["host_confirmation_on_time"],
                "tasks": validation["tasks"],
                "return": sum(row["return"] for row in validation["episodes"]),
                "selection_used": False,
            })
            dump(output / "validation-curve.json", validation_curve)
            policy.train()
        if state["best_loss"] is None or metrics["loss"] < state["best_loss"]:
            state["best_loss"] = metrics["loss"]
            metadata = {"run_id": run_id, "variant": variant, "seed": seed, "environment_steps": state["environment_steps"], "optimizer_updates": state["optimizer_updates"], "selection_scope": "training loss only; final analysis uses last checkpoint"}
            if variant == "E":
                save_event_aware_checkpoint(output / "best-inference.pt", policy, metadata, inference=True)
            else:
                save_history_checkpoint(output / "best-inference.pt", policy, metadata, inference=True)
        metadata = {"run_id": run_id, "variant": variant, "seed": seed, "environment_steps": state["environment_steps"], "optimizer_updates": state["optimizer_updates"], "state": state, "config": asdict(config), "ppo": asdict(batch_ppo), "auxiliary_schema": AUXILIARY_SCHEMA if variant == "E" else None}
        if variant == "E":
            save_event_aware_checkpoint(output / "last.pt", policy, metadata, optimizer=optimizer)
        else:
            save_history_checkpoint(output / "last.pt", policy, metadata, optimizer=optimizer)
        if pilot and time.perf_counter() - started >= PILOT_GROUP_WALL_SECONDS:
            state["stop_reason"] = "pilot_group_wall_clock_budget"
            break
    if state["stop_reason"] is None:
        state["stop_reason"] = "environment_step_budget" if state["environment_steps"] >= steps else "optimizer_update_budget"
    elapsed = time.perf_counter() - started
    state.update({"status": "complete" if state["environment_steps"] >= steps else "incomplete", "elapsed_seconds": elapsed, "device": str(device), "threads": torch.get_num_threads() if device.type == "cpu" else None, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None})
    state["validation_curve_points"] = validation_curve
    dump(output / "run-status.json", state)
    dump(output / "run-identity.json", {"run_id": run_id, "variant": variant, "seed": seed, "source_commit": subprocess_git_commit(), "protocol": asdict(config), "ppo": asdict(ppo), "budgets": {"environment_steps": steps, "optimizer_updates": max_updates, "wall_seconds": PILOT_GROUP_WALL_SECONDS if pilot else None}, "actual": state, "world_model_used": False, "old_candidate_prior_used": False, "notification_changed": False, "formal_test_used_during_training": False})
    # Round-trip the actual last checkpoint without taking an optimizer step.
    payload = torch.load(output / "last.pt", map_location=device, weights_only=False)
    restored = make_policy(config, variant, seed, device)
    restored.load_state_dict(payload["state_dict"])
    restored.eval(); policy.eval()
    probe = torch.zeros((1, policy.base_policy.input_dim if variant == "E" else policy.input_dim), device=device)
    with torch.no_grad():
        first = policy(probe)[0]; second = restored(probe)[0]
    if not torch.allclose(first, second, atol=1e-6, rtol=1e-5):
        raise RuntimeError("checkpoint prediction round-trip mismatch")
    dump(output / "checkpoint-roundtrip.json", {"status": "passed", "cumulative_environment_steps": state["environment_steps"], "cumulative_optimizer_updates": state["optimizer_updates"], "checkpoint_format": payload.get("format"), "no_optimizer_step_during_check": True})
    return {"run_id": run_id, "variant": variant, "seed": seed, "output": str(output), "status": state["status"], "environment_steps": state["environment_steps"], "optimizer_updates": state["optimizer_updates"], "stop_reason": state["stop_reason"], "elapsed_seconds": elapsed}


def run_rules(scenarios: list[M10Scenario], config: M10Config, device: torch.device, root: Path) -> dict[str, Any]:
    output = root / "rules-final-test"
    output.mkdir(parents=True, exist_ok=False)
    records = []
    latencies = []
    for scenario in scenarios:
        env = M10Environment(config, scenario); obs = env.reset(); done = False; steps = 0; total_reward = 0.0; last_info = {}
        while not done and steps < int(config.horizon / config.decision_interval) + 2:
            start = time.perf_counter(); action = traditional_action(obs); latencies.append((time.perf_counter() - start) * 1000.0)
            obs, reward, done, last_info = env.step(action, submit_command=True); total_reward += float(reward); steps += 1
        completion = last_info.get("completion_records", {})
        physical = sum(bool(item.get("physical_arrival_before_deadline")) for item in completion.values())
        host = sum(bool(item.get("host_confirmation_before_deadline")) for item in completion.values())
        records.append({"tape_id": scenario.tape_id, "tasks": len(scenario.tasks), "return": total_reward, "physical_arrival_on_time": physical, "host_confirmation_on_time": host, "steps": steps})
    result = {"variant": "rule", "episodes": records, "tasks": int(sum(r["tasks"] for r in records)), "physical_arrival_on_time": int(sum(r["physical_arrival_on_time"] for r in records)), "host_confirmation_on_time": int(sum(r["host_confirmation_on_time"] for r in records)), "latency": {"samples": len(latencies), "mean_ms": float(np.mean(latencies)), "p95_ms": float(np.percentile(latencies, 95)), "p99_ms": float(np.percentile(latencies, 99)), "warmup_steps": 0, "includes": "public rule score and action selection; excludes environment progression and disk I/O", "raw_ms": latencies}, "device": str(device)}
    dump(output / "summary.json", result); return result


def completed_run_result(root: Path, variant: str, seed: int, label: str) -> dict[str, Any] | None:
    output = root / "runs" / f"m10-eawm-aux-{variant.lower()}-{seed}-{label}"
    status_path = output / "run-status.json"
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        return None
    return {"run_id": f"m10-eawm-aux-{variant.lower()}-{seed}-{label}", "variant": variant, "seed": seed, "output": str(output), "status": status["status"], "environment_steps": status.get("environment_steps", 0), "optimizer_updates": status.get("optimizer_updates", 0), "stop_reason": status.get("stop_reason"), "elapsed_seconds": status.get("elapsed_seconds")}


def run_directory_exists(root: Path, variant: str, seed: int, label: str) -> bool:
    return (root / "runs" / f"m10-eawm-aux-{variant.lower()}-{seed}-{label}" / "last.pt").exists()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "pilot", "formal"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 4:
        raise SystemExit("threads must be 1..4")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; refusing silent fallback")
    if args.root.exists() and any(args.root.iterdir()) and not args.resume:
        raise SystemExit(f"refusing non-empty matrix root: {args.root}")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    config = arrival_config()
    tapes = freeze_tapes(args.root, config)
    dump(args.root / "matrix-identity.json", {"source_commit": subprocess_git_commit(), "method": AUXILIARY_METHOD, "device": str(device), "threads": torch.get_num_threads() if device.type == "cpu" else None, "config": asdict(config), "budgets": {"pilot_per_variant_steps": PILOT_STEPS, "formal_per_run_steps": FORMAL_STEPS, "formal_runs": 6, "formal_total_steps": 49152, "max_updates_per_run": MAX_UPDATES, "rollout_steps": ROLLOUT_STEPS, "update_epochs": UPDATE_EPOCHS, "total_wall_seconds": TOTAL_WALL_SECONDS}, "initialization_audit": shared_initialization_audit(config, 1101), "world_model": "none; E uses public event auxiliary heads only", "validation_only_curve": True, "final_checkpoint_rule": "last checkpoint at fixed environment-step budget"})
    started = time.perf_counter(); deadline = started + TOTAL_WALL_SECONDS
    results = {"pilot": [], "formal": [], "validation": [], "final_test": None, "rule": None}
    if args.stage in ("all", "pilot"):
        for variant in ("H", "E"):
            result = completed_run_result(args.root, variant, 1101, "pilot") if args.resume else None
            if result is None:
                result = train_one(variant, 1101, PILOT_STEPS, max_updates=MAX_UPDATES, root=args.root, config=config, train_tape=tapes["train"], validation_tape=tapes["validation"], device=device, wall_deadline=min(deadline, time.perf_counter() + PILOT_GROUP_WALL_SECONDS), run_label="pilot", pilot=True, resume=args.resume and run_directory_exists(args.root, variant, 1101, "pilot"))
            results["pilot"].append(result)
        if args.stage == "pilot":
            dump(args.root / "matrix-results.json", results); return 0
    if args.stage in ("all", "formal"):
        for seed in SEEDS:
            for variant in ("H", "E"):
                if time.perf_counter() >= deadline:
                    break
                result = completed_run_result(args.root, variant, seed, "formal") if args.resume else None
                if result is None:
                    result = train_one(variant, seed, FORMAL_STEPS, max_updates=MAX_UPDATES, root=args.root, config=config, train_tape=tapes["train"], validation_tape=tapes["validation"], device=device, wall_deadline=deadline, run_label="formal", pilot=False, resume=args.resume and run_directory_exists(args.root, variant, seed, "formal"))
                results["formal"].append(result)
            if time.perf_counter() >= deadline:
                break
        if len(results["formal"]) == 6:
            results["rule"] = run_rules(tapes["final_test"], config, device, args.root)
            for result in results["formal"]:
                run_dir = Path(result["output"])
                policy = make_policy(config, result["variant"], result["seed"], device)
                payload = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
                policy.load_state_dict(payload["state_dict"])
                results["final_test"] = results["final_test"] or []
                results["final_test"].append(evaluate_variant(policy, result["variant"], tapes["final_test"], config, device, result["run_id"], run_dir))
        else:
            results["final_test"] = {"status": "not_run", "reason": "formal_matrix_incomplete_or_wall_clock_budget"}
    results["elapsed_seconds"] = time.perf_counter() - started
    results["status"] = "complete" if len(results["formal"]) == 6 else "incomplete"
    dump(args.root / "matrix-results.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
