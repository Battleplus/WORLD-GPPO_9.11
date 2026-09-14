"""Future bounded training entry for EAWM-inspired auxiliary GPPO-History.

This entry is intentionally not invoked by the development-validation command.
It uses the arrival protocol and public weak-communication tape, freezes the
auxiliary coefficient at 0.1, and requires an explicit ``--resume`` to reuse
an output directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gppo_world.eawm_auxiliary import (  # noqa: E402
    AUXILIARY_METHOD,
    AUXILIARY_SCHEMA,
    EventAwareHistoryPolicy,
    collect_event_aware_rollout,
    save_event_aware_checkpoint,
    update_event_aware_policy,
)
from gppo_world.m10_environment import M10Config, weak_communication_tape  # noqa: E402
from gppo_world.m10_training import M10ActorCritic, PPOConfig, seed_everything  # noqa: E402


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_policy(config: M10Config) -> EventAwareHistoryPolicy:
    return EventAwareHistoryPolicy(M10ActorCritic(
        uav_count=config.uav_count, task_capacity=config.task_capacity,
        action_count=config.action_count, encoder="graph", type_count=5,
        history=True, context_dim=0, region_count=config.region_count,
        target_count=config.target_count, event_capacity=config.event_capacity,
        relation_width=config.relation_width,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--environment-steps", type=int, default=8192)
    parser.add_argument("--max-updates", type=int, default=4096)
    parser.add_argument("--max-wall-seconds", type=float, default=3600.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 4:
        raise SystemExit("threads must be within 1..4")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; refusing silent device fallback")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit("output exists; pass --resume explicitly to reuse it")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    seed_everything(args.seed)
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    ppo = PPOConfig(rollout_steps=64, update_epochs=1, minibatch_size=64)
    scenarios = list(weak_communication_tape("train", count=8, base_seed=471001, level="composite"))
    policy = make_policy(config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo.learning_rate)
    state = {"environment_steps": 0, "optimizer_updates": 0, "best_loss": None, "data_order": [], "early_stopping": {"reason": "not_applicable"}}
    last_path = args.out / "last.pt"
    identity_path = args.out / "run-identity.json"
    if args.resume:
        if not last_path.exists() or not identity_path.exists():
            raise SystemExit("--resume requires last.pt and run-identity.json")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if identity.get("run_id") != args.run_id or identity.get("seed") != args.seed:
            raise SystemExit("resume identity mismatch")
        payload = torch.load(last_path, map_location=device, weights_only=False)
        if payload.get("format") != EventAwareHistoryPolicy.format_version:
            raise SystemExit("last checkpoint is not an event-aware training checkpoint")
        policy.load_state_dict(payload["state_dict"])
        if payload.get("optimizer_state_dict") is None:
            raise SystemExit("training checkpoint lacks optimizer state")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state.update(payload.get("metadata", {}).get("state", {}))
        recovery = payload.get("recovery_state", {})
        if recovery.get("rng_state"):
            import random
            random.setstate(recovery["rng_state"]["python"])
            np.random.set_state(recovery["rng_state"]["numpy"])
            torch.set_rng_state(recovery["rng_state"]["torch"])
            if args.device == "cuda" and recovery["rng_state"].get("cuda") is not None:
                torch.cuda.set_rng_state_all(recovery["rng_state"]["cuda"])

    dump(identity_path, {
        "run_id": args.run_id, "status": "running", "method": AUXILIARY_METHOD,
        "auxiliary_schema": AUXILIARY_SCHEMA, "seed": args.seed,
        "device": str(device), "threads": args.threads if args.device == "cpu" else None,
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "protocol": asdict(config), "ppo": asdict(ppo),
        "environment_step_budget": args.environment_steps, "optimizer_update_budget": args.max_updates,
        "wall_seconds_budget": args.max_wall_seconds, "world_model": "not used; auxiliary public transition heads only",
        "scenarios": [scenario.tape_id for scenario in scenarios], "resumed": args.resume,
    })
    log_path = args.out / "updates.jsonl"
    started = time.perf_counter()
    while state["environment_steps"] < args.environment_steps and state["optimizer_updates"] < args.max_updates:
        if time.perf_counter() - started >= args.max_wall_seconds:
            state["early_stopping"] = {"reason": "wall_clock_budget"}
            break
        ppo_batch = PPOConfig(**{**asdict(ppo), "rollout_steps": min(ppo.rollout_steps, args.environment_steps - state["environment_steps"])})
        transitions = collect_event_aware_rollout(policy, config=ppo_batch, env_config=config, seed=args.seed, device=device, scenarios=scenarios)
        metrics = update_event_aware_policy(policy, transitions, ppo=ppo_batch, device=device, optimizer=optimizer, coefficient=0.1)
        state["environment_steps"] += len(transitions)
        state["optimizer_updates"] += int(metrics["optimizer_step"])
        state["data_order"].append([{"index": index, "action": item.transition.action, "episode_start": item.transition.episode_start} for index, item in enumerate(transitions)])
        metrics.update({"environment_steps": state["environment_steps"], "optimizer_updates": state["optimizer_updates"], "elapsed_seconds": time.perf_counter() - started})
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        if state["best_loss"] is None or metrics["loss"] < state["best_loss"]:
            state["best_loss"] = metrics["loss"]
            save_event_aware_checkpoint(args.out / "best-inference.pt", policy, {"run_id": args.run_id, "seed": args.seed, "environment_steps": state["environment_steps"], "optimizer_updates": state["optimizer_updates"], "selection_scope": "training loss only; no validation in this entry"}, inference=True)
        metadata = {"run_id": args.run_id, "seed": args.seed, "environment_steps": state["environment_steps"], "optimizer_updates": state["optimizer_updates"], "state": state, "data_order": state["data_order"], "early_stopping": state["early_stopping"]}
        save_event_aware_checkpoint(last_path, policy, metadata, optimizer=optimizer)
    state["status"] = "complete" if state["environment_steps"] >= args.environment_steps else "stopped"
    state["elapsed_seconds"] = time.perf_counter() - started
    dump(args.out / "run-status.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
