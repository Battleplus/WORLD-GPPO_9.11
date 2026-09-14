"""Event-aware auxiliary learning for arrival-protocol GPPO-History.

The module is deliberately separate from candidate-consequence fusion.  It
uses only the current public observation, the policy's own recurrent history
representation, and the sampled action.  The next public observation is used
only to build detached supervision targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .m10_environment import M10Config, M10Environment, M10Scenario, scenario_tape
from .m10_training import M10ActorCritic, PPOConfig, Transition, _gae, masked_distribution, seed_everything


AUXILIARY_SCHEMA = "gppo-eawm-inspired-public-transition-v1"
AUXILIARY_METHOD = "EAWM-inspired auxiliary GPPO-History"
POSITION_CLASSES = ("decrease", "unchanged", "increase")
STATE_CLASSES = ("unchanged", "changed")
INVARIANCE_TOLERANCE = 1e-6
UAV_FIELDS = ("x", "y", "energy", "alive", "connected", "idle")
TASK_FIELDS = ("x", "y", "deadline", "remaining_service", "priority", "pending", "region_id", "target_id")


def _field(row: np.ndarray, index: int) -> tuple[float, bool]:
    offset = index * 4
    if row.ndim != 1 or offset + 3 >= row.shape[0]:
        return float("nan"), False
    value = float(row[offset])
    valid = bool(row[offset + 2] > 0.5)
    return value, valid and np.isfinite(value)


def _ids(observation: dict[str, Any], kind: str) -> tuple[str, ...]:
    payload = observation.get("entity_ids")
    if not isinstance(payload, dict) or kind not in payload:
        raise ValueError(f"public observation lacks auditable {kind} identities")
    values = tuple(str(item) for item in payload[kind])
    if len(set(values)) != len(values):
        raise ValueError(f"public observation has duplicate {kind} identities")
    return values


def _classify_delta(previous: float, current: float) -> int:
    delta = current - previous
    if abs(delta) <= INVARIANCE_TOLERANCE:
        return 1
    return 0 if delta < 0 else 2


@dataclass(frozen=True)
class PublicEventTargets:
    """Detached targets from two adjacent public snapshots.

    Invalid labels use -1 and must always be excluded by the corresponding
    mask.  Shapes are [U,2] for x/y position, [U] for energy, and [T] for the
    public task ``pending`` state.
    """

    position_class: np.ndarray
    position_mask: np.ndarray
    energy_class: np.ndarray
    energy_mask: np.ndarray
    task_state_changed: np.ndarray
    task_state_mask: np.ndarray
    episode_boundary: bool
    identity_matches: dict[str, int]
    mask_reasons: dict[str, int]

    def __post_init__(self) -> None:
        if self.position_class.shape != self.position_mask.shape or self.position_class.ndim != 2 or self.position_class.shape[1] != 2:
            raise ValueError("position target/mask must be [uav,2]")
        if self.energy_class.shape != self.energy_mask.shape or self.energy_class.ndim != 1:
            raise ValueError("energy target/mask must be [uav]")
        if self.task_state_changed.shape != self.task_state_mask.shape or self.task_state_changed.ndim != 1:
            raise ValueError("task-state target/mask must be [task]")
        if np.any(self.position_mask & ~np.isin(self.position_class, (0, 1, 2))):
            raise ValueError("valid position labels must be in {0,1,2}")
        if np.any(self.energy_mask & ~np.isin(self.energy_class, (0, 1, 2))):
            raise ValueError("valid energy labels must be in {0,1,2}")
        if np.any(self.task_state_mask & ~np.isin(self.task_state_changed, (0, 1))):
            raise ValueError("valid task-state labels must be binary")

    @property
    def valid_counts(self) -> dict[str, int]:
        return {
            "position": int(self.position_mask.sum()),
            "energy": int(self.energy_mask.sum()),
            "task_state": int(self.task_state_mask.sum()),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": AUXILIARY_SCHEMA,
            "position_class": self.position_class.tolist(),
            "position_mask": self.position_mask.tolist(),
            "energy_class": self.energy_class.tolist(),
            "energy_mask": self.energy_mask.tolist(),
            "task_state_changed": self.task_state_changed.tolist(),
            "task_state_mask": self.task_state_mask.tolist(),
            "episode_boundary": self.episode_boundary,
            "identity_matches": self.identity_matches,
            "mask_reasons": self.mask_reasons,
        }


def build_public_event_targets(previous: dict[str, Any], current: dict[str, Any], *, episode_boundary: bool = False) -> PublicEventTargets:
    """Build labels strictly from adjacent public observations.

    No simulator, hidden state, event schedule, completion ledger or future
    value is accepted.  Entity alignment is by the explicit public IDs in
    each observation, never by an assumed slot position.
    """
    previous_uav_ids, current_uav_ids = _ids(previous, "uavs"), _ids(current, "uavs")
    previous_task_ids, current_task_ids = _ids(previous, "tasks"), _ids(current, "tasks")
    previous_uavs = np.asarray(previous.get("uavs"), dtype=np.float64)
    current_uavs = np.asarray(current.get("uavs"), dtype=np.float64)
    previous_tasks = np.asarray(previous.get("tasks"), dtype=np.float64)
    current_tasks = np.asarray(current.get("tasks"), dtype=np.float64)
    uav_count = max(len(previous_uav_ids), len(current_uav_ids), previous_uavs.shape[0], current_uavs.shape[0])
    task_count = max(len(previous_task_ids), len(current_task_ids), previous_tasks.shape[0], current_tasks.shape[0])
    position_class = np.full((uav_count, 2), -1, dtype=np.int8)
    position_mask = np.zeros((uav_count, 2), dtype=bool)
    energy_class = np.full((uav_count,), -1, dtype=np.int8)
    energy_mask = np.zeros((uav_count,), dtype=bool)
    task_state_changed = np.full((task_count,), -1, dtype=np.int8)
    task_state_mask = np.zeros((task_count,), dtype=bool)
    identity_matches = {"uav": 0, "task": 0}
    reasons = {"episode_boundary": 0, "identity_mismatch": 0, "missing_or_invalid": 0, "nonfinite": 0}
    if episode_boundary:
        reasons["episode_boundary"] = uav_count * 3 + task_count
        return PublicEventTargets(position_class, position_mask, energy_class, energy_mask, task_state_changed, task_state_mask, True, identity_matches, reasons)

    for index in range(uav_count):
        same_identity = index < len(previous_uav_ids) and index < len(current_uav_ids) and previous_uav_ids[index] == current_uav_ids[index]
        if same_identity:
            identity_matches["uav"] += 1
        for coordinate, field_index in enumerate((0, 1)):
            if not same_identity:
                reasons["identity_mismatch"] += 1
                continue
            old, old_valid = _field(previous_uavs[index], field_index) if index < previous_uavs.shape[0] else (float("nan"), False)
            new, new_valid = _field(current_uavs[index], field_index) if index < current_uavs.shape[0] else (float("nan"), False)
            if not old_valid or not new_valid:
                reasons["missing_or_invalid"] += 1
            elif not np.isfinite(old) or not np.isfinite(new):
                reasons["nonfinite"] += 1
            else:
                position_class[index, coordinate] = _classify_delta(old, new)
                position_mask[index, coordinate] = True
        if not same_identity:
            reasons["identity_mismatch"] += 1
        else:
            old, old_valid = _field(previous_uavs[index], 2) if index < previous_uavs.shape[0] else (float("nan"), False)
            new, new_valid = _field(current_uavs[index], 2) if index < current_uavs.shape[0] else (float("nan"), False)
            if not old_valid or not new_valid:
                reasons["missing_or_invalid"] += 1
            elif not np.isfinite(old) or not np.isfinite(new):
                reasons["nonfinite"] += 1
            else:
                energy_class[index] = _classify_delta(old, new)
                energy_mask[index] = True

    for index in range(task_count):
        same_identity = index < len(previous_task_ids) and index < len(current_task_ids) and previous_task_ids[index] == current_task_ids[index]
        if same_identity:
            identity_matches["task"] += 1
            old, old_valid = _field(previous_tasks[index], 5) if index < previous_tasks.shape[0] else (float("nan"), False)
            new, new_valid = _field(current_tasks[index], 5) if index < current_tasks.shape[0] else (float("nan"), False)
            if not old_valid or not new_valid:
                reasons["missing_or_invalid"] += 1
            elif not np.isfinite(old) or not np.isfinite(new):
                reasons["nonfinite"] += 1
            else:
                task_state_changed[index] = int(abs(new - old) > INVARIANCE_TOLERANCE)
                task_state_mask[index] = True
        else:
            reasons["identity_mismatch"] += 1
    return PublicEventTargets(position_class, position_mask, energy_class, energy_mask, task_state_changed, task_state_mask, False, identity_matches, reasons)


class EventAwareHistoryPolicy(nn.Module):
    """History policy with training-only public transition prediction heads."""

    format_version = "gppo-eawm-inspired-arrival-training-v1"
    inference_format_version = "gppo-eawm-inspired-arrival-inference-v1"

    def __init__(self, base_policy: M10ActorCritic):
        super().__init__()
        if not base_policy.history or base_policy.encoder_name != "graph" or base_policy.type_count != 5:
            raise ValueError("EAWM-inspired auxiliary policy requires Graph-5 GPPO-History")
        self.base_policy = base_policy
        self.action_count = base_policy.action_count
        self.uav_count = base_policy.uav_count
        self.task_capacity = base_policy.task_capacity
        self.history_width = 128
        aux_width = self.history_width + self.action_count
        self.position_head = nn.Linear(aux_width, self.uav_count * 2 * 3)
        self.energy_head = nn.Linear(aux_width, self.uav_count * 3)
        self.task_state_head = nn.Linear(aux_width, self.task_capacity)

    def _policy_logits(self, features: torch.Tensor, pair_messages: torch.Tensor | None) -> torch.Tensor:
        if pair_messages is None:
            if self.base_policy.actor is None:
                raise RuntimeError("MLP actor unexpectedly missing")
            return self.base_policy.actor(features)
        pair_logits = self.base_policy.pair_actor(torch.cat((pair_messages, features[:, None, None, :].expand(-1, self.uav_count, self.task_capacity, -1)), dim=-1)).squeeze(-1)
        return torch.cat((pair_logits.reshape(-1, self.uav_count * self.task_capacity), self.base_policy.noop_actor(features)), dim=-1)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        features, pair_messages, next_hidden = self.base_policy._features(obs, hidden)
        return self._policy_logits(features, pair_messages), self.base_policy.critic(features).squeeze(-1), next_hidden

    def forward_with_aux(self, obs: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor | None = None):
        features, pair_messages, next_hidden = self.base_policy._features(obs, hidden)
        logits = self._policy_logits(features, pair_messages)
        action_one_hot = F.one_hot(action.long(), self.action_count).to(dtype=features.dtype)
        aux_input = torch.cat((features, action_one_hot), dim=-1)
        return (
            logits,
            self.base_policy.critic(features).squeeze(-1),
            {
                "position_logits": self.position_head(aux_input).reshape(-1, self.uav_count, 2, 3),
                "energy_logits": self.energy_head(aux_input).reshape(-1, self.uav_count, 3),
                "task_state_logits": self.task_state_head(aux_input),
            },
            next_hidden,
        )


def _target_tensors(targets: PublicEventTargets, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "position_class": torch.as_tensor(targets.position_class, dtype=torch.long, device=device),
        "position_mask": torch.as_tensor(targets.position_mask, dtype=torch.bool, device=device),
        "energy_class": torch.as_tensor(targets.energy_class, dtype=torch.long, device=device),
        "energy_mask": torch.as_tensor(targets.energy_mask, dtype=torch.bool, device=device),
        "task_state_changed": torch.as_tensor(targets.task_state_changed, dtype=torch.float32, device=device),
        "task_state_mask": torch.as_tensor(targets.task_state_mask, dtype=torch.bool, device=device),
    }


def auxiliary_loss(predictions: dict[str, torch.Tensor], targets: PublicEventTargets, *, device: torch.device | str) -> tuple[torch.Tensor, dict[str, Any]]:
    device_obj = torch.device(device)
    target = _target_tensors(targets, device_obj)
    family_losses: list[torch.Tensor] = []
    details: dict[str, Any] = {"valid_counts": targets.valid_counts, "skipped_families": []}
    if bool(target["position_mask"].any()):
        logits = predictions["position_logits"]
        if logits.ndim == target["position_mask"].ndim + 2 and tuple(logits.shape[1:-1]) == tuple(target["position_mask"].shape):
            if logits.shape[0] != 1:
                raise ValueError("auxiliary_loss expects one transition target per prediction")
            logits = logits[0]
        logits = logits.reshape(-1, 3)
        mask = target["position_mask"].reshape(-1)
        family_losses.append(F.cross_entropy(logits[mask], target["position_class"].reshape(-1)[mask]))
    else:
        details["skipped_families"].append("position")
    if bool(target["energy_mask"].any()):
        logits = predictions["energy_logits"]
        if logits.ndim == target["energy_mask"].ndim + 2 and tuple(logits.shape[1:-1]) == tuple(target["energy_mask"].shape):
            if logits.shape[0] != 1:
                raise ValueError("auxiliary_loss expects one transition target per prediction")
            logits = logits[0]
        logits = logits.reshape(-1, 3)
        mask = target["energy_mask"].reshape(-1)
        family_losses.append(F.cross_entropy(logits[mask], target["energy_class"].reshape(-1)[mask]))
    else:
        details["skipped_families"].append("energy")
    if bool(target["task_state_mask"].any()):
        logits = predictions["task_state_logits"]
        if logits.ndim == target["task_state_mask"].ndim + 1:
            if logits.shape[0] != 1:
                raise ValueError("auxiliary_loss expects one transition target per prediction")
            logits = logits[0]
        logits = logits.reshape(-1)
        mask = target["task_state_mask"].reshape(-1)
        family_losses.append(F.binary_cross_entropy_with_logits(logits[mask], target["task_state_changed"].reshape(-1)[mask]))
    else:
        details["skipped_families"].append("task_state")
    if not family_losses:
        zero = predictions["position_logits"].sum() * 0.0
        details["event_loss"] = 0.0
        return zero, details
    loss = torch.stack(family_losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite auxiliary event loss")
    details["event_loss"] = float(loss.detach().cpu())
    return loss, details


def total_event_aware_loss(ppo_loss: torch.Tensor, event_loss: torch.Tensor, *, coefficient: float = 0.1) -> torch.Tensor:
    if coefficient != 0.1 and coefficient != 0.0:
        raise ValueError("the frozen auxiliary coefficient is 0.1; coefficient 0 is test-only")
    result = ppo_loss + coefficient * event_loss
    if not torch.isfinite(result):
        raise FloatingPointError("non-finite total event-aware loss")
    return result


@dataclass
class EventAwareTransition:
    transition: Transition
    next_public_observation: dict[str, Any]
    targets: PublicEventTargets


def collect_event_aware_rollout(policy: EventAwareHistoryPolicy, *, config: PPOConfig, env_config: M10Config,
                                seed: int, device: torch.device | str, scenarios: Iterable[M10Scenario]) -> list[EventAwareTransition]:
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("event-aware rollout requires frozen scenarios")
    device_obj = torch.device(device)
    env = M10Environment(env_config, scenario_list[0])
    obs = env.reset()
    hidden = None
    result: list[EventAwareTransition] = []
    scenario_index = 0
    episode_start = True
    for index in range(config.rollout_steps):
        public = torch.as_tensor(obs["flat"], dtype=torch.float32, device=device_obj)[None, :]
        mask = torch.as_tensor(obs["mask"], dtype=torch.bool, device=device_obj)[None, :]
        logits, value, hidden = policy(public, hidden)
        distribution = masked_distribution(logits, mask)
        action_tensor = distribution.sample()
        action = int(action_tensor.item())
        next_obs, reward, done, info = env.step(action, submit_command=True)
        terminated = bool(info.get("terminated", done))
        truncated = bool(info.get("truncated", False)) or (not done and index == config.rollout_steps - 1)
        target = build_public_event_targets(obs, next_obs, episode_boundary=False)
        next_value = 0.0 if terminated else float(policy(torch.as_tensor(next_obs["flat"], dtype=torch.float32, device=device_obj)[None, :], hidden)[1].item())
        base = Transition(
            obs=np.asarray(obs["flat"], dtype=np.float32), context=np.zeros(0, dtype=np.float32),
            mask=np.asarray(obs["mask"], dtype=bool).copy(), action=action,
            log_prob=float(distribution.log_prob(action_tensor).item()), value=float(value.item()),
            reward=float(reward), done=bool(terminated or truncated), episode_start=episode_start,
            info=dict(info), terminated=terminated, truncated=truncated, next_value=next_value,
            actor_decision=True,
        )
        result.append(EventAwareTransition(base, next_obs, target))
        if done:
            scenario_index = (scenario_index + 1) % len(scenario_list)
            env = M10Environment(env_config, scenario_list[scenario_index])
            obs = env.reset()
            hidden = None
            episode_start = True
        else:
            obs = next_obs
            episode_start = False
    return result


def evaluate_event_aware_sequence(policy: EventAwareHistoryPolicy, transitions: list[EventAwareTransition], device: torch.device | str):
    device_obj = torch.device(device)
    hidden = None
    log_probs, values, entropies, predictions = [], [], [], []
    for item in transitions:
        if item.transition.episode_start:
            hidden = None
        obs = torch.as_tensor(item.transition.obs, dtype=torch.float32, device=device_obj)[None, :]
        action = torch.tensor([item.transition.action], dtype=torch.long, device=device_obj)
        mask = torch.as_tensor(item.transition.mask, dtype=torch.bool, device=device_obj)[None, :]
        logits, value, prediction, hidden = policy.forward_with_aux(obs, action, hidden)
        distribution = masked_distribution(logits, mask)
        log_probs.append(distribution.log_prob(action)[0])
        values.append(value[0])
        entropies.append(distribution.entropy()[0])
        predictions.append(prediction)
    return torch.stack(log_probs), torch.stack(values), torch.stack(entropies), predictions


def update_event_aware_policy(policy: EventAwareHistoryPolicy, transitions: list[EventAwareTransition], *, ppo: PPOConfig,
                              device: torch.device | str, optimizer: torch.optim.Optimizer, coefficient: float = 0.1) -> dict[str, Any]:
    device_obj = torch.device(device)
    base_transitions = [item.transition for item in transitions]
    advantages, returns = _gae(base_transitions, ppo.gamma, ppo.gae_lambda)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    old_log_probs = torch.as_tensor([item.transition.log_prob for item in transitions], dtype=torch.float32, device=device_obj)
    new_log_probs, values, entropies, predictions = evaluate_event_aware_sequence(policy, transitions, device_obj)
    advantages, returns = advantages.to(device_obj), returns.to(device_obj)
    ratio = (new_log_probs - old_log_probs).exp()
    clipped = torch.clamp(ratio, 1.0 - ppo.clip_epsilon, 1.0 + ppo.clip_epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = entropies.mean()
    ppo_loss = policy_loss + ppo.value_weight * value_loss - ppo.entropy_weight * entropy
    last: dict[str, Any] = {}
    for epoch in range(ppo.update_epochs):
        if epoch:
            new_log_probs, values, entropies, predictions = evaluate_event_aware_sequence(policy, transitions, device_obj)
            ratio = (new_log_probs - old_log_probs).exp()
            clipped = torch.clamp(ratio, 1.0 - ppo.clip_epsilon, 1.0 + ppo.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = entropies.mean()
            ppo_loss = policy_loss + ppo.value_weight * value_loss - ppo.entropy_weight * entropy
        event_losses, event_details = [], []
        for prediction, item in zip(predictions, transitions):
            event_loss_item, details = auxiliary_loss(prediction, item.targets, device=device_obj)
            if any(item.targets.valid_counts.values()):
                event_losses.append(event_loss_item)
            event_details.append(details)
        event_loss = torch.stack(event_losses).mean() if event_losses else ppo_loss * 0.0
        loss = total_event_aware_loss(ppo_loss, event_loss, coefficient=coefficient)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite event-aware PPO loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo.grad_clip))
        if not np.isfinite(grad_norm):
            raise FloatingPointError("non-finite event-aware gradient")
        optimizer.step()
        for value in policy.state_dict().values():
            if not torch.isfinite(value).all():
                raise FloatingPointError("non-finite event-aware parameter")
        for optimizer_state in optimizer.state.values():
            for value in optimizer_state.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise FloatingPointError("non-finite event-aware optimizer state")
        last = {
            "loss": float(loss.detach().cpu()),
            "ppo_loss": float(ppo_loss.detach().cpu()),
            "event_loss": float(event_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "approx_kl": float((old_log_probs - new_log_probs).mean().detach().cpu()),
            "clip_fraction": float(((ratio - 1.0).abs() > ppo.clip_epsilon).float().mean().detach().cpu()),
            "optimizer_step": 1,
            "grad_norm": grad_norm,
            "valid_event_steps": len(event_losses),
            "event_details": event_details,
            "update_epoch": epoch + 1,
        }
    last["optimizer_steps"] = ppo.update_epochs
    return last


def save_event_aware_checkpoint(path, policy: EventAwareHistoryPolicy, metadata: dict[str, Any], *, inference: bool = False, optimizer: torch.optim.Optimizer | None = None) -> None:
    recovery = None if inference else {
        "seed": metadata.get("seed"),
        "environment_steps": metadata.get("environment_steps", 0),
        "optimizer_updates": metadata.get("optimizer_updates", 0),
        "data_order": metadata.get("data_order", []),
        "early_stopping": metadata.get("early_stopping", {}),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    payload = {
        "format": EventAwareHistoryPolicy.inference_format_version if inference else EventAwareHistoryPolicy.format_version,
        "metadata": dict(metadata),
        "state_dict": policy.state_dict(),
        "optimizer_state_dict": None if inference or optimizer is None else optimizer.state_dict(),
        "recovery_state": recovery,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


__all__ = [
    "AUXILIARY_SCHEMA", "AUXILIARY_METHOD", "POSITION_CLASSES", "STATE_CLASSES", "INVARIANCE_TOLERANCE",
    "PublicEventTargets", "build_public_event_targets", "EventAwareHistoryPolicy", "auxiliary_loss",
    "total_event_aware_loss", "EventAwareTransition", "collect_event_aware_rollout",
    "evaluate_event_aware_sequence", "update_event_aware_policy", "save_event_aware_checkpoint",
]
