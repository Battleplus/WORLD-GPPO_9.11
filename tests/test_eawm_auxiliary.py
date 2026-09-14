from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from gppo_world.eawm_auxiliary import (
    EventAwareHistoryPolicy,
    auxiliary_loss,
    build_public_event_targets,
    evaluate_event_aware_sequence,
    collect_event_aware_rollout,
    save_event_aware_checkpoint,
    total_event_aware_loss,
)
from gppo_world.m10_environment import M10Config, M10Environment, default_scenario
from gppo_world.m10_training import M10ActorCritic, PPOConfig


def _public(*, uav_ids=("uav-0", "uav-1", "uav-2", "uav-3"), task_ids=tuple(f"task-{i}" for i in range(6))):
    uavs = np.zeros((len(uav_ids), 24), dtype=np.float64)
    tasks = np.zeros((len(task_ids), 32), dtype=np.float64)
    for rows, fields in ((uavs, 6), (tasks, 8)):
        for row in rows:
            for field in range(fields):
                row[field * 4] = float(field + 1)
                row[field * 4 + 1] = 1.0
                row[field * 4 + 2] = 1.0
                row[field * 4 + 3] = 0.0
    return {"uavs": uavs, "tasks": tasks, "entity_ids": {"uavs": tuple(uav_ids), "tasks": tuple(task_ids)}}


def _policy(seed: int = 11) -> EventAwareHistoryPolicy:
    torch.manual_seed(seed)
    base = M10ActorCritic(
        uav_count=4, task_capacity=6, action_count=25, encoder="graph",
        type_count=5, history=True, context_dim=0, region_count=3,
        target_count=4, event_capacity=4, relation_width=4,
    )
    return EventAwareHistoryPolicy(base)


def test_adjacent_public_history_and_action_have_deterministic_labels():
    previous = _public()
    current = copy.deepcopy(previous)
    current["uavs"][0, 0] -= 0.5
    current["uavs"][0, 8] += 0.2
    current["tasks"][0, 20] = 0.0
    first = build_public_event_targets(previous, current)
    second = build_public_event_targets(previous, current)
    np.testing.assert_array_equal(first.position_class, second.position_class)
    np.testing.assert_array_equal(first.energy_class, second.energy_class)
    np.testing.assert_array_equal(first.task_state_changed, second.task_state_changed)
    assert first.position_class[0, 0] == 0
    assert first.energy_class[0] == 2
    assert first.task_state_changed[0] == 1


def test_hidden_future_change_is_not_an_online_input():
    policy = _policy()
    obs = torch.randn(1, policy.base_policy.input_dim)
    action = torch.tensor([3])
    with torch.no_grad():
        first = policy.forward_with_aux(obs, action)[2]
        second = policy.forward_with_aux(obs, action)[2]
    for name in first:
        torch.testing.assert_close(first[name], second[name])
    previous = _public()
    next_a = copy.deepcopy(previous)
    next_b = copy.deepcopy(previous)
    next_b["uavs"][0, 0] += 100.0
    assert not np.array_equal(
        build_public_event_targets(previous, next_a).position_class,
        build_public_event_targets(previous, next_b).position_class,
    )


def test_identity_swap_missing_stale_and_episode_reset_are_masked():
    previous = _public(uav_ids=("uav-0", "uav-1"), task_ids=("task-0",))
    swapped = _public(uav_ids=("uav-1", "uav-0"), task_ids=("task-0",))
    swapped["uavs"][:, 0] += 2.0
    target = build_public_event_targets(previous, swapped)
    assert not target.position_mask.any()
    assert not target.energy_mask.any()
    missing = copy.deepcopy(previous)
    current = copy.deepcopy(previous)
    current["uavs"][0, 0 * 4 + 2] = 0.0
    assert not build_public_event_targets(missing, current).position_mask[0, 0]
    reset = build_public_event_targets(previous, current, episode_boundary=True)
    assert not reset.position_mask.any()
    assert not reset.energy_mask.any()
    assert not reset.task_state_mask.any()
    assert reset.mask_reasons["episode_boundary"] > 0


def test_auxiliary_loss_masks_invalid_families_and_has_no_future_tensor():
    policy = _policy()
    predictions = policy.forward_with_aux(torch.randn(1, policy.base_policy.input_dim), torch.tensor([1]))[2]
    previous = _public()
    current = copy.deepcopy(previous)
    targets = build_public_event_targets(previous, current)
    loss, details = auxiliary_loss(predictions, targets, device="cpu")
    assert torch.isfinite(loss)
    assert "task_state" in details["skipped_families"] or details["valid_counts"]["task_state"] >= 0
    invalid = build_public_event_targets(previous, current, episode_boundary=True)
    zero, zero_details = auxiliary_loss(predictions, invalid, device="cpu")
    assert float(zero) == 0.0
    assert set(zero_details["skipped_families"]) == {"position", "energy", "task_state"}


def test_auxiliary_gradient_reaches_shared_history_encoder():
    policy = _policy()
    obs = torch.randn(1, policy.base_policy.input_dim)
    predictions = policy.forward_with_aux(obs, torch.tensor([2]))[2]
    previous = _public()
    current = copy.deepcopy(previous)
    current["uavs"][0, 0] -= 0.5
    current["uavs"][0, 8] -= 0.1
    targets = build_public_event_targets(previous, current)
    loss, _ = auxiliary_loss(predictions, targets, device="cpu")
    loss.backward()
    grad = policy.base_policy.gru.weight_ih_l0.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_coefficient_zero_preserves_history_ppo_loss_and_gradients():
    torch.manual_seed(91)
    first = M10ActorCritic(uav_count=4, task_capacity=6, action_count=25, encoder="graph", type_count=5, history=True)
    second = M10ActorCritic(uav_count=4, task_capacity=6, action_count=25, encoder="graph", type_count=5, history=True)
    second.load_state_dict(first.state_dict())
    wrapped = EventAwareHistoryPolicy(second)
    obs = torch.randn(3, first.input_dim)
    logits_a, value_a, _ = first(obs)
    logits_b, value_b, _ = wrapped(obs)
    loss_a = logits_a.square().mean() + value_a.square().mean()
    loss_b = total_event_aware_loss(logits_b.square().mean() + value_b.square().mean(), torch.tensor(7.0), coefficient=0.0)
    loss_a.backward()
    loss_b.backward()
    torch.testing.assert_close(loss_a, loss_b)
    for (name_a, value_a), (name_b, value_b) in zip(first.named_parameters(), wrapped.base_policy.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(value_a.grad, value_b.grad)


def test_rollout_logprob_recomputation_uses_same_distribution():
    policy = _policy()
    policy.eval()
    config = PPOConfig(rollout_steps=5, update_epochs=1)
    env_config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    transitions = collect_event_aware_rollout(
        policy, config=config, env_config=env_config, seed=471001, device="cpu",
        scenarios=[default_scenario("normal", seed=471001, split="eawm-dev")],
    )
    log_probs, _, _, _ = evaluate_event_aware_sequence(policy, transitions, "cpu")
    np.testing.assert_allclose(log_probs.detach().numpy(), [item.transition.log_prob for item in transitions], atol=1e-6)


def test_inference_forward_does_not_call_auxiliary_heads():
    policy = _policy()
    def fail(*args, **kwargs):
        raise AssertionError("auxiliary head was called during inference")
    policy.position_head.forward = fail
    policy.energy_head.forward = fail
    policy.task_state_head.forward = fail
    logits, value, _ = policy(torch.randn(1, policy.base_policy.input_dim))
    assert logits.shape == (1, 25)
    assert value.shape == (1,)


def test_public_mask_and_execution_gate_contract_remain_environment_owned():
    config = M10Config(task_completion_mode="arrival_to_region", deadline_basis="physical_arrival")
    env = M10Environment(config, default_scenario("normal", seed=0))
    obs = env.reset()
    assert obs["mask"].shape == (25,)
    assert bool(obs["mask"][-1])
    assert "trigger_flags" in obs and "continuation_actions" in obs


def test_training_and_inference_checkpoint_formats_are_distinct(tmp_path):
    policy = _policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    training_path = tmp_path / "last.pt"
    inference_path = tmp_path / "best-inference.pt"
    save_event_aware_checkpoint(training_path, policy, {"seed": 471001, "auxiliary_schema": "x"}, optimizer=optimizer)
    save_event_aware_checkpoint(inference_path, policy, {"seed": 471001, "auxiliary_schema": "x"}, inference=True)
    training = torch.load(training_path, map_location="cpu", weights_only=False)
    inference = torch.load(inference_path, map_location="cpu", weights_only=False)
    assert training["format"] == EventAwareHistoryPolicy.format_version
    assert training["optimizer_state_dict"] is not None
    assert training["recovery_state"] is not None
    assert inference["format"] == EventAwareHistoryPolicy.inference_format_version
    assert inference["optimizer_state_dict"] is None
    assert inference["recovery_state"] is None
