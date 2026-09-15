import torch

import tools.run_minimal_scheduling_continuation as continuation
from gppo_world.minimal_scheduling import Instance, Subtask, UAV, initial_state, legal_actions


def fixture():
    tasks = (
        Subtask("a0", "a", 0, (0, 0), 2, (0, 1), ()),
        Subtask("a1", "a", 1, (0, 0), 1, (0, 1), ("a0",)),
        Subtask("b0", "b", 0, (10, 0), 2, (0, 1), ()),
        Subtask("b1", "b", 1, (10, 0), 1, (0, 1), ("b0",)),
    )
    return Instance("continuation-fixture", "homogeneous", (UAV(0, (0, 0)), UAV(1, (10, 0))), tasks)


class SpyScorer:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def __call__(self, sample):
        self.calls += 1
        self.sample = sample
        return torch.tensor(self.scores, dtype=torch.float32)


def test_model_is_called_once_then_greedy_continues_and_ledger_replays(monkeypatch):
    instance = fixture()
    actions = legal_actions(instance, initial_state(instance))
    spy = SpyScorer([0.0] * len(actions))
    monkeypatch.setattr(continuation, "solve_exact", lambda *a, **k: (_ for _ in ()).throw(AssertionError("solver entered policy path")))
    first, evidence, _ = continuation.choose_model_action(spy, instance, torch.device("cpu"))
    schedule = continuation.execute_schedule(instance, forced_first=first)
    assert spy.calls == 1
    assert evidence["model_forward_calls"] == 1
    assert schedule["decisions"][0]["controller"] == "frozen_model_first_action"
    assert all(x["controller"] == "frozen_greedy_continuation" for x in schedule["decisions"][1:])
    assert continuation.replay_validate(instance, schedule)["replayed_makespan"] == schedule["makespan"]


def test_initial_model_tie_uses_frozen_greedy_action():
    instance = fixture()
    actions = legal_actions(instance, initial_state(instance))
    spy = SpyScorer([0.25] * len(actions))
    selected, evidence, _ = continuation.choose_model_action(spy, instance, torch.device("cpu"))
    assert selected == continuation.greedy_action(instance, actions)
    assert evidence["tie_count"] == len(actions)


def test_ledger_rejects_illegal_or_overlapping_schedule():
    instance = fixture()
    schedule = continuation.execute_schedule(instance)
    schedule["assignments"][1]["uav_id"] = schedule["assignments"][0]["uav_id"]
    try:
        continuation.replay_validate(instance, schedule)
    except AssertionError:
        pass
    else:
        raise AssertionError("overlapping UAV use was not rejected")


def test_greedy_continuation_uses_current_uav_position():
    instance = fixture()
    state = initial_state(instance)
    from gppo_world.minimal_scheduling import assign, wait_to_next_event
    state, _ = assign(instance, state, ("a0", 0))
    state, _ = wait_to_next_event(instance, state)
    actions = legal_actions(instance, state)
    assert continuation.greedy_action(instance, actions, state.positions) == min(
        actions,
        key=lambda action: (
            continuation.manhattan(state.positions[action[1]], instance.by_id[action[0]].position)
            + instance.by_id[action[0]].process_time,
            action[0], action[1],
        ),
    )
