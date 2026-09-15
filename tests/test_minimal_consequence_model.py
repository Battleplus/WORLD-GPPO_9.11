import copy

import torch

from gppo_world.minimal_consequence_model import CandidateRegretNet, graph_sample, instance_loss
from gppo_world.minimal_scheduling import Instance, Subtask, UAV, initial_state, legal_actions, solve_exact


def record():
    tasks = (
        Subtask("a0", "a", 0, (0, 0), 2, (0, 1), ()),
        Subtask("a1", "a", 1, (0, 0), 1, (0, 1), ("a0",)),
        Subtask("b0", "b", 0, (10, 0), 2, (0, 1), ()),
        Subtask("b1", "b", 1, (10, 0), 1, (0, 1), ("b0",)),
    )
    instance = Instance("model-test", "homogeneous", (UAV(0, (0, 0)), UAV(1, (10, 0))), tasks)
    exact = solve_exact(instance)
    c = exact.makespan
    scale = 20.0
    candidates = []
    for task, uav in legal_actions(instance, initial_state(instance)):
        q = solve_exact(instance, forced_first=(task, uav)).makespan
        candidates.append({"task_id": task, "task_index": next(i for i, t in enumerate(tasks) if t.task_id == task), "uav_id": uav, "q_star": q, "c_star": c, "scale": scale, "normalized_regret": (q-c)/scale})
    return {"instance": {"uavs": [{"uav_id": u.uav_id, "position": list(u.position)} for u in instance.uavs], "subtasks": [{"task_id": t.task_id, "chain_id": t.chain_id, "index": t.index, "position": list(t.position), "process_time": t.process_time, "eligible_uavs": list(t.eligible_uavs), "predecessors": list(t.predecessors)} for t in tasks]}, "candidates": candidates}


def test_labels_are_not_model_inputs_and_loss_keeps_ties_equal():
    row = record()
    model = CandidateRegretNet()
    left = graph_sample(row, torch.device("cpu"))
    right_row = copy.deepcopy(row)
    right_row["candidates"][0]["normalized_regret"] += 99.0
    right = graph_sample(right_row, torch.device("cpu"))
    torch.manual_seed(7)
    model = CandidateRegretNet()
    assert torch.allclose(model(left), model(right))
    loss, parts = instance_loss(model, left)
    assert torch.isfinite(loss) and parts["rank_pairs"] >= 0


def test_candidate_graph_uses_legal_actions_and_solver_labels_are_recomputable():
    row = record()
    sample = graph_sample(row, torch.device("cpu"))
    assert len(sample.candidate_actions) == len(row["candidates"])
    assert all(float(x["normalized_regret"]) >= -1e-8 for x in row["candidates"])


def test_candidate_scores_follow_uav_permutation():
    row = record()
    model = CandidateRegretNet()
    model.eval()
    original = graph_sample(row, torch.device("cpu"))
    swapped = copy.deepcopy(row)
    swapped["instance"]["uavs"] = list(reversed(swapped["instance"]["uavs"]))
    for u in swapped["instance"]["uavs"]:
        u["uav_id"] = 1 - u["uav_id"]
    for c in swapped["candidates"]:
        c["uav_id"] = 1 - c["uav_id"]
    permuted = graph_sample(swapped, torch.device("cpu"))
    with torch.no_grad():
        first = model(original)
        second = model(permuted)
    first_by_action = {a: float(v) for a, v in zip(original.candidate_actions, first)}
    second_by_action = {a: float(v) for a, v in zip(permuted.candidate_actions, second)}
    for (task, uav), value in first_by_action.items():
        assert abs(value - second_by_action[(task, 1 - uav)]) < 1e-6
