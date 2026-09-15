from gppo_world.minimal_scheduling import Instance, Subtask, UAV, graph_view, greedy_schedule, initial_state, legal_actions, solve_exact, wait_to_next_event


def make_fixture(*, eligible=((0, 1),) * 4, positions=((0, 0), (10, 0)), tasks=None):
    if tasks is None:
        tasks = (
            Subtask("a0", "a", 0, (0, 0), 2, eligible[0], ()),
            Subtask("a1", "a", 1, (0, 0), 1, eligible[1], ("a0",)),
            Subtask("b0", "b", 0, (10, 0), 2, eligible[2], ()),
            Subtask("b1", "b", 1, (10, 0), 1, eligible[3], ("b0",)),
        )
    return Instance("fixture", "test", tuple(UAV(i, positions[i]) for i in range(2)), tasks)


def test_single_task_flight_plus_processing_and_wait_release():
    task = (Subtask("a0", "a", 0, (3, 2), 4, (0,), ()), Subtask("a1", "a", 1, (3, 2), 1, (0,), ()), Subtask("b0", "b", 0, (9, 9), 1, (1,), ()), Subtask("b1", "b", 1, (9, 9), 1, (1,), ("b0",)))
    instance = make_fixture(tasks=task)
    state, assignment = __import__("gppo_world.minimal_scheduling", fromlist=["assign"]).assign(instance, initial_state(instance), ("a0", 0))
    assert assignment.finish_time == 9
    next_state, events = wait_to_next_event(instance, state)
    assert next_state.time == 9 and events == [{"task": "a0", "uav": 0, "time": 9}]
    assert ("a1", 0) in legal_actions(instance, next_state)


def test_dependencies_mask_ineligible_and_graph_consistency():
    instance = make_fixture(eligible=((0,), (0,), (1,), (1,)))
    state = initial_state(instance)
    assert legal_actions(instance, state) == (("a0", 0), ("b0", 1))
    view = graph_view(instance, state)
    assert view["legal_actions"] == [["a0", 0], ["b0", 1]]
    assert view["action_mask"]["a1|uav-0"] is False
    assert view["action_mask"]["b0|uav-0"] is False


def test_parallel_execution_and_non_overlap():
    instance = make_fixture()
    result = solve_exact(instance)
    assert result.status == "optimal"
    assert result.makespan == 3
    assert len(result.assignments) == 4
    by_uav = {0: [], 1: []}
    for item in result.assignments:
        by_uav[item.uav_id].append(item)
    for items in by_uav.values():
        assert all(a.finish_time <= b.start_time or b.finish_time <= a.start_time for i, a in enumerate(items) for b in items[i + 1:])


def test_exact_matches_hand_calculated_opportunity_cost_fixture():
    tasks = (
        Subtask("a0", "a", 0, (0, 0), 4, (1,), ()),
        Subtask("a1", "a", 1, (0, 0), 1, (1,), ("a0",)),
        Subtask("b0", "b", 0, (0, 0), 1, (0, 1), ()),
        Subtask("b1", "b", 1, (0, 0), 1, (0,), ("b0",)),
    )
    instance = Instance("opportunity-cost-demo", "mechanism_demo", (UAV(0, (3, 0)), UAV(1, (0, 0))), tasks)
    greedy = greedy_schedule(instance)
    exact = solve_exact(instance)
    assert greedy.makespan == 6
    assert exact.makespan == 5
    assert greedy.makespan > exact.makespan


def test_simultaneous_completion_is_deterministic_and_forced_first_is_covered():
    instance = make_fixture()
    result = solve_exact(instance)
    assert result.status == "optimal"
    assert result.waits[0]["time"] == result.waits[1]["time"] == 2
    forced = solve_exact(instance, forced_first=("a0", 0))
    assert forced.status == "optimal" and forced.assignments[0].task_id == "a0"
