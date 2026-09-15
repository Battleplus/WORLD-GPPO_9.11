"""Small exact scheduling benchmark used to test first-action opportunity cost.

This module is intentionally separate from the M-10 arrival environment.  It
implements only two UAVs, two length-two chains, precedence, eligibility,
non-preemption, and event-driven completion.  No communication, energy, or
learning code is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
import time
from typing import Iterable


MAX_TASKS = 4
MAX_UAVS = 2


@dataclass(frozen=True)
class Subtask:
    task_id: str
    chain_id: str
    index: int
    position: tuple[int, int]
    process_time: int
    eligible_uavs: tuple[int, ...]
    predecessors: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.process_time <= 0:
            raise ValueError("process_time must be positive")
        if not self.eligible_uavs:
            raise ValueError("each subtask needs an eligible UAV")
        if any(u not in range(MAX_UAVS) for u in self.eligible_uavs):
            raise ValueError("only UAV 0 and UAV 1 are supported")


@dataclass(frozen=True)
class UAV:
    uav_id: int
    position: tuple[int, int]


@dataclass(frozen=True)
class Instance:
    instance_id: str
    group: str
    uavs: tuple[UAV, ...]
    subtasks: tuple[Subtask, ...]

    def __post_init__(self) -> None:
        if len(self.uavs) != MAX_UAVS or len(self.subtasks) != MAX_TASKS:
            raise ValueError("the minimum benchmark is fixed at 2 UAVs and 4 subtasks")
        ids = {t.task_id for t in self.subtasks}
        if len(ids) != len(self.subtasks):
            raise ValueError("subtask IDs must be unique")
        for task in self.subtasks:
            if any(p not in ids for p in task.predecessors):
                raise ValueError(f"unknown predecessor for {task.task_id}")

    @property
    def by_id(self) -> dict[str, Subtask]:
        return {t.task_id: t for t in self.subtasks}


@dataclass(frozen=True)
class Assignment:
    task_id: str
    uav_id: int
    start_time: int
    finish_time: int
    flight_time: int
    process_time: int


@dataclass(frozen=True)
class Running:
    task_id: str
    finish_time: int
    start_time: int


@dataclass(frozen=True)
class State:
    time: int
    positions: tuple[tuple[int, int], ...]
    running: tuple[Running | None, ...]
    completed: tuple[str, ...]


@dataclass
class Schedule:
    status: str
    makespan: int | None
    assignments: list[Assignment] = field(default_factory=list)
    waits: list[dict[str, int]] = field(default_factory=list)
    expanded_states: int = 0
    elapsed_seconds: float = 0.0
    proof: str = ""


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def initial_state(instance: Instance) -> State:
    return State(
        time=0,
        positions=tuple(u.position for u in instance.uavs),
        running=(None, None),
        completed=(),
    )


def legal_actions(instance: Instance, state: State) -> tuple[tuple[str, int], ...]:
    done = set(state.completed)
    active = {r.task_id for r in state.running if r is not None}
    ready = [
        task for task in instance.subtasks
        if task.task_id not in done and task.task_id not in active
        and all(p in done for p in task.predecessors)
    ]
    actions: list[tuple[str, int]] = []
    for uav_id, running in enumerate(state.running):
        if running is not None:
            continue
        for task in ready:
            if uav_id in task.eligible_uavs:
                actions.append((task.task_id, uav_id))
    return tuple(sorted(actions, key=lambda x: (x[0], x[1])))


def assign(instance: Instance, state: State, action: tuple[str, int]) -> tuple[State, Assignment]:
    task_id, uav_id = action
    if action not in legal_actions(instance, state):
        raise ValueError(f"illegal action {action} at t={state.time}")
    task = instance.by_id[task_id]
    flight = manhattan(state.positions[uav_id], task.position)
    finish = state.time + flight + task.process_time
    running = list(state.running)
    running[uav_id] = Running(task_id=task_id, finish_time=finish, start_time=state.time)
    assignment = Assignment(task_id, uav_id, state.time, finish, flight, task.process_time)
    return State(state.time, state.positions, tuple(running), state.completed), assignment


def wait_to_next_event(instance: Instance, state: State) -> tuple[State, list[dict[str, int]]]:
    finishes = [r.finish_time for r in state.running if r is not None]
    if not finishes:
        raise ValueError("cannot wait without a running task")
    next_time = min(finishes)
    completed = set(state.completed)
    positions = list(state.positions)
    running = list(state.running)
    events: list[dict[str, int]] = []
    for uav_id, current in enumerate(running):
        if current is not None and current.finish_time == next_time:
            task = instance.by_id[current.task_id]
            completed.add(task.task_id)
            positions[uav_id] = task.position
            running[uav_id] = None
            events.append({"task": task.task_id, "uav": uav_id, "time": next_time})
    return State(next_time, tuple(positions), tuple(running), tuple(sorted(completed))), events


def _state_key(state: State) -> tuple[object, ...]:
    return (state.time, state.positions, state.running, state.completed)


def _assignment_from_transition(instance: Instance, state: State, action: tuple[str, int]) -> Assignment:
    task = instance.by_id[action[0]]
    flight = manhattan(state.positions[action[1]], task.position)
    return Assignment(action[0], action[1], state.time, state.time + flight + task.process_time, flight, task.process_time)


def solve_exact(instance: Instance, *, max_seconds: float = 30.0, forced_first: tuple[str, int] | None = None) -> Schedule:
    """Exhaustively enumerate assignments and event waits for this bounded instance.

    The search branches on every currently legal task-UAV assignment, keeps the
    time unchanged so multiple UAVs can be assigned concurrently, and also
    branches on waiting to the next completion event whenever work is running.
    Positive processing times make wait transitions strictly advance time.
    """
    if len(instance.subtasks) > MAX_TASKS or len(instance.uavs) > MAX_UAVS:
        raise ValueError("exact solver bound exceeded")
    started = time.monotonic()
    deadline = started + max_seconds
    best: Schedule | None = None
    seen: dict[tuple[object, ...], int] = {}
    expanded = 0

    def better(candidate: Schedule, incumbent: Schedule | None) -> bool:
        return incumbent is None or (candidate.makespan is not None and (incumbent.makespan is None or candidate.makespan < incumbent.makespan))

    def dfs(state: State, assignments: list[Assignment], waits: list[dict[str, int]]) -> None:
        nonlocal best, expanded
        if time.monotonic() >= deadline:
            raise TimeoutError
        expanded += 1
        if len(state.completed) == len(instance.subtasks) and all(r is None for r in state.running):
            candidate = Schedule("optimal", state.time, list(assignments), list(waits), expanded, 0.0, "exhaustive event-driven enumeration")
            if better(candidate, best):
                best = candidate
            return
        if best is not None and state.time >= (best.makespan or math.inf):
            return
        key = _state_key(state)
        prior = seen.get(key)
        if prior is not None and prior <= len(assignments):
            return
        seen[key] = len(assignments)

        for action in legal_actions(instance, state):
            next_state, _ = assign(instance, state, action)
            assignments.append(_assignment_from_transition(instance, state, action))
            dfs(next_state, assignments, waits)
            assignments.pop()
        if any(r is not None for r in state.running):
            next_state, events = wait_to_next_event(instance, state)
            waits.extend(events)
            dfs(next_state, assignments, waits)
            del waits[-len(events):]

    try:
        state = initial_state(instance)
        prefix: list[Assignment] = []
        if forced_first is not None:
            if forced_first not in legal_actions(instance, state):
                return Schedule("invalid_forced_action", None, elapsed_seconds=time.monotonic() - started, proof="forced action was not legal at t=0")
            state, _ = assign(instance, state, forced_first)
            prefix.append(_assignment_from_transition(instance, initial_state(instance), forced_first))
        dfs(state, prefix, [])
    except TimeoutError:
        elapsed = time.monotonic() - started
        if best is None:
            return Schedule("timeout_no_solution", None, expanded_states=expanded, elapsed_seconds=elapsed, proof="search timeout before proving a complete schedule")
        best.status = "timeout_feasible_not_proven"
        best.expanded_states = expanded
        best.elapsed_seconds = elapsed
        best.proof = "a feasible schedule was found, but exhaustive optimality was not proven before timeout"
        return best
    elapsed = time.monotonic() - started
    if best is None:
        return Schedule("infeasible", None, expanded_states=expanded, elapsed_seconds=elapsed, proof="all legal branches exhausted")
    best.expanded_states = expanded
    best.elapsed_seconds = elapsed
    return best


def greedy_schedule(instance: Instance) -> Schedule:
    """Frozen public rule: earliest predicted finish, task ID then UAV ID tie-break."""
    state = initial_state(instance)
    assignments: list[Assignment] = []
    waits: list[dict[str, int]] = []
    started = time.monotonic()
    while len(state.completed) < len(instance.subtasks):
        actions = legal_actions(instance, state)
        if actions:
            action = min(actions, key=lambda a: (state.time + manhattan(state.positions[a[1]], instance.by_id[a[0]].position) + instance.by_id[a[0]].process_time, a[0], a[1]))
            state, _ = assign(instance, state, action)
            assignments.append(_assignment_from_transition(instance, state=State(state.time, state.positions, state.running, state.completed), action=action))
            # The helper above must use pre-assignment state; reconstruct the
            # exact record from the assignment just appended if needed.
            assignments[-1] = Assignment(action[0], action[1], assignments[-1].start_time, assignments[-1].finish_time, assignments[-1].flight_time, assignments[-1].process_time)
            continue
        if not any(r is not None for r in state.running):
            return Schedule("infeasible", None, assignments, waits, elapsed_seconds=time.monotonic() - started, proof="no legal assignment and no completion event")
        state, events = wait_to_next_event(instance, state)
        waits.extend(events)
    return Schedule("feasible", state.time, assignments, waits, elapsed_seconds=time.monotonic() - started, proof="frozen earliest-finish public rule")


def graph_view(instance: Instance, state: State) -> dict[str, object]:
    """Return the minimal heterogeneous graph and action mask for auditing."""
    done = set(state.completed)
    active = {r.task_id for r in state.running if r is not None}
    nodes: list[dict[str, object]] = []
    for uav_id, uav in enumerate(instance.uavs):
        nodes.append({"type": "uav", "id": f"uav-{uav_id}", "position": list(state.positions[uav_id]), "busy": state.running[uav_id] is not None})
    for task in instance.subtasks:
        nodes.append({"type": "subtask", "id": task.task_id, "chain_id": task.chain_id, "index": task.index, "position": list(task.position), "process_time": task.process_time, "completed": task.task_id in done, "running": task.task_id in active})
    precedence = [{"from": p, "to": task.task_id, "kind": "precedence"} for task in instance.subtasks for p in task.predecessors]
    capability = [{"from": f"uav-{uav}", "to": task.task_id, "kind": "capability", "flight_time": manhattan(state.positions[uav], task.position), "process_time": task.process_time} for task in instance.subtasks for uav in task.eligible_uavs]
    actions = legal_actions(instance, state)
    return {"nodes": nodes, "edges": precedence + capability, "legal_actions": [list(a) for a in actions], "action_mask": {f"{task}|uav-{uav}": [task, uav] in actions for task in [t.task_id for t in instance.subtasks] for uav in range(MAX_UAVS)}}


def instance_to_dict(instance: Instance) -> dict[str, object]:
    return {"instance_id": instance.instance_id, "group": instance.group, "uavs": [{"uav_id": u.uav_id, "position": list(u.position)} for u in instance.uavs], "subtasks": [{"task_id": t.task_id, "chain_id": t.chain_id, "index": t.index, "position": list(t.position), "process_time": t.process_time, "eligible_uavs": list(t.eligible_uavs), "predecessors": list(t.predecessors)} for t in instance.subtasks]}

