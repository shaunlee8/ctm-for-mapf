"""Global temporal repair over fixed nominal paths."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ctm_for_mapf.envs.plan_tracking import Plan

GridPosition = tuple[int, int]


WAIT_DECISION = 0
FOLLOW_PLAN_DECISION = 1
JointIndexState = tuple[int, ...]
JointDecision = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TemporalRepairPlan:
    """A collision-free temporal schedule over fixed nominal paths."""

    decisions: tuple[JointDecision, ...]
    expanded_states: int

    @property
    def first_decision(self) -> JointDecision:
        if not self.decisions:
            return tuple()
        return self.decisions[0]


def temporal_repair_plan(
    plan: Plan,
    start_indices: JointIndexState,
    *,
    max_expansions: int = 100_000,
) -> TemporalRepairPlan:
    """Find a shortest safe WAIT/FOLLOW schedule from current path indices.

    The spatial paths are fixed. Each transition either keeps an agent at its current
    index or advances it by one waypoint, while enforcing vertex-conflict and edge-swap
    safety. A* minimizes remaining makespan; action ordering prefers progress when equal
    makespan schedules exist.
    """
    goal_indices = tuple(len(path) - 1 for path in plan.positions)
    if len(start_indices) != plan.num_agents:
        raise ValueError("start_indices must have one entry per agent.")
    if any(index < 0 or index > goal for index, goal in zip(start_indices, goal_indices)):
        raise ValueError("start_indices must lie within each nominal path.")
    if start_indices == goal_indices:
        return TemporalRepairPlan(decisions=tuple(), expanded_states=0)

    def heuristic(state: JointIndexState) -> int:
        return max((goal - index for index, goal in zip(state, goal_indices)), default=0)

    frontier: list[tuple[int, int, int, JointIndexState]] = []
    counter = 0
    heappush(frontier, (heuristic(start_indices), 0, counter, start_indices))
    costs: dict[JointIndexState, int] = {start_indices: 0}
    parents: dict[JointIndexState, tuple[JointIndexState, JointDecision] | None] = {start_indices: None}
    expanded = 0

    while frontier:
        _, cost, _, state = heappop(frontier)
        if cost != costs[state]:
            continue
        if state == goal_indices:
            return TemporalRepairPlan(decisions=_reconstruct_decisions(parents, state), expanded_states=expanded)
        expanded += 1
        if expanded > max_expansions:
            raise RuntimeError(f"Temporal repair exceeded max_expansions={max_expansions}.")

        for decision in _candidate_decisions(state, goal_indices):
            next_state = tuple(
                index + 1 if choice == FOLLOW_PLAN_DECISION and index < goal else index
                for index, goal, choice in zip(state, goal_indices, decision)
            )
            if not _is_safe_transition(plan, state, next_state):
                continue
            next_cost = cost + 1
            if next_cost >= costs.get(next_state, 10**12):
                continue
            costs[next_state] = next_cost
            parents[next_state] = (state, decision)
            counter += 1
            heappush(frontier, (next_cost + heuristic(next_state), next_cost, counter, next_state))

    raise RuntimeError("No collision-free temporal repair schedule exists on the nominal paths.")


def _candidate_decisions(state: JointIndexState, goal_indices: JointIndexState) -> list[JointDecision]:
    choices = [
        (WAIT_DECISION,) if index >= goal else (FOLLOW_PLAN_DECISION, WAIT_DECISION)
        for index, goal in zip(state, goal_indices)
    ]
    decisions = [tuple(choice) for choice in product(*choices)]
    # Prefer schedules that make progress when multiple shortest plans exist.
    decisions.sort(key=lambda item: (-sum(item), item))
    return decisions


def _position(plan: Plan, agent_id: int, index: int) -> GridPosition:
    return plan.positions[agent_id][index]


def _is_safe_transition(plan: Plan, state: JointIndexState, next_state: JointIndexState) -> bool:
    current_positions = tuple(_position(plan, agent_id, index) for agent_id, index in enumerate(state))
    next_positions = tuple(_position(plan, agent_id, index) for agent_id, index in enumerate(next_state))
    if len(set(next_positions)) != len(next_positions):
        return False
    for left in range(len(state)):
        for right in range(left + 1, len(state)):
            if current_positions[left] == next_positions[right] and current_positions[right] == next_positions[left]:
                if current_positions[left] != next_positions[left] or current_positions[right] != next_positions[right]:
                    return False
    return True


def _reconstruct_decisions(
    parents: dict[JointIndexState, tuple[JointIndexState, JointDecision] | None],
    goal_state: JointIndexState,
) -> tuple[JointDecision, ...]:
    decisions: list[JointDecision] = []
    state = goal_state
    while parents[state] is not None:
        previous, decision = parents[state]
        decisions.append(decision)
        state = previous
    return tuple(reversed(decisions))
