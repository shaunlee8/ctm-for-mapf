"""Plan containers and time-indexed progress tracking."""

from __future__ import annotations

from dataclasses import dataclass

from ctm_for_mapf.planners import positions_to_actions


GridPosition = tuple[int, int]
WAIT = 0


@dataclass(frozen=True, slots=True)
class Plan:
    """Nominal paths and actions for each agent."""

    positions: tuple[tuple[GridPosition, ...], ...]
    actions: tuple[tuple[int, ...], ...]

    @classmethod
    def from_paths(cls, paths: list[list[GridPosition]]) -> "Plan":
        return cls(
            positions=tuple(tuple(path) for path in paths),
            actions=tuple(tuple(positions_to_actions(path)) for path in paths),
        )

    @property
    def num_agents(self) -> int:
        return len(self.positions)

    @property
    def makespan(self) -> int:
        return max((len(path) - 1 for path in self.positions), default=0)

    def position_at(self, agent_id: int, timestep: int) -> GridPosition:
        path = self.positions[agent_id]
        return path[min(timestep, len(path) - 1)]

    def action_at(self, agent_id: int, timestep: int) -> int:
        actions = self.actions[agent_id]
        if timestep >= len(actions):
            return WAIT
        return actions[timestep]


@dataclass(slots=True)
class PlanTracker:
    """Track actual execution against a nominal time-indexed plan."""

    plan: Plan
    timestep: int = 0

    # Schedule-indexed accessors for lateness / deviation metrics.
    def scheduled_position(self, agent_id: int) -> GridPosition:
        return self.plan.position_at(agent_id, self.timestep)

    def scheduled_next_position(self, agent_id: int) -> GridPosition:
        return self.plan.position_at(agent_id, self.timestep + 1)

    def scheduled_next_action(self, agent_id: int) -> int:
        return self.plan.action_at(agent_id, self.timestep)

    # Path-progress accessors for actually following the nominal path after delays.
    def plan_index_for_position(self, agent_id: int, actual_position: GridPosition) -> int:
        path = self.plan.positions[agent_id]
        eligible_indices = [idx for idx, position in enumerate(path) if position == actual_position and idx <= self.timestep]
        return max(eligible_indices, default=0)

    def path_next_position(self, agent_id: int, actual_position: GridPosition) -> GridPosition:
        path = self.plan.positions[agent_id]
        index = self.plan_index_for_position(agent_id, actual_position)
        return path[min(index + 1, len(path) - 1)]

    def path_next_action(self, agent_id: int, actual_position: GridPosition) -> int:
        actions = self.plan.actions[agent_id]
        index = self.plan_index_for_position(agent_id, actual_position)
        if index >= len(actions):
            return WAIT
        return actions[index]

    def lateness(self, agent_id: int, actual_position: GridPosition) -> int:
        return max(0, self.timestep - self.plan_index_for_position(agent_id, actual_position))

    def off_plan(self, agent_id: int, actual_position: GridPosition) -> bool:
        return actual_position != self.scheduled_position(agent_id)

    def remaining_plan_length(self, agent_id: int, actual_position: GridPosition) -> int:
        index = self.plan_index_for_position(agent_id, actual_position)
        return max(0, len(self.plan.positions[agent_id]) - 1 - index)

    def advance(self) -> None:
        self.timestep += 1
