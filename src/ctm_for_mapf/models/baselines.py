"""Simple non-neural policies and experts for recovery rollouts."""

from __future__ import annotations

from dataclasses import dataclass

from ctm_for_mapf.envs.plan_tracking import PlanTracker
from ctm_for_mapf.planners import MOVES, temporal_repair_plan


# POGEMA uses action 0 as physical WAIT. The learned recovery policy has a
# smaller decision space: WAIT, or follow the next nominal-plan action selected
# by the plan tracker.
WAIT = 0
WAIT_DECISION = 0
FOLLOW_PLAN_DECISION = 1


class FollowPlanPolicy:
    """Follow the next waypoint on each agent's nominal path."""

    def act(
        self,
        tracker: PlanTracker,
        actual_positions: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    ) -> list[int]:
        return [
            tracker.path_next_action(agent_id, tuple(actual_position))
            for agent_id, actual_position in enumerate(actual_positions)
        ]


@dataclass(frozen=True, slots=True)
class TemporalExpertOutput:
    decisions: tuple[int, ...]
    actions: tuple[int, ...]


class SafeTemporalRecoveryExpert:
    """Use global state to decide whether agents should follow their path or wait."""

    def act(
        self,
        tracker: PlanTracker,
        actual_positions: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    ) -> TemporalExpertOutput:
        positions = tuple(tuple(position) for position in actual_positions)
        proposed_actions = [
            tracker.path_next_action(agent_id, position)
            for agent_id, position in enumerate(positions)
        ]
        proposed_targets = [self._target(position, action) for position, action in zip(positions, proposed_actions)]

        # Start from agents that have no unfinished path progress.
        should_wait = [action == WAIT for action in proposed_actions]

        # Same-target conflicts: if multiple moving agents want one cell, all wait.
        target_to_agents: dict[tuple[int, int], list[int]] = {}
        for agent_id, (position, target, action) in enumerate(zip(positions, proposed_targets, proposed_actions)):
            if action != WAIT and target != position:
                target_to_agents.setdefault(target, []).append(agent_id)
        for agent_ids in target_to_agents.values():
            if len(agent_ids) > 1:
                for agent_id in agent_ids:
                    should_wait[agent_id] = True

        # Edge swaps: both agents wait.
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                if proposed_targets[left] == positions[right] and proposed_targets[right] == positions[left]:
                    if proposed_targets[left] != positions[left] or proposed_targets[right] != positions[right]:
                        should_wait[left] = True
                        should_wait[right] = True

        # Moving into a stationary agent is unsafe if that agent will remain in place.
        occupied_by = {position: agent_id for agent_id, position in enumerate(positions)}
        changed = True
        while changed:
            changed = False
            for agent_id, (target, position) in enumerate(zip(proposed_targets, positions)):
                if should_wait[agent_id] or target == position:
                    continue
                blocker = occupied_by.get(target)
                if blocker is not None and should_wait[blocker]:
                    should_wait[agent_id] = True
                    changed = True

        actions = tuple(WAIT if should_wait[agent_id] else proposed_actions[agent_id] for agent_id in range(len(positions)))
        decisions = tuple(WAIT_DECISION if action == WAIT else FOLLOW_PLAN_DECISION for action in actions)
        return TemporalExpertOutput(decisions=decisions, actions=actions)

    @staticmethod
    def _target(position: tuple[int, int], action: int) -> tuple[int, int]:
        dx, dy = MOVES[action]
        return position[0] + dx, position[1] + dy


class CentralizedTemporalRepairExpert:
    """Globally repair timing while keeping every agent on its nominal path.

    This expert searches for a complete collision-free WAIT/FOLLOW schedule and emits the 
    first joint decision of that schedule. A stronger imitation teacher.
    """

    def __init__(self, *, max_expansions: int = 100_000, fallback_to_safe: bool = True) -> None:
        self.max_expansions = max_expansions
        self.fallback_to_safe = fallback_to_safe
        self._safe_fallback = SafeTemporalRecoveryExpert()

    def act(
        self,
        tracker: PlanTracker,
        actual_positions: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    ) -> TemporalExpertOutput:
        positions = tuple(tuple(position) for position in actual_positions)
        # Repair search operates in path-index space.
        start_indices = tuple(
            tracker.plan_index_for_position(agent_id, position)
            for agent_id, position in enumerate(positions)
        )
        try:
            repair = temporal_repair_plan(tracker.plan, start_indices, max_expansions=self.max_expansions)
        except RuntimeError:
            if not self.fallback_to_safe:
                raise
            return self._safe_fallback.act(tracker, positions)

        if not repair.decisions:
            decisions = tuple(WAIT_DECISION for _ in positions)
        else:
            decisions = repair.first_decision
        actions = tuple(
            tracker.path_next_action(agent_id, position) if decision == FOLLOW_PLAN_DECISION else WAIT
            for agent_id, (position, decision) in enumerate(zip(positions, decisions))
        )
        return TemporalExpertOutput(decisions=decisions, actions=actions)
