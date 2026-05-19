"""Recovery wrapper around a POGEMA environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pogema import GridConfig, pogema_v0

from ctm_for_mapf.envs.disturbances import DisturbanceModel, DisturbanceResult, ForcedWaitDelayModel
from ctm_for_mapf.envs.observations import RecoveryObservationBuilder
from ctm_for_mapf.envs.plan_tracking import Plan, PlanTracker
from ctm_for_mapf.planners import prioritized_plan


@dataclass(frozen=True, slots=True)
class RecoveryStepInfo:
    proposed_actions: tuple[int, ...]
    executed_actions: tuple[int, ...]
    forced_waits: tuple[bool, ...]
    actual_positions: tuple[tuple[int, int], ...]
    planned_positions: tuple[tuple[int, int], ...]
    lateness: tuple[int, ...]
    off_plan: tuple[bool, ...]
    injected_delay_count: int


class RecoveryEnv:
    """Add nominal plans, disturbances, and plan-relative observations to POGEMA."""

    def __init__(
        self,
        grid_config: GridConfig,
        *,
        delay_model: DisturbanceModel | None = None,
        seed: int | None = None,
    ) -> None:
        if grid_config.observation_type != "POMAPF":
            raise ValueError("RecoveryEnv expects GridConfig(observation_type='POMAPF').")
        self.grid_config = grid_config
        self.base_env = pogema_v0(grid_config)
        self.delay_model = delay_model or ForcedWaitDelayModel()
        self.observation_builder = RecoveryObservationBuilder()
        self.rng = np.random.default_rng(seed if seed is not None else grid_config.seed)

        self.plan: Plan | None = None
        self.tracker: PlanTracker | None = None
        self._last_forced_waits: tuple[bool, ...] = tuple(False for _ in range(grid_config.num_agents))
        self._injected_delay_count = 0

    @property
    def num_agents(self) -> int:
        return self.grid_config.num_agents

    def reset(self) -> tuple[list[dict], dict]:
        pogema_observations, _ = self.base_env.reset()
        obstacles = self.base_env.get_obstacles(ignore_borders=True)
        starts = [tuple(position) for position in self.base_env.get_agents_xy(ignore_borders=True)]
        goals = [tuple(position) for position in self.base_env.get_targets_xy(ignore_borders=True)]
        paths = prioritized_plan(obstacles, starts, goals)
        self.plan = Plan.from_paths(paths)
        self.tracker = PlanTracker(self.plan)
        self._last_forced_waits = tuple(False for _ in range(self.num_agents))
        self._injected_delay_count = 0

        observations = self._build_observations(pogema_observations)
        info = {
            "nominal_plan": self.plan,
            "nominal_makespan": self.plan.makespan,
            "positions": tuple(starts),
            "goals": tuple(goals),
        }
        return observations, info

    def step(self, proposed_actions: list[int] | tuple[int, ...]) -> tuple[list[dict], list[float], list[bool], list[bool], RecoveryStepInfo]:
        tracker = self._require_tracker()
        if len(proposed_actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(proposed_actions)}.")

        disturbance = self.delay_model.apply(proposed_actions, self.rng)
        pogema_observations, rewards, terminated, truncated, _ = self.base_env.step(list(disturbance.executed_actions))
        tracker.advance()
        self._last_forced_waits = disturbance.forced_waits
        self._injected_delay_count += sum(disturbance.forced_waits)

        observations = self._build_observations(pogema_observations)
        step_info = self._build_step_info(proposed_actions, disturbance)
        return observations, rewards, terminated, truncated, step_info

    def _build_observations(self, pogema_observations: list[dict]) -> list[dict]:
        tracker = self._require_tracker()
        actual_positions = [tuple(position) for position in self.base_env.get_agents_xy(ignore_borders=True)]
        return self.observation_builder.build(
            pogema_observations=pogema_observations,
            actual_positions=actual_positions,
            tracker=tracker,
            was_delayed_last_step=self._last_forced_waits,
        )

    def _build_step_info(
        self,
        proposed_actions: list[int] | tuple[int, ...],
        disturbance: DisturbanceResult,
    ) -> RecoveryStepInfo:
        tracker = self._require_tracker()
        actual_positions = tuple(tuple(position) for position in self.base_env.get_agents_xy(ignore_borders=True))
        planned_positions = tuple(tracker.scheduled_position(agent_id) for agent_id in range(self.num_agents))
        lateness = tuple(tracker.lateness(agent_id, position) for agent_id, position in enumerate(actual_positions))
        off_plan = tuple(tracker.off_plan(agent_id, position) for agent_id, position in enumerate(actual_positions))
        return RecoveryStepInfo(
            proposed_actions=tuple(int(action) for action in proposed_actions),
            executed_actions=disturbance.executed_actions,
            forced_waits=disturbance.forced_waits,
            actual_positions=actual_positions,
            planned_positions=planned_positions,
            lateness=lateness,
            off_plan=off_plan,
            injected_delay_count=self._injected_delay_count,
        )

    def _require_tracker(self) -> PlanTracker:
        if self.tracker is None:
            raise RuntimeError("RecoveryEnv must be reset before use.")
        return self.tracker
