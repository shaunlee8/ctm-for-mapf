"""Recovery-specific observation augmentation on top of POGEMA observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ctm_for_mapf.envs.plan_tracking import PlanTracker


@dataclass(slots=True)
class RecoveryObservationBuilder:
    """Attach plan-relative execution features to per-agent POGEMA observations."""

    num_actions: int = 5

    def build(
        self,
        pogema_observations: list[dict],
        actual_positions: list[tuple[int, int]],
        tracker: PlanTracker,
        was_delayed_last_step: tuple[bool, ...],
    ) -> list[dict]:
        observations: list[dict] = []
        for agent_id, (pogema_obs, actual_position) in enumerate(zip(pogema_observations, actual_positions)):
            path_action = tracker.path_next_action(agent_id, actual_position)
            path_next_position = tracker.path_next_position(agent_id, actual_position)
            path_delta = (
                path_next_position[0] - actual_position[0],
                path_next_position[1] - actual_position[1],
            )
            action_one_hot = np.zeros(self.num_actions, dtype=np.float32)
            action_one_hot[path_action] = 1.0

            observations.append(
                {
                    **pogema_obs,
                    "planned_next_action": action_one_hot,
                    "planned_next_delta": np.asarray(path_delta, dtype=np.float32),
                    "scheduled_next_action": np.float32(tracker.scheduled_next_action(agent_id)),
                    "lateness": np.float32(tracker.lateness(agent_id, actual_position)),
                    "off_plan": np.float32(tracker.off_plan(agent_id, actual_position)),
                    "remaining_plan_length": np.float32(tracker.remaining_plan_length(agent_id, actual_position)),
                    "was_delayed_last_step": np.float32(was_delayed_last_step[agent_id]),
                }
            )
        return observations
