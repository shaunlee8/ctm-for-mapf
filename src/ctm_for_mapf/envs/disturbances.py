"""Execution-time disturbances for the recovery environment. Mess up MAPF here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

WAIT = 0

@dataclass(frozen=True, slots=True)
class DisturbanceResult:
    executed_actions: tuple[int, ...]
    forced_waits: tuple[bool, ...]


class DisturbanceModel(Protocol):
    """Interface implemented by execution-time disturbance models."""

    def apply(self, actions: list[int] | tuple[int, ...], rng: np.random.Generator) -> DisturbanceResult:
        raise NotImplementedError


@dataclass(slots=True)
class ForcedWaitDelayModel:
    """Independently override proposed actions with WAIT."""

    delay_probability: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.delay_probability <= 1.0:
            raise ValueError("delay_probability must lie in [0, 1].")

    def apply(self, actions: list[int] | tuple[int, ...], rng: np.random.Generator) -> DisturbanceResult:
        forced_waits = tuple(bool(value) for value in rng.random(len(actions)) < self.delay_probability)
        executed_actions = tuple(WAIT if forced else int(action) for action, forced in zip(actions, forced_waits))
        return DisturbanceResult(executed_actions=executed_actions, forced_waits=forced_waits)


@dataclass(slots=True)
class ScheduledForcedWaitDelayModel:
    """Force specific agents to wait at specified environment timesteps."""

    schedule: dict[int, tuple[int, ...]]
    timestep: int = 0

    def apply(self, actions: list[int] | tuple[int, ...], rng: np.random.Generator) -> DisturbanceResult:
        forced_ids = set(self.schedule.get(self.timestep, tuple()))
        if any(agent_id < 0 or agent_id >= len(actions) for agent_id in forced_ids):
            raise ValueError("scheduled agent id is outside the action batch.")
        forced_waits = tuple(agent_id in forced_ids for agent_id in range(len(actions)))
        self.timestep += 1
        executed_actions = tuple(WAIT if forced else int(action) for action, forced in zip(actions, forced_waits))
        return DisturbanceResult(executed_actions=executed_actions, forced_waits=forced_waits)


@dataclass(slots=True)
class BurstDelayModel:
    """Force one or more agents to wait for a contiguous burst of timesteps."""

    agent_ids: tuple[int, ...]
    start_timestep: int
    duration: int
    timestep: int = 0

    def __post_init__(self) -> None:
        if self.start_timestep < 0:
            raise ValueError("start_timestep must be >= 0.")
        if self.duration < 1:
            raise ValueError("duration must be >= 1.")

    def apply(self, actions: list[int] | tuple[int, ...], rng: np.random.Generator) -> DisturbanceResult:
        active = self.start_timestep <= self.timestep < self.start_timestep + self.duration
        if any(agent_id < 0 or agent_id >= len(actions) for agent_id in self.agent_ids):
            raise ValueError("burst-delay agent id is outside the action batch.")
        forced_ids = set(self.agent_ids) if active else set()
        forced_waits = tuple(agent_id in forced_ids for agent_id in range(len(actions)))
        self.timestep += 1
        executed_actions = tuple(WAIT if forced else int(action) for action, forced in zip(actions, forced_waits))
        return DisturbanceResult(executed_actions=executed_actions, forced_waits=forced_waits)
