"""Minimal domain objects for the MAPF recovery task.

A full environment API is a work in progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


GridPosition = tuple[int, int]


class Action(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    WAIT = "wait"


@dataclass(frozen=True, slots=True)
class AgentSpec:
    agent_id: int
    start: GridPosition
    goal: GridPosition


@dataclass(frozen=True, slots=True)
class RecoveryEvent:
    timestep: int
    agent_id: int
    reason: str
