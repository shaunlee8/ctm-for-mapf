"""A small, deterministic prioritized planner for nominal MAPF plans."""

from __future__ import annotations

from collections import deque

import numpy as np

from ctm_for_mapf.planners.reservation_table import GridPosition, ReservationTable


WAIT = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

MOVES: tuple[GridPosition, ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


def _position_after(position: GridPosition, action: int) -> GridPosition:
    dx, dy = MOVES[action]
    return position[0] + dx, position[1] + dy


def _in_bounds(obstacles: np.ndarray, position: GridPosition) -> bool:
    x, y = position
    return 0 <= x < obstacles.shape[0] and 0 <= y < obstacles.shape[1]


def _is_free(obstacles: np.ndarray, position: GridPosition) -> bool:
    return _in_bounds(obstacles, position) and obstacles[position] == 0


def _reconstruct_path(
    parents: dict[tuple[GridPosition, int], tuple[GridPosition, int] | None],
    goal_state: tuple[GridPosition, int],
) -> list[GridPosition]:
    path: list[GridPosition] = []
    state: tuple[GridPosition, int] | None = goal_state
    while state is not None:
        position, _ = state
        path.append(position)
        state = parents[state]
    return list(reversed(path))


def _plan_single_agent(
    obstacles: np.ndarray,
    start: GridPosition,
    goal: GridPosition,
    reservations: ReservationTable,
    max_time: int,
) -> list[GridPosition]:
    start_state = (start, 0)
    queue: deque[tuple[GridPosition, int]] = deque([start_state])
    parents: dict[tuple[GridPosition, int], tuple[GridPosition, int] | None] = {start_state: None}

    while queue:
        position, timestep = queue.popleft()
        if position == goal:
            return _reconstruct_path(parents, (position, timestep))
        if timestep >= max_time:
            continue

        next_timestep = timestep + 1
        for action in range(len(MOVES)):
            next_position = _position_after(position, action)
            next_state = (next_position, next_timestep)
            if next_state in parents:
                continue
            if not _is_free(obstacles, next_position):
                continue
            if reservations.is_vertex_reserved(next_position, next_timestep):
                continue
            if reservations.is_edge_conflict(position, next_position, next_timestep):
                continue
            parents[next_state] = (position, timestep)
            queue.append(next_state)

    raise RuntimeError(f"No prioritized path found from {start} to {goal} within horizon {max_time}.")


def positions_to_actions(path: list[GridPosition]) -> list[int]:
    """Convert a position sequence into POGEMA action ids."""
    if not path:
        raise ValueError("Cannot convert an empty path to actions.")

    reverse_moves = {move: idx for idx, move in enumerate(MOVES)}
    actions: list[int] = []
    for start, end in zip(path, path[1:]):
        delta = end[0] - start[0], end[1] - start[1]
        try:
            actions.append(reverse_moves[delta])
        except KeyError as exc:
            raise ValueError(f"Non-adjacent positions in path: {start} -> {end}") from exc
    return actions


def prioritized_plan(
    obstacles: np.ndarray,
    starts: list[GridPosition],
    goals: list[GridPosition],
    *,
    max_time: int | None = None,
) -> list[list[GridPosition]]:
    """Plan agents sequentially while reserving earlier agents' trajectories."""
    if len(starts) != len(goals):
        raise ValueError("starts and goals must have the same length.")
    if not starts:
        return []

    free_cells = int((obstacles == 0).sum())
    horizon = max_time or max(obstacles.size, free_cells * max(2, len(starts)))
    reservations = ReservationTable()
    paths: list[list[GridPosition]] = []

    for start, goal in zip(starts, goals):
        path = _plan_single_agent(obstacles, start, goal, reservations, horizon)
        reservations.reserve_path(path, hold_until=horizon)
        paths.append(path)

    return paths
