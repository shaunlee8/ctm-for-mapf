"""Time-indexed reservations for simple prioritized MAPF planning."""

from __future__ import annotations

from dataclasses import dataclass, field


GridPosition = tuple[int, int]


@dataclass(slots=True)
class ReservationTable:
    """Reserve occupied vertices and directed edges over discrete timesteps."""

    vertex_reservations: dict[int, set[GridPosition]] = field(default_factory=dict)
    edge_reservations: dict[int, set[tuple[GridPosition, GridPosition]]] = field(default_factory=dict)

    def reserve_path(self, path: list[GridPosition], hold_until: int) -> None:
        """Reserve a path and keep the final cell occupied through ``hold_until``."""
        if not path:
            raise ValueError("Cannot reserve an empty path.")

        for timestep, position in enumerate(path):
            self.vertex_reservations.setdefault(timestep, set()).add(position)
            if timestep > 0:
                prev_position = path[timestep - 1]
                self.edge_reservations.setdefault(timestep, set()).add((prev_position, position))

        goal = path[-1]
        for timestep in range(len(path), hold_until + 1):
            self.vertex_reservations.setdefault(timestep, set()).add(goal)
            self.edge_reservations.setdefault(timestep, set()).add((goal, goal))

    def is_vertex_reserved(self, position: GridPosition, timestep: int) -> bool:
        return position in self.vertex_reservations.get(timestep, set())

    def is_edge_conflict(self, start: GridPosition, end: GridPosition, timestep: int) -> bool:
        """Return whether moving ``start -> end`` would swap with a reserved edge."""
        return (end, start) in self.edge_reservations.get(timestep, set())
