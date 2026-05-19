"""Convert structured recovery observations into model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class RecoveryObservationVectorizer:
    """Flatten current recovery observations into deterministic feature vectors."""

    include_scheduled_next_action: bool = True

    def vectorize(self, observation: dict) -> np.ndarray:
        # Flat order is fixed and intentionally boring.
        parts = [
            np.asarray(observation["obstacles"], dtype=np.float32).reshape(-1),
            np.asarray(observation["agents"], dtype=np.float32).reshape(-1),
            np.asarray(observation["xy"], dtype=np.float32).reshape(-1),
            np.asarray(observation["target_xy"], dtype=np.float32).reshape(-1),
            np.asarray(observation["planned_next_action"], dtype=np.float32).reshape(-1),
            np.asarray(observation["planned_next_delta"], dtype=np.float32).reshape(-1),
        ]
        if self.include_scheduled_next_action:
            parts.append(np.asarray([observation["scheduled_next_action"]], dtype=np.float32))
        parts.extend(
            [
                np.asarray([observation["lateness"]], dtype=np.float32),
                np.asarray([observation["off_plan"]], dtype=np.float32),
                np.asarray([observation["remaining_plan_length"]], dtype=np.float32),
                np.asarray([observation["was_delayed_last_step"]], dtype=np.float32),
            ]
        )
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def batch_numpy(self, observations: list[dict]) -> np.ndarray:
        if not observations:
            raise ValueError("Cannot vectorize an empty observation batch.")
        return np.stack([self.vectorize(observation) for observation in observations], axis=0)

    def batch_tensor(self, observations: list[dict], *, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.as_tensor(self.batch_numpy(observations), dtype=torch.float32, device=device)


@dataclass(frozen=True, slots=True)
class RecoveryObservationTokenizer:
    """Convert recovery observations into deterministic token sequences.

    Token feature layout:
    ``[obstacle, agent, rel_row, rel_col, is_grid, is_aux, aux_id, value]``
    Learned embeddings and global map state not supported. Need to add more
    stuff here for a stronger model.
    """

    include_scheduled_next_action: bool = True

    @property
    def token_dim(self) -> int:
        return 8

    def tokenize(self, observation: dict) -> np.ndarray:
        obstacles = np.asarray(observation["obstacles"], dtype=np.float32)
        agents = np.asarray(observation["agents"], dtype=np.float32)
        if obstacles.shape != agents.shape:
            raise ValueError(f"obstacles and agents must have matching shape, got {obstacles.shape} and {agents.shape}.")
        if obstacles.ndim != 2:
            raise ValueError(f"Expected 2D local patches, got obstacle shape {obstacles.shape}.")

        height, width = obstacles.shape
        center_row = (height - 1) / 2.0
        center_col = (width - 1) / 2.0
        row_scale = max(center_row, 1.0)
        col_scale = max(center_col, 1.0)
        tokens: list[np.ndarray] = []

        # Each local grid cell becomes one token with normalized relative position.
        # This lets CTM action synchronization learn to attend to nearby blockers,
        # free cells, or other agents instead of receiving the patch as an opaque vector.
        for row in range(height):
            for col in range(width):
                tokens.append(
                    np.asarray(
                        [
                            obstacles[row, col],
                            agents[row, col],
                            (row - center_row) / row_scale,
                            (col - center_col) / col_scale,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        dtype=np.float32,
                    )
                )

        # Non-spatial recovery context is represented as scalar auxiliary tokens.
        # The aux_id coordinate gives the model a stable type indicator while the
        # value slot carries the actual plan / delay quantity.
        aux_values = self._auxiliary_values(observation)
        denom = max(len(aux_values) - 1, 1)
        for aux_idx, value in enumerate(aux_values):
            tokens.append(
                np.asarray(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        aux_idx / denom,
                        float(value),
                    ],
                    dtype=np.float32,
                )
            )

        return np.stack(tokens, axis=0).astype(np.float32, copy=False)

    def _auxiliary_values(self, observation: dict) -> list[float]:
        values: list[float] = []
        values.extend(np.asarray(observation["xy"], dtype=np.float32).reshape(-1).tolist())
        values.extend(np.asarray(observation["target_xy"], dtype=np.float32).reshape(-1).tolist())
        values.extend(np.asarray(observation["planned_next_action"], dtype=np.float32).reshape(-1).tolist())
        values.extend(np.asarray(observation["planned_next_delta"], dtype=np.float32).reshape(-1).tolist())
        if self.include_scheduled_next_action:
            values.append(float(observation["scheduled_next_action"]))
        values.extend(
            [
                float(observation["lateness"]),
                float(observation["off_plan"]),
                float(observation["remaining_plan_length"]),
                float(observation["was_delayed_last_step"]),
            ]
        )
        return values

    def batch_numpy(self, observations: list[dict]) -> np.ndarray:
        if not observations:
            raise ValueError("Cannot tokenize an empty observation batch.")
        tokenized = [self.tokenize(observation) for observation in observations]
        first_shape = tokenized[0].shape
        if any(tokens.shape != first_shape for tokens in tokenized):
            raise ValueError("All observations in a token batch must produce the same token shape.")
        return np.stack(tokenized, axis=0)

    def batch_tensor(self, observations: list[dict], *, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.as_tensor(self.batch_numpy(observations), dtype=torch.float32, device=device)
