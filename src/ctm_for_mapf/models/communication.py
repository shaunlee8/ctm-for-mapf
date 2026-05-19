"""Decentralized message-passing helpers for multi-agent recovery policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CommunicationState:
    """Messages already delivered and ready to be eaten this environment step.

    Agents cannot use instantaneous global information from other agents while choosing the current action.
    """

    received_messages: torch.Tensor

    def to(self, device: torch.device | str) -> "CommunicationState":
        return CommunicationState(received_messages=self.received_messages.to(device))


def radius_neighbor_mask(
    positions: torch.Tensor,
    *,
    radius: int,
    agent_mask: torch.Tensor | None = None,
    include_self: bool = False,
) -> torch.Tensor:
    """Return a decentralized Manhattan-radius communication graph.

    The returned tensor has shape ``[batch, receivers, senders]``. Entry ``[b, i, j]``
    is true when sender ``j`` can send to receiver ``i`` in batch item ``b``.
    """
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"Expected positions with shape [batch, agents, 2], got {tuple(positions.shape)}.")
    if radius < 0:
        raise ValueError("radius must be >= 0.")

    # Use Manhattan distance since movement is grid-based and four-neighbor.
    pairwise_manhattan = (positions[:, :, None, :] - positions[:, None, :, :]).abs().sum(dim=-1)
    neighbors = pairwise_manhattan <= radius

    num_agents = positions.shape[1]
    if not include_self:
        eye = torch.eye(num_agents, dtype=torch.bool, device=positions.device).unsqueeze(0)
        neighbors = neighbors & ~eye

    if agent_mask is not None:
        if agent_mask.shape != positions.shape[:2]:
            raise ValueError(
                f"Expected agent_mask with shape {tuple(positions.shape[:2])}, got {tuple(agent_mask.shape)}."
            )
        valid_receivers = agent_mask[:, :, None]
        valid_senders = agent_mask[:, None, :]
        neighbors = neighbors & valid_receivers & valid_senders

    return neighbors


def mean_pool_neighbor_messages(messages: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool sender messages into one received vector per agent.

    ``messages`` has shape ``[batch, senders, message_dim]`` and ``neighbor_mask`` has
    shape ``[batch, receivers, senders]``. Agents with no neighbors receive zeros.
    """
    if messages.ndim != 3:
        raise ValueError(f"Expected messages with shape [batch, agents, message_dim], got {tuple(messages.shape)}.")
    expected_mask_shape = (messages.shape[0], messages.shape[1], messages.shape[1])
    if neighbor_mask.shape != expected_mask_shape:
        raise ValueError(f"Expected neighbor_mask with shape {expected_mask_shape}, got {tuple(neighbor_mask.shape)}.")

    weights = neighbor_mask.to(dtype=messages.dtype)
    summed = torch.einsum("brs,bsd->brd", weights, messages)
    counts = weights.sum(dim=-1, keepdim=True)
    return summed / counts.clamp_min(1.0)
