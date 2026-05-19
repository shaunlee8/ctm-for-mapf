"""LSTM baseline policy for temporal recovery. Sanity check, not used."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ctm_for_mapf.models.recovery_policy import PolicyOutput, RecurrentState, RecurrentTemporalRecoveryPolicy


class LSTMTemporalRecoveryPolicy(RecurrentTemporalRecoveryPolicy):

    def __init__(self, *, input_dim: int, hidden_dim: int = 128, internal_iterations: int = 1) -> None:
        super().__init__()
        if internal_iterations < 1:
            raise ValueError("internal_iterations must be >= 1.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.internal_iterations = internal_iterations
        self.input_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, self.num_decisions)
        self.start_hidden_state = nn.Parameter(torch.empty(hidden_dim).uniform_(-math.sqrt(1 / hidden_dim), math.sqrt(1 / hidden_dim)))
        self.start_cell_state = nn.Parameter(torch.empty(hidden_dim).uniform_(-math.sqrt(1 / hidden_dim), math.sqrt(1 / hidden_dim)))

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> RecurrentState:
        hidden = self.start_hidden_state.unsqueeze(0).repeat(batch_size, 1)
        cell = self.start_cell_state.unsqueeze(0).repeat(batch_size, 1)
        if device is not None:
            hidden = hidden.to(device)
            cell = cell.to(device)
        return hidden, cell

    def forward(self, observations: torch.Tensor, state: RecurrentState) -> PolicyOutput:
        if observations.ndim != 2:
            raise ValueError(f"Expected observations with shape [batch, features], got {tuple(observations.shape)}.")
        features = self.input_projector(observations)
        hidden, cell = state
        for _ in range(self.internal_iterations):
            hidden, cell = self.cell(features, (hidden, cell))
        logits = self.action_head(hidden)
        return PolicyOutput(logits=logits, state=(hidden, cell), latent=hidden)
