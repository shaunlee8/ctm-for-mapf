"""Shared interfaces and helpers for recurrent temporal-recovery policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn as nn

from ctm_for_mapf.models.communication import CommunicationState

# For compatibility.
RecurrentState: TypeAlias = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    logits: torch.Tensor
    state: RecurrentState
    latent: torch.Tensor


@dataclass(frozen=True, slots=True)
class CommunicatingPolicyOutput(PolicyOutput):
    """Joint-agent CTM output with synchronization-derived communication."""

    outgoing_messages: torch.Tensor
    received_messages: torch.Tensor
    next_communication_state: "CommunicationState"
    neighbor_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class AdaptiveCTMPolicyOutput(PolicyOutput):
    """Policy output that preserves the CTM's internal thought trajectory."""

    all_logits: torch.Tensor
    all_latents: torch.Tensor
    certainties: torch.Tensor


@dataclass(frozen=True, slots=True)
class AttentionCTMPolicyOutput(AdaptiveCTMPolicyOutput):
    """Attention-enabled CTM output for action synchronization and attention."""

    action_latents: torch.Tensor
    attention_weights: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class UnifiedCTMPolicyOutput(AttentionCTMPolicyOutput):
    """Joint unified CTM output with attention, adaptive-ready ticks, and messages. The real deal."""

    outgoing_messages: torch.Tensor | None = None
    received_messages: torch.Tensor | None = None
    next_communication_state: CommunicationState | None = None
    neighbor_mask: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class UnifiedAdaptiveActionOutput:
    """Joint adaptive inference output for the unified CTM policy."""

    decisions: torch.Tensor
    state: RecurrentState
    next_communication_state: CommunicationState
    ticks_used: torch.Tensor
    selected_logits: torch.Tensor
    selected_latents: torch.Tensor
    selected_certainties: torch.Tensor
    outgoing_messages: torch.Tensor
    received_messages: torch.Tensor
    neighbor_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class AdaptiveActionOutput:
    """Inference-time decisions plus actual internal compute used per agent."""

    decisions: torch.Tensor
    state: RecurrentState
    ticks_used: torch.Tensor
    selected_logits: torch.Tensor
    selected_latents: torch.Tensor
    selected_certainties: torch.Tensor


class RecurrentTemporalRecoveryPolicy(nn.Module, ABC):
    """Common online contract for policies."""

    num_decisions: int = 2

    @abstractmethod
    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> RecurrentState:
        raise NotImplementedError

    @abstractmethod
    def forward(self, observations: torch.Tensor, state: RecurrentState) -> PolicyOutput:
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations: torch.Tensor, state: RecurrentState) -> tuple[torch.Tensor, RecurrentState]:
        output = self(observations, state)
        return output.logits.argmax(dim=-1), output.state
