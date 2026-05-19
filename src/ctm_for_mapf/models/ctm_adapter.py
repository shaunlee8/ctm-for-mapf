"""
Copyright (c) 2026 Michael Mulder, Shaun Christopher Lee.

Compatibility layer for CTM-based MAPF recovery.

Third-party dependencies used by this module:
- Continuous Thought Machines, SakanaAI, Apache License 2.0:
  https://github.com/SakanaAI/continuous-thought-machines
- PyTorch, PyTorch Foundation / contributors:
  https://pytorch.org/
- NumPy, NumPy developers:
  https://numpy.org/

The official CTM repository is kept as a git submodule rather than copied into
this package. This module imports and adapts the relevant CTM classes for MAPF
recovery experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch


def _ctm_repo_root() -> Path:
    return Path(__file__).resolve().parents[3] / "third_party" / "continuous-thought-machines"


def ensure_ctm_on_path() -> Path:
    repo_root = _ctm_repo_root()
    if not repo_root.exists():
        raise FileNotFoundError(
            "CTM submodule not found. Run `git submodule update --init --recursive` first."
        )
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


ensure_ctm_on_path()

from models.ctm import ContinuousThoughtMachine
from models.ctm_rl import ContinuousThoughtMachineRL


HiddenState = tuple[torch.Tensor, torch.Tensor]
AttentionHiddenState = tuple[torch.Tensor, torch.Tensor]
AttentionDecayState = tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class AttentionCTMCoreOutput:
    """Per-tick outputs from the attention-enabled CTM recovery core."""

    output_synchronizations: torch.Tensor
    action_synchronizations: torch.Tensor
    state: AttentionHiddenState
    attention_weights: torch.Tensor | None = None
    pre_activations: np.ndarray | None = None
    post_activations: np.ndarray | None = None

    @property
    def final_synchronization(self) -> torch.Tensor:
        return self.output_synchronizations[:, -1]


class AttentionCTMRecoveryCore(ContinuousThoughtMachine):
    """Attention-enabled CTM core for tokenized MAPF recovery observations.

    Accepts token tensors directly with shape ``[batch, tokens, token_features]``. 
    At each internal thought tick it computes action synchronization, uses it as 
    the attention query over input tokens, updates the CTM traces, and then emits 
    output synchronization.
    """

    def __init__(
        self,
        *,
        iterations: int,
        d_model: int,
        d_input: int,
        heads: int,
        n_synch_out: int,
        n_synch_action: int,
        synapse_depth: int,
        memory_length: int,
        deep_nlms: bool,
        memory_hidden_dims: int,
        do_layernorm_nlm: bool,
        dropout: float = 0.0,
        neuron_select_type: str = "first-last",
        n_random_pairing_self: int = 0,
    ) -> None:
        if heads < 1:
            raise ValueError("AttentionCTMRecoveryCore requires heads >= 1.")
        if n_synch_action < 1:
            raise ValueError("AttentionCTMRecoveryCore requires n_synch_action >= 1.")
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=heads,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            backbone_type="none",
            positional_embedding_type="none",
            out_dims=1,
            prediction_reshaper=[-1],
            dropout=dropout,
            neuron_select_type=neuron_select_type,
            n_random_pairing_self=n_random_pairing_self,
        )

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> AttentionHiddenState:
        state_trace = self.start_trace.unsqueeze(0).repeat(batch_size, 1, 1)
        activated_state = self.start_activated_state.unsqueeze(0).repeat(batch_size, 1)
        if device is not None:
            state_trace = state_trace.to(device)
            activated_state = activated_state.to(device)
        return state_trace, activated_state

    def token_features(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor with shape [batch, tokens, token_features], got {tuple(tokens.shape)}.")
        return self.kv_proj(tokens)

    def initial_decay_state(self, activated_state: torch.Tensor) -> AttentionDecayState:
        """Initialize per-environment-step synchronization decay state."""
        batch_size = activated_state.shape[0]
        # CTM clamps decay parameters before exponentiating.
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(batch_size, 1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(batch_size, 1)
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, None, None, r_out, synch_type="out"
        )
        return None, None, decay_alpha_out, decay_beta_out, r_action, r_out

    def step_from_kv(
        self,
        kv: torch.Tensor,
        hidden_states: AttentionHiddenState,
        decay_state: AttentionDecayState,
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionHiddenState, AttentionDecayState, torch.Tensor]:
        """Advance the attention CTM by one internal thought tick."""
        state_trace, activated_state = hidden_states
        (
            decay_alpha_action,
            decay_beta_action,
            decay_alpha_out,
            decay_beta_out,
            r_action,
            r_out,
        ) = decay_state
        # Action synchronization is the CTM-native query signal. They use British spelling???
        synchronization_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
            activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type="action"
        )
        q = self.q_proj(synchronization_action).unsqueeze(1)
        attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
        attn_out = attn_out.squeeze(1)
        # Attended MAPF evidence is fused with the previous activated neuron
        # state before the synapse MLP updates the CTM trace.
        pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)
        state = self.synapses(pre_synapse_input)
        state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
        activated_state = self.trace_processor(state_trace)
        synchronization_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type="out"
        )
        return (
            synchronization_out,
            synchronization_action,
            (state_trace, activated_state),
            (decay_alpha_action, decay_beta_action, decay_alpha_out, decay_beta_out, r_action, r_out),
            attn_weights.squeeze(2),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        hidden_states: AttentionHiddenState,
        *,
        track: bool = False,
    ) -> AttentionCTMCoreOutput:
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor with shape [batch, tokens, token_features], got {tuple(tokens.shape)}.")
        batch_size = tokens.shape[0]
        kv = self.token_features(tokens)
        state_trace, activated_state = hidden_states
        if state_trace.shape[:2] != (batch_size, self.d_model):
            raise ValueError("state_trace must have shape [batch, d_model, memory_length].")
        if activated_state.shape != (batch_size, self.d_model):
            raise ValueError("activated_state must have shape [batch, d_model].")

        output_synchronizations: list[torch.Tensor] = []
        action_synchronizations: list[torch.Tensor] = []
        attention_weights: list[torch.Tensor] = []
        pre_activations_tracking: list[np.ndarray] = []
        post_activations_tracking: list[np.ndarray] = []

        # Decay state is local to one environment step's internal thought loop.
        # The recurrent neuron trace carries across environment steps.
        # The alpha / beta synchronization accumulators carry across internal ticks only.
        decay_state = self.initial_decay_state(activated_state)

        for _ in range(self.iterations):
            synchronization_out, synchronization_action, next_state, decay_state, attn_weights = self.step_from_kv(
                kv, (state_trace, activated_state), decay_state
            )
            state_trace, activated_state = next_state
            output_synchronizations.append(synchronization_out)
            action_synchronizations.append(synchronization_action)
            attention_weights.append(attn_weights)

            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())

        return AttentionCTMCoreOutput(
            output_synchronizations=torch.stack(output_synchronizations, dim=1),
            action_synchronizations=torch.stack(action_synchronizations, dim=1),
            state=(state_trace, activated_state),
            attention_weights=torch.stack(attention_weights, dim=1),
            pre_activations=np.array(pre_activations_tracking) if track else None,
            post_activations=np.array(post_activations_tracking) if track else None,
        )


@dataclass(frozen=True, slots=True)
class AdaptiveCTMCoreOutput:
    """Per-thought-tick outputs from AdaptiveCTMRecoveryCore.

    ``synchronizations`` keeps the internal thought dimension explicit with shape
    ``[batch, internal_ticks, synchronization_features]``.
    """

    synchronizations: torch.Tensor
    state: HiddenState
    pre_activations: np.ndarray | None = None
    post_activations: np.ndarray | None = None

    @property
    def final_synchronization(self) -> torch.Tensor:
        """Return the final-tick synchronization."""
        return self.synchronizations[:, -1]


class CTMRecoveryCore(ContinuousThoughtMachineRL):
    """Project alias for the recurrent CTM core used by recovery policies."""


class AdaptiveCTMRecoveryCore(CTMRecoveryCore):
    """CTM recovery core that exposes one synchronization latent per thought tick.

    For MAPF recovery it would be interesting to observe the CTM's internal thought trajectory.
    Later additions can place action heads, certainty estimates, and stopping rules on top.
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Featurize one environment-step input batch once before internal ticking."""
        return self.backbone(x)

    def step_from_features(
        self,
        features: torch.Tensor,
        hidden_states: HiddenState,
    ) -> tuple[torch.Tensor, HiddenState, torch.Tensor, torch.Tensor]:
        """Advance the CTM by exactly one internal thought tick.

        Lets the adaptive policy stop different agents at different internal depths during inference 
        while preserving the state corresponding to the tick at which each agent actually stopped.
        """
        state_trace, activated_state_trace = hidden_states
        pre_synapse_input = torch.concatenate((features.reshape(features.size(0), -1), activated_state_trace[:, :, -1]), -1)
        state = self.synapses(pre_synapse_input)
        state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

        activated_state = self.trace_processor(state_trace)
        activated_state_trace = torch.concatenate((activated_state_trace[:, :, 1:], activated_state.unsqueeze(-1)), -1)
        synchronization = self.compute_synchronisation(activated_state_trace)
        return synchronization, (state_trace, activated_state_trace), state, activated_state

    def forward(self, x: torch.Tensor, hidden_states: HiddenState, track: bool = False) -> AdaptiveCTMCoreOutput:
        pre_activations_tracking: list[np.ndarray] = []
        post_activations_tracking: list[np.ndarray] = []
        synchronizations: list[torch.Tensor] = []

        # Feature extraction is done once per environment step. 
        # Only the CTM recurrent state evolves over the internal thought ticks.
        features = self.features(x)
        state = hidden_states

        for _ in range(self.iterations):
            synchronization, state, pre_activation, post_activation = self.step_from_features(features, state)
            synchronizations.append(synchronization)

            if track:
                pre_activations_tracking.append(pre_activation.detach().cpu().numpy())
                post_activations_tracking.append(post_activation.detach().cpu().numpy())

        return AdaptiveCTMCoreOutput(
            synchronizations=torch.stack(synchronizations, dim=1),
            state=state,
            pre_activations=np.array(pre_activations_tracking) if track else None,
            post_activations=np.array(post_activations_tracking) if track else None,
        )


__all__ = [
    "AdaptiveCTMCoreOutput",
    "AdaptiveCTMRecoveryCore",
    "AttentionCTMCoreOutput",
    "AttentionDecayState",
    "AttentionCTMRecoveryCore",
    "CTMRecoveryCore",
    "ContinuousThoughtMachine",
    "ContinuousThoughtMachineRL",
    "ensure_ctm_on_path",
]
