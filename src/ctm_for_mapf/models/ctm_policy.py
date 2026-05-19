"""
Copyright (c) 2026 Michael Mulder, Shaun Christopher Lee.

CTM policy wrappers for temporal MAPF recovery.

Third-party dependencies used by this module:
- Continuous Thought Machines, SakanaAI, Apache License 2.0:
  https://github.com/SakanaAI/continuous-thought-machines
- PyTorch, PyTorch Foundation / contributors:
  https://pytorch.org/
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ctm_for_mapf.models.communication import CommunicationState, mean_pool_neighbor_messages, radius_neighbor_mask
from ctm_for_mapf.models.ctm_adapter import AdaptiveCTMRecoveryCore, AttentionCTMRecoveryCore, CTMRecoveryCore
from ctm_for_mapf.models.recovery_policy import (
    AdaptiveActionOutput,
    AdaptiveCTMPolicyOutput,
    AttentionCTMPolicyOutput,
    CommunicatingPolicyOutput,
    PolicyOutput,
    RecurrentState,
    RecurrentTemporalRecoveryPolicy,
    UnifiedAdaptiveActionOutput,
    UnifiedCTMPolicyOutput,
)


class _BaseCTMTemporalRecoveryPolicy(RecurrentTemporalRecoveryPolicy):
    """Shared construction and validation for CTM temporal-recovery policies."""

    core_type = CTMRecoveryCore

    def __init__(
        self,
        *,
        input_dim: int,
        iterations: int = 2,
        d_model: int = 128,
        d_input: int = 64,
        n_synch_out: int = 16,
        synapse_depth: int = 1,
        memory_length: int = 5,
        deep_nlms: bool = True,
        memory_hidden_dims: int = 16,
        do_layernorm_nlm: bool = False,
        dropout: float = 0.0,
        neuron_select_type: str = "first-last",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        # The flat-vector CTM path uses the official RL-style CTM core. An old baseline.
        self.core = self.core_type(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            n_synch_out=n_synch_out,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            backbone_type="classic-control-backbone",
            prediction_reshaper=[-1],
            dropout=dropout,
            neuron_select_type=neuron_select_type,
        )
        self.action_head = nn.Linear(self.core.synch_representation_size_out, self.num_decisions)

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> RecurrentState:
        state_trace = self.core.start_trace.unsqueeze(0).repeat(batch_size, 1, 1)
        activated_trace = self.core.start_activated_trace.unsqueeze(0).repeat(batch_size, 1, 1)
        if device is not None:
            state_trace = state_trace.to(device)
            activated_trace = activated_trace.to(device)
        return state_trace, activated_trace

    def _validate_observations(self, observations: torch.Tensor) -> None:
        if observations.ndim != 2:
            raise ValueError(f"Expected observations with shape [batch, features], got {tuple(observations.shape)}.")
        if observations.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {observations.shape[-1]}.")


class AttentionCTMTemporalRecoveryPolicy(RecurrentTemporalRecoveryPolicy):
    """Trainable temporal-recovery policy using action-synchronization attention."""

    def __init__(
        self,
        *,
        token_dim: int,
        iterations: int = 2,
        d_model: int = 128,
        d_input: int = 64,
        heads: int = 4,
        n_synch_out: int = 16,
        n_synch_action: int = 16,
        synapse_depth: int = 1,
        memory_length: int = 5,
        deep_nlms: bool = True,
        memory_hidden_dims: int = 16,
        do_layernorm_nlm: bool = False,
        dropout: float = 0.0,
        neuron_select_type: str = "first-last",
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.core = AttentionCTMRecoveryCore(
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
            dropout=dropout,
            neuron_select_type=neuron_select_type,
        )
        self.action_head = nn.Linear(self.core.synch_representation_size_out, self.num_decisions)

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> RecurrentState:
        return self.core.initial_state(batch_size, device=device)

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [batch, tokens, token_dim], got {tuple(tokens.shape)}.")
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected token feature dimension {self.token_dim}, got {tokens.shape[-1]}.")

    def forward(self, observations: torch.Tensor, state: RecurrentState) -> AttentionCTMPolicyOutput:
        self._validate_tokens(observations)
        # Attention-only policy, no communication here.
        core_output = self.core(observations, state)
        all_latents = core_output.output_synchronizations
        all_logits = self.action_head(all_latents)
        certainties = torch.stack(
            [self.core.compute_certainty(all_logits[:, tick]) for tick in range(all_logits.shape[1])],
            dim=-1,
        )
        return AttentionCTMPolicyOutput(
            logits=all_logits[:, -1],
            state=core_output.state,
            latent=all_latents[:, -1],
            all_logits=all_logits,
            all_latents=all_latents,
            certainties=certainties,
            action_latents=core_output.action_synchronizations,
            attention_weights=core_output.attention_weights,
        )


class CTMTemporalRecoveryPolicy(_BaseCTMTemporalRecoveryPolicy):
    """Fixed-iteration CTM policy matching the official RL-style final latent."""

    def forward(self, observations: torch.Tensor, state: RecurrentState) -> PolicyOutput:
        self._validate_observations(observations)
        latent, next_state = self.core(observations, state)
        logits = self.action_head(latent)
        return PolicyOutput(logits=logits, state=next_state, latent=latent)


class AdaptiveCTMTemporalRecoveryPolicy(_BaseCTMTemporalRecoveryPolicy):
    """CTM policy that exposes logits and certainty at every internal thought tick."""

    core_type = AdaptiveCTMRecoveryCore

    def forward(self, observations: torch.Tensor, state: RecurrentState) -> AdaptiveCTMPolicyOutput:
        self._validate_observations(observations)
        core_output = self.core(observations, state)
        all_latents = core_output.synchronizations
        all_logits = self.action_head(all_latents)
        certainties = torch.stack(
            [self.core.compute_certainty(all_logits[:, tick]) for tick in range(all_logits.shape[1])],
            dim=-1,
        )
        return AdaptiveCTMPolicyOutput(
            logits=all_logits[:, -1],
            state=core_output.state,
            latent=all_latents[:, -1],
            all_logits=all_logits,
            all_latents=all_latents,
            certainties=certainties,
        )

    @torch.no_grad()
    def act_adaptive(
        self,
        observations: torch.Tensor,
        state: RecurrentState,
        *,
        certainty_threshold: float | None,
        min_ticks: int = 1,
        max_ticks: int | None = None,
    ) -> AdaptiveActionOutput:
        """Act with per-agent adaptive stopping.

        If ``certainty_threshold`` is ``None``, the method behaves like a fixed-budget
        policy and uses exactly ``max_ticks`` internal ticks. Otherwise each agent stops
        at the first tick at or after ``min_ticks`` whose certainty crosses the threshold.
        Agents that never cross it use the last permitted tick.
        """
        self._validate_observations(observations)
        if min_ticks < 1:
            raise ValueError("min_ticks must be >= 1.")
        tick_budget = self.core.iterations if max_ticks is None else max_ticks
        if tick_budget < 1 or tick_budget > self.core.iterations:
            raise ValueError(f"max_ticks must be in [1, {self.core.iterations}], got {tick_budget}.")
        if min_ticks > tick_budget:
            raise ValueError("min_ticks cannot exceed max_ticks.")
        if certainty_threshold is not None and not 0.0 <= certainty_threshold <= 1.0:
            raise ValueError("certainty_threshold must be in [0, 1] or None.")

        batch_size = observations.shape[0]
        device = observations.device
        # Featurize once, then run only active examples through one-tick CTM steps.
        features = self.core.features(observations)
        running_state = (state[0].clone(), state[1].clone())
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        ticks_used = torch.zeros(batch_size, dtype=torch.long, device=device)
        selected_logits = torch.empty(batch_size, self.num_decisions, device=device, dtype=observations.dtype)
        selected_latents = torch.empty(batch_size, self.core.synch_representation_size_out, device=device, dtype=observations.dtype)
        selected_certainties = torch.empty(batch_size, 2, device=device, dtype=observations.dtype)

        for tick in range(1, tick_budget + 1):
            # Only agents that have not stopped continue consuming CTM compute.
            active_indices = active.nonzero(as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                break

            active_features = features[active_indices]
            active_state = (running_state[0][active_indices], running_state[1][active_indices])
            active_latents, next_active_state, _, _ = self.core.step_from_features(active_features, active_state)
            active_logits = self.action_head(active_latents)
            active_certainties = self.core.compute_certainty(active_logits)

            running_state[0][active_indices] = next_active_state[0]
            running_state[1][active_indices] = next_active_state[1]

            if certainty_threshold is None:
                should_stop = torch.full((active_indices.numel(),), tick == tick_budget, dtype=torch.bool, device=device)
            else:
                should_stop = active_certainties[:, 1] >= certainty_threshold
                if tick < min_ticks:
                    should_stop = torch.zeros_like(should_stop)
                if tick == tick_budget:
                    should_stop = torch.ones_like(should_stop)

            if should_stop.any():
                stopping_indices = active_indices[should_stop]
                selected_logits[stopping_indices] = active_logits[should_stop]
                selected_latents[stopping_indices] = active_latents[should_stop]
                selected_certainties[stopping_indices] = active_certainties[should_stop]
                ticks_used[stopping_indices] = tick
                active[stopping_indices] = False

        return AdaptiveActionOutput(
            decisions=selected_logits.argmax(dim=-1),
            state=running_state,
            ticks_used=ticks_used,
            selected_logits=selected_logits,
            selected_latents=selected_latents,
            selected_certainties=selected_certainties,
        )


class CommunicatingCTMTemporalRecoveryPolicy(_BaseCTMTemporalRecoveryPolicy):
    """Fixed-budget CTM policy with one-step-delayed synchronization messages.

    The CTM still acts from its native output synchronization latent. Communication is
    derived from that same latent through ``message_head`` and delivered to neighbors on
    the next environment step, preserving explicit decentralized timing semantics.
    """

    def __init__(
        self,
        *,
        local_input_dim: int,
        message_dim: int = 16,
        communication_radius: int = 2,
        **kwargs,
    ) -> None:
        if message_dim < 1:
            raise ValueError("message_dim must be >= 1.")
        if communication_radius < 0:
            raise ValueError("communication_radius must be >= 0.")
        super().__init__(input_dim=local_input_dim + message_dim, **kwargs)
        self.local_input_dim = local_input_dim
        self.message_dim = message_dim
        self.communication_radius = communication_radius
        self.message_head = nn.Linear(self.core.synch_representation_size_out, message_dim)

    def forward(self, observations: torch.Tensor, state: RecurrentState) -> PolicyOutput:
        """Run the underlying fixed-budget CTM on already-augmented inputs."""
        self._validate_observations(observations)
        latent, next_state = self.core(observations, state)
        logits = self.action_head(latent)
        return PolicyOutput(logits=logits, state=next_state, latent=latent)

    def initial_joint_state(
        self,
        batch_size: int,
        num_agents: int,
        *,
        device: torch.device | str | None = None,
    ) -> RecurrentState:
        flat_state = self.initial_state(batch_size * num_agents, device=device)
        return tuple(tensor.reshape(batch_size, num_agents, *tensor.shape[1:]) for tensor in flat_state)  # type: ignore[return-value]

    def initial_communication_state(
        self,
        batch_size: int,
        num_agents: int,
        *,
        device: torch.device | str | None = None,
    ) -> CommunicationState:
        return CommunicationState(
            received_messages=torch.zeros(batch_size, num_agents, self.message_dim, device=device),
        )

    def _validate_joint_inputs(
        self,
        observations: torch.Tensor,
        positions: torch.Tensor,
        state: RecurrentState,
        communication_state: CommunicationState,
        agent_mask: torch.Tensor | None,
    ) -> None:
        if observations.ndim != 3:
            raise ValueError(f"Expected observations with shape [batch, agents, features], got {tuple(observations.shape)}.")
        if observations.shape[-1] != self.local_input_dim:
            raise ValueError(f"Expected local feature dimension {self.local_input_dim}, got {observations.shape[-1]}.")
        if positions.shape != (*observations.shape[:2], 2):
            raise ValueError(f"Expected positions with shape {(*observations.shape[:2], 2)}, got {tuple(positions.shape)}.")
        if communication_state.received_messages.shape != (*observations.shape[:2], self.message_dim):
            raise ValueError(
                "Expected received messages with shape "
                f"{(*observations.shape[:2], self.message_dim)}, got {tuple(communication_state.received_messages.shape)}."
            )
        if state[0].shape[:2] != observations.shape[:2] or state[1].shape[:2] != observations.shape[:2]:
            raise ValueError("Joint recurrent state must begin with [batch, agents, ...].")
        if agent_mask is not None and agent_mask.shape != observations.shape[:2]:
            raise ValueError(f"Expected agent_mask with shape {tuple(observations.shape[:2])}, got {tuple(agent_mask.shape)}.")


    @torch.no_grad()
    def act_joint(
        self,
        observations: torch.Tensor,
        positions: torch.Tensor,
        state: RecurrentState,
        communication_state: CommunicationState,
        *,
        agent_mask: torch.Tensor | None = None,
        message_mode: str = "learned",
    ) -> tuple[torch.Tensor, RecurrentState, CommunicationState]:
        """Return decentralized decisions plus next recurrent and communication state."""
        output = self.forward_joint(
            observations,
            positions,
            state,
            communication_state,
            agent_mask=agent_mask,
            message_mode=message_mode,
        )
        return output.logits.argmax(dim=-1), output.state, output.next_communication_state

    def forward_joint(
        self,
        observations: torch.Tensor,
        positions: torch.Tensor,
        state: RecurrentState,
        communication_state: CommunicationState,
        *,
        agent_mask: torch.Tensor | None = None,
        message_mode: str = "learned",
    ) -> CommunicatingPolicyOutput:
        """Run one decentralized environment step over a joint agent batch."""
        self._validate_joint_inputs(observations, positions, state, communication_state, agent_mask)
        batch_size, num_agents, _ = observations.shape
        # Communication-only ablation ablation test: Does the message do anything?
        full_inputs = torch.cat((observations, communication_state.received_messages), dim=-1)
        flat_inputs = full_inputs.reshape(batch_size * num_agents, -1)
        flat_state = tuple(tensor.reshape(batch_size * num_agents, *tensor.shape[2:]) for tensor in state)
        latent, flat_next_state = self.core(flat_inputs, flat_state)
        logits = self.action_head(latent).reshape(batch_size, num_agents, self.num_decisions)
        latent = latent.reshape(batch_size, num_agents, -1)
        updated_state = tuple(tensor.reshape(batch_size, num_agents, *tensor.shape[1:]) for tensor in flat_next_state)
        if agent_mask is None:
            next_state = updated_state
        else:
            next_state = tuple(
                torch.where(agent_mask[..., None, None], updated, previous)
                for updated, previous in zip(updated_state, state)
            )
            logits = logits * agent_mask.unsqueeze(-1).to(logits.dtype)
            latent = latent * agent_mask.unsqueeze(-1).to(latent.dtype)

        # Messages are derived from CTM output synchronization.
        outgoing_messages = self.message_head(latent)
        # Experimental permutations to mess with the model.
        if message_mode == "zero":
            outgoing_messages = torch.zeros_like(outgoing_messages)
        elif message_mode == "shuffled":
            outgoing_messages = torch.roll(outgoing_messages, shifts=1, dims=1)
        elif message_mode != "learned":
            raise ValueError("message_mode must be one of {'learned', 'zero', 'shuffled'}.")
        if agent_mask is not None:
            outgoing_messages = outgoing_messages * agent_mask.unsqueeze(-1).to(outgoing_messages.dtype)
        neighbor_mask = radius_neighbor_mask(
            positions,
            radius=self.communication_radius,
            agent_mask=agent_mask,
            include_self=False,
        )
        next_received_messages = mean_pool_neighbor_messages(outgoing_messages, neighbor_mask)
        if agent_mask is not None:
            next_received_messages = next_received_messages * agent_mask.unsqueeze(-1).to(next_received_messages.dtype)

        return CommunicatingPolicyOutput(
            logits=logits,
            state=next_state,
            latent=latent,
            outgoing_messages=outgoing_messages,
            received_messages=communication_state.received_messages,
            next_communication_state=CommunicationState(received_messages=next_received_messages),
            neighbor_mask=neighbor_mask,
        )


class UnifiedCTMRecoveryPolicy(RecurrentTemporalRecoveryPolicy):
    """Unified CTM policy with token attention, adaptive compute, and communication. This is the real deal."""

    def __init__(
        self,
        *,
        token_dim: int,
        message_dim: int = 16,
        communication_radius: int = 2,
        iterations: int = 2,
        d_model: int = 128,
        d_input: int = 64,
        heads: int = 4,
        n_synch_out: int = 16,
        n_synch_action: int = 16,
        synapse_depth: int = 1,
        memory_length: int = 5,
        deep_nlms: bool = True,
        memory_hidden_dims: int = 16,
        do_layernorm_nlm: bool = False,
        dropout: float = 0.0,
        neuron_select_type: str = "first-last",
    ) -> None:
        super().__init__()
        if message_dim < 1:
            raise ValueError("message_dim must be >= 1.")
        if communication_radius < 0:
            raise ValueError("communication_radius must be >= 0.")
        self.token_dim = token_dim
        self.message_dim = message_dim
        self.communication_radius = communication_radius
        self.core = AttentionCTMRecoveryCore(
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
            dropout=dropout,
            neuron_select_type=neuron_select_type,
        )
        self.action_head = nn.Linear(self.core.synch_representation_size_out, self.num_decisions)
        # Both actions and outgoing messages are read from output synchronization. The
        # received message is projected into token space so the next step's action synchronization 
        # can decide whether to attend to it.
        self.message_head = nn.Linear(self.core.synch_representation_size_out, message_dim)
        self.message_token_projector = nn.Linear(message_dim, token_dim)

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> RecurrentState:
        return self.core.initial_state(batch_size, device=device)

    def initial_joint_state(
        self,
        batch_size: int,
        num_agents: int,
        *,
        device: torch.device | str | None = None,
    ) -> RecurrentState:
        flat_state = self.initial_state(batch_size * num_agents, device=device)
        return tuple(tensor.reshape(batch_size, num_agents, *tensor.shape[1:]) for tensor in flat_state)  # type: ignore[return-value]

    def initial_communication_state(
        self,
        batch_size: int,
        num_agents: int,
        *,
        device: torch.device | str | None = None,
    ) -> CommunicationState:
        return CommunicationState(
            received_messages=torch.zeros(batch_size, num_agents, self.message_dim, device=device),
        )

    def _augment_tokens_with_messages(self, tokens: torch.Tensor, communication_state: CommunicationState) -> torch.Tensor:
        # Append communication as one extra token instead of concatenating it to every feature.
        message_token = self.message_token_projector(communication_state.received_messages).unsqueeze(-2)
        return torch.cat((tokens, message_token), dim=-2)

    def _validate_joint_tokens(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        state: RecurrentState,
        communication_state: CommunicationState,
        agent_mask: torch.Tensor | None,
    ) -> None:
        if tokens.ndim != 4:
            raise ValueError(f"Expected tokens with shape [batch, agents, tokens, token_dim], got {tuple(tokens.shape)}.")
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected token feature dimension {self.token_dim}, got {tokens.shape[-1]}.")
        if positions.shape != (*tokens.shape[:2], 2):
            raise ValueError(f"Expected positions with shape {(*tokens.shape[:2], 2)}, got {tuple(positions.shape)}.")
        if communication_state.received_messages.shape != (*tokens.shape[:2], self.message_dim):
            raise ValueError(
                "Expected received messages with shape "
                f"{(*tokens.shape[:2], self.message_dim)}, got {tuple(communication_state.received_messages.shape)}."
            )
        if state[0].shape[:2] != tokens.shape[:2] or state[1].shape[:2] != tokens.shape[:2]:
            raise ValueError("Joint recurrent state must begin with [batch, agents, ...].")
        if agent_mask is not None and agent_mask.shape != tokens.shape[:2]:
            raise ValueError(f"Expected agent_mask with shape {tuple(tokens.shape[:2])}, got {tuple(agent_mask.shape)}.")

    def forward(self, observations: torch.Tensor, state: RecurrentState) -> AttentionCTMPolicyOutput:
        if observations.ndim != 3:
            raise ValueError(f"Expected tokens with shape [batch, tokens, token_dim], got {tuple(observations.shape)}.")
        if observations.shape[-1] != self.token_dim:
            raise ValueError(f"Expected token feature dimension {self.token_dim}, got {observations.shape[-1]}.")
        core_output = self.core(observations, state)
        all_latents = core_output.output_synchronizations
        all_logits = self.action_head(all_latents)
        certainties = torch.stack(
            [self.core.compute_certainty(all_logits[:, tick]) for tick in range(all_logits.shape[1])],
            dim=-1,
        )
        return AttentionCTMPolicyOutput(
            logits=all_logits[:, -1],
            state=core_output.state,
            latent=all_latents[:, -1],
            all_logits=all_logits,
            all_latents=all_latents,
            certainties=certainties,
            action_latents=core_output.action_synchronizations,
            attention_weights=core_output.attention_weights,
        )

    def forward_joint_tokens(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        state: RecurrentState,
        communication_state: CommunicationState,
        *,
        agent_mask: torch.Tensor | None = None,
        message_mode: str = "learned",
    ) -> UnifiedCTMPolicyOutput:
        """Fixed-budget joint forward path for training and evaluation."""
        self._validate_joint_tokens(tokens, positions, state, communication_state, agent_mask)
        batch_size, num_agents, _, _ = tokens.shape
        augmented_tokens = self._augment_tokens_with_messages(tokens, communication_state)
        # We flatten [batch, agent], then reshape outputs back to joint-agent form for
        # message routing and MAPF metrics.
        flat_tokens = augmented_tokens.reshape(batch_size * num_agents, augmented_tokens.shape[-2], self.token_dim)
        flat_state = tuple(tensor.reshape(batch_size * num_agents, *tensor.shape[2:]) for tensor in state)
        core_output = self.core(flat_tokens, flat_state)
        all_latents = core_output.output_synchronizations.reshape(batch_size, num_agents, self.core.iterations, -1)
        action_latents = core_output.action_synchronizations.reshape(batch_size, num_agents, self.core.iterations, -1)
        all_logits = self.action_head(all_latents)
        certainties = torch.stack(
            [self.core.compute_certainty(all_logits[:, :, tick].reshape(batch_size * num_agents, -1)).reshape(batch_size, num_agents, 2) for tick in range(all_logits.shape[2])],
            dim=-1,
        )
        # Fixed-budget training exposes every tick but uses the final tick as the
        # common online-policy output.
        logits = all_logits[:, :, -1]
        latent = all_latents[:, :, -1]
        attention_weights = None
        if core_output.attention_weights is not None:
            attention_weights = core_output.attention_weights.reshape(
                batch_size, num_agents, self.core.iterations, core_output.attention_weights.shape[-2], core_output.attention_weights.shape[-1]
            )
        updated_state = tuple(tensor.reshape(batch_size, num_agents, *tensor.shape[1:]) for tensor in core_output.state)
        if agent_mask is None:
            next_state = updated_state
        else:
            next_state = tuple(
                torch.where(agent_mask[..., None, None] if updated.ndim == 4 else agent_mask[..., None], updated, previous)
                for updated, previous in zip(updated_state, state)
            )
            mask_f = agent_mask.to(logits.dtype)
            logits = logits * mask_f.unsqueeze(-1)
            latent = latent * mask_f.unsqueeze(-1)
            all_logits = all_logits * mask_f.unsqueeze(-1).unsqueeze(-1)
            all_latents = all_latents * mask_f.unsqueeze(-1).unsqueeze(-1)
            action_latents = action_latents * mask_f.unsqueeze(-1).unsqueeze(-1)
            certainties = certainties * mask_f.unsqueeze(-1).unsqueeze(-1)

        # Messages emitted now become the next CommunicationState. They are not fed
        # back into the current action, preserving one-step-delayed decentralized timing.
        outgoing_messages = self._messages_from_latent(latent, agent_mask=agent_mask, message_mode=message_mode)
        neighbor_mask = radius_neighbor_mask(positions, radius=self.communication_radius, agent_mask=agent_mask, include_self=False)
        next_received_messages = mean_pool_neighbor_messages(outgoing_messages, neighbor_mask)
        if agent_mask is not None:
            next_received_messages = next_received_messages * agent_mask.unsqueeze(-1).to(next_received_messages.dtype)

        return UnifiedCTMPolicyOutput(
            logits=logits,
            state=next_state,
            latent=latent,
            all_logits=all_logits,
            all_latents=all_latents,
            certainties=certainties,
            action_latents=action_latents,
            attention_weights=attention_weights,
            outgoing_messages=outgoing_messages,
            received_messages=communication_state.received_messages,
            next_communication_state=CommunicationState(received_messages=next_received_messages),
            neighbor_mask=neighbor_mask,
        )

    def _messages_from_latent(
        self,
        latent: torch.Tensor,
        *,
        agent_mask: torch.Tensor | None,
        message_mode: str,
    ) -> torch.Tensor:
        outgoing_messages = self.message_head(latent)
        if message_mode == "zero":
            outgoing_messages = torch.zeros_like(outgoing_messages)
        elif message_mode == "shuffled":
            outgoing_messages = torch.roll(outgoing_messages, shifts=1, dims=1)
        elif message_mode != "learned":
            raise ValueError("message_mode must be one of {'learned', 'zero', 'shuffled'}.")
        if agent_mask is not None:
            outgoing_messages = outgoing_messages * agent_mask.unsqueeze(-1).to(outgoing_messages.dtype)
        return outgoing_messages

    @torch.no_grad()
    def act_joint_adaptive_tokens(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        state: RecurrentState,
        communication_state: CommunicationState,
        *,
        agent_mask: torch.Tensor | None = None,
        certainty_threshold: float | None,
        min_ticks: int = 1,
        max_ticks: int | None = None,
        message_mode: str = "learned",
    ) -> UnifiedAdaptiveActionOutput:
        """Adaptive joint inference over tokenized observations."""
        self._validate_joint_tokens(tokens, positions, state, communication_state, agent_mask)
        if min_ticks < 1:
            raise ValueError("min_ticks must be >= 1.")
        tick_budget = self.core.iterations if max_ticks is None else max_ticks
        if tick_budget < 1 or tick_budget > self.core.iterations:
            raise ValueError(f"max_ticks must be in [1, {self.core.iterations}], got {tick_budget}.")
        if min_ticks > tick_budget:
            raise ValueError("min_ticks cannot exceed max_ticks.")
        if certainty_threshold is not None and not 0.0 <= certainty_threshold <= 1.0:
            raise ValueError("certainty_threshold must be in [0, 1] or None.")

        batch_size, num_agents, _, _ = tokens.shape
        total_agents = batch_size * num_agents
        augmented_tokens = self._augment_tokens_with_messages(tokens, communication_state)
        flat_tokens = augmented_tokens.reshape(total_agents, augmented_tokens.shape[-2], self.token_dim)
        # Key/value token features do not change across internal thought ticks.
        # Compute once and reuse.
        kv = self.core.token_features(flat_tokens)
        running_state = tuple(tensor.reshape(total_agents, *tensor.shape[2:]).clone() for tensor in state)
        flat_agent_mask = torch.ones(total_agents, dtype=torch.bool, device=tokens.device) if agent_mask is None else agent_mask.reshape(-1).to(tokens.device)
        active = flat_agent_mask.clone()
        ticks_used = torch.zeros(total_agents, dtype=torch.long, device=tokens.device)
        selected_logits = torch.zeros(total_agents, self.num_decisions, device=tokens.device, dtype=tokens.dtype)
        selected_latents = torch.zeros(total_agents, self.core.synch_representation_size_out, device=tokens.device, dtype=tokens.dtype)
        selected_certainties = torch.zeros(total_agents, 2, device=tokens.device, dtype=tokens.dtype)

        decay_state = self.core.initial_decay_state(running_state[1])
        action_alpha, action_beta, out_alpha, out_beta, r_action, r_out = decay_state

        for tick in range(1, tick_budget + 1):
            active_indices = active.nonzero(as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                break
            active_decay = (
                None if action_alpha is None else action_alpha[active_indices],
                None if action_beta is None else action_beta[active_indices],
                out_alpha[active_indices],
                out_beta[active_indices],
                r_action[active_indices],
                r_out[active_indices],
            )
            active_out, _, next_active_state, next_active_decay, _ = self.core.step_from_kv(
                kv[active_indices],
                (running_state[0][active_indices], running_state[1][active_indices]),
                active_decay,
            )
            active_logits = self.action_head(active_out)
            active_certainties = self.core.compute_certainty(active_logits)
            running_state[0][active_indices] = next_active_state[0]
            running_state[1][active_indices] = next_active_state[1]
            next_action_alpha, next_action_beta, next_out_alpha, next_out_beta, _, _ = next_active_decay
            if action_alpha is None:
                action_alpha = torch.zeros(total_agents, next_action_alpha.shape[-1], device=tokens.device, dtype=tokens.dtype)
                action_beta = torch.zeros_like(action_alpha)
            assert action_beta is not None
            action_alpha[active_indices] = next_action_alpha
            action_beta[active_indices] = next_action_beta
            out_alpha[active_indices] = next_out_alpha
            out_beta[active_indices] = next_out_beta

            if certainty_threshold is None:
                should_stop = torch.full((active_indices.numel(),), tick == tick_budget, dtype=torch.bool, device=tokens.device)
            else:
                should_stop = active_certainties[:, 1] >= certainty_threshold
                if tick < min_ticks:
                    should_stop = torch.zeros_like(should_stop)
                if tick == tick_budget:
                    should_stop = torch.ones_like(should_stop)
            if should_stop.any():
                stopping_indices = active_indices[should_stop]
                selected_logits[stopping_indices] = active_logits[should_stop]
                selected_latents[stopping_indices] = active_out[should_stop]
                selected_certainties[stopping_indices] = active_certainties[should_stop]
                ticks_used[stopping_indices] = tick
                active[stopping_indices] = False

        selected_logits = selected_logits.reshape(batch_size, num_agents, self.num_decisions)
        selected_latents = selected_latents.reshape(batch_size, num_agents, -1)
        selected_certainties = selected_certainties.reshape(batch_size, num_agents, 2)
        ticks_used = ticks_used.reshape(batch_size, num_agents)
        next_state = tuple(tensor.reshape(batch_size, num_agents, *tensor.shape[1:]) for tensor in running_state)
        # Adaptive communication sends the synchronization latent from the stopping
        # tick, not always the max-budget final tick. That makes message content
        # consistent with the evidence used for the selected action.
        outgoing_messages = self._messages_from_latent(selected_latents, agent_mask=agent_mask, message_mode=message_mode)
        neighbor_mask = radius_neighbor_mask(positions, radius=self.communication_radius, agent_mask=agent_mask, include_self=False)
        next_received_messages = mean_pool_neighbor_messages(outgoing_messages, neighbor_mask)
        if agent_mask is not None:
            next_received_messages = next_received_messages * agent_mask.unsqueeze(-1).to(next_received_messages.dtype)
        return UnifiedAdaptiveActionOutput(
            decisions=selected_logits.argmax(dim=-1),
            state=next_state,
            next_communication_state=CommunicationState(received_messages=next_received_messages),
            ticks_used=ticks_used,
            selected_logits=selected_logits,
            selected_latents=selected_latents,
            selected_certainties=selected_certainties,
            outgoing_messages=outgoing_messages,
            received_messages=communication_state.received_messages,
            neighbor_mask=neighbor_mask,
        )
