"""Paired communication-ablation diagnostics for communicating CTM policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
from pogema import GridConfig

from ctm_for_mapf.envs import DisturbanceModel, ForcedWaitDelayModel, RecoveryEnv
from ctm_for_mapf.models import (
    FOLLOW_PLAN_DECISION,
    CommunicatingCTMTemporalRecoveryPolicy,
    RecoveryObservationVectorizer,
)

MessageMode = Literal["learned", "zero", "shuffled"]


@dataclass(frozen=True, slots=True)
class MessageSensitivityResult:
    success: bool
    nominal_makespan: int
    actual_makespan: int
    compared_agent_steps: int
    divergent_agent_steps: int
    communication_edges: int
    receive_agent_steps: int
    total_received_norm: float
    total_outgoing_norm: float

    @property
    def action_divergence_rate(self) -> float:
        return self.divergent_agent_steps / self.compared_agent_steps if self.compared_agent_steps else 0.0


@dataclass(frozen=True, slots=True)
class MessageSensitivitySummary:
    episodes: int
    success_rate: float
    mean_actual_makespan: float
    action_divergence_rate: float
    mean_communication_edges: float
    mean_receive_agent_steps: float
    mean_received_norm_per_decision: float
    mean_outgoing_norm_per_decision: float


def run_message_sensitivity_rollout(
    *,
    policy: CommunicatingCTMTemporalRecoveryPolicy,
    grid_config: GridConfig,
    seed: int,
    reference_mode: MessageMode = "learned",
    ablation_mode: MessageMode = "zero",
    delay_probability: float = 0.0,
    delay_model_factory: Callable[[], DisturbanceModel] | None = None,
    device: torch.device | str = "cpu",
) -> MessageSensitivityResult:
    """Compare two message modes on the same trajectory driven by ``reference_mode``."""
    env = RecoveryEnv(
        grid_config.copy(update={"seed": seed}),
        delay_model=(delay_model_factory() if delay_model_factory is not None else ForcedWaitDelayModel(delay_probability=delay_probability)),
        seed=seed,
    )
    observations, reset_info = env.reset()
    vectorizer = RecoveryObservationVectorizer()
    reference_state = policy.initial_joint_state(1, env.num_agents, device=device)
    reference_comm = policy.initial_communication_state(1, env.num_agents, device=device)
    ablation_state = policy.initial_joint_state(1, env.num_agents, device=device)
    ablation_comm = policy.initial_communication_state(1, env.num_agents, device=device)

    compared_agent_steps = 0
    divergent_agent_steps = 0
    communication_edges = 0
    receive_agent_steps = 0
    total_received_norm = 0.0
    total_outgoing_norm = 0.0
    terminated = [False] * env.num_agents
    truncated = [False] * env.num_agents

    for _ in range(grid_config.max_episode_steps):
        positions = env.base_env.get_agents_xy(ignore_borders=True)
        batch = vectorizer.batch_tensor(observations, device=device).unsqueeze(0)
        joint_positions = torch.as_tensor(positions, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            reference_output = policy.forward_joint(
                batch,
                joint_positions,
                reference_state,
                reference_comm,
                message_mode=reference_mode,
            )
            ablation_output = policy.forward_joint(
                batch,
                joint_positions,
                ablation_state,
                ablation_comm,
                message_mode=ablation_mode,
            )
        reference_decisions = reference_output.logits.argmax(dim=-1).squeeze(0)
        ablation_decisions = ablation_output.logits.argmax(dim=-1).squeeze(0)
        compared_agent_steps += int(reference_decisions.numel())
        divergent_agent_steps += int((reference_decisions != ablation_decisions).sum().item())
        communication_edges += int(reference_output.neighbor_mask.sum().item())
        received_norms = reference_output.received_messages.norm(dim=-1)
        outgoing_norms = reference_output.outgoing_messages.norm(dim=-1)
        receive_agent_steps += int((received_norms > 0).sum().item())
        total_received_norm += float(received_norms.sum().item())
        total_outgoing_norm += float(outgoing_norms.sum().item())

        actions = [
            0 if int(decision) != FOLLOW_PLAN_DECISION else env.tracker.path_next_action(agent_id, tuple(position))
            for agent_id, (decision, position) in enumerate(zip(reference_decisions.cpu().tolist(), positions))
        ]
        observations, _, terminated, truncated, _ = env.step(actions)
        reference_state = reference_output.state
        reference_comm = reference_output.next_communication_state
        ablation_state = ablation_output.state
        ablation_comm = ablation_output.next_communication_state
        if all(terminated) or all(truncated):
            break

    return MessageSensitivityResult(
        success=all(terminated),
        nominal_makespan=int(reset_info["nominal_makespan"]),
        actual_makespan=env.tracker.timestep,
        compared_agent_steps=compared_agent_steps,
        divergent_agent_steps=divergent_agent_steps,
        communication_edges=communication_edges,
        receive_agent_steps=receive_agent_steps,
        total_received_norm=total_received_norm,
        total_outgoing_norm=total_outgoing_norm,
    )


def summarize_message_sensitivity(results: list[MessageSensitivityResult]) -> MessageSensitivitySummary:
    if not results:
        raise ValueError("Cannot summarize zero message-sensitivity results.")
    compared = sum(result.compared_agent_steps for result in results)
    divergent = sum(result.divergent_agent_steps for result in results)
    return MessageSensitivitySummary(
        episodes=len(results),
        success_rate=float(np.mean([result.success for result in results])),
        mean_actual_makespan=float(np.mean([result.actual_makespan for result in results])),
        action_divergence_rate=(divergent / compared) if compared else 0.0,
        mean_communication_edges=float(np.mean([result.communication_edges for result in results])),
        mean_receive_agent_steps=float(np.mean([result.receive_agent_steps for result in results])),
        mean_received_norm_per_decision=(sum(result.total_received_norm for result in results) / compared) if compared else 0.0,
        mean_outgoing_norm_per_decision=(sum(result.total_outgoing_norm for result in results) / compared) if compared else 0.0,
    )
