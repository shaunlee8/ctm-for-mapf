"""Supervised imitation training for joint communicating CTM policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ctm_for_mapf.datasets import JointSequenceBatch
from ctm_for_mapf.models import CommunicatingCTMTemporalRecoveryPolicy
from ctm_for_mapf.training.imitation import ImitationMetrics


@dataclass(frozen=True, slots=True)
class JointSequencePolicyOutput:
    """Communicating policy outputs replayed over environment time."""

    logits: torch.Tensor
    outgoing_messages: torch.Tensor
    received_messages: torch.Tensor


def joint_sequence_policy_outputs(
    policy: CommunicatingCTMTemporalRecoveryPolicy,
    batch: JointSequenceBatch,
    *,
    message_mode: str = "learned",
) -> JointSequencePolicyOutput:
    """Replay a padded joint episode batch through one-step-delayed communication."""
    batch_size, sequence_length, num_agents, _ = batch.features.shape
    state = policy.initial_joint_state(batch_size, num_agents, device=batch.features.device)
    communication_state = policy.initial_communication_state(batch_size, num_agents, device=batch.features.device)
    logits_by_time: list[torch.Tensor] = []
    outgoing_by_time: list[torch.Tensor] = []
    received_by_time: list[torch.Tensor] = []

    for timestep in range(sequence_length):
        present_agents = batch.presence_mask[:, timestep]
        output = policy.forward_joint(
            batch.features[:, timestep],
            batch.positions[:, timestep],
            state,
            communication_state,
            agent_mask=present_agents,
            message_mode=message_mode,
        )
        logits_by_time.append(output.logits)
        outgoing_by_time.append(output.outgoing_messages)
        received_by_time.append(output.received_messages)
        state = output.state
        communication_state = output.next_communication_state

    return JointSequencePolicyOutput(
        logits=torch.stack(logits_by_time, dim=1),
        outgoing_messages=torch.stack(outgoing_by_time, dim=1),
        received_messages=torch.stack(received_by_time, dim=1),
    )


def masked_joint_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"Expected logits with shape [batch, time, agents, decisions], got {tuple(logits.shape)}.")
    valid_logits = logits[mask]
    valid_labels = labels[mask]
    if valid_labels.numel() == 0:
        raise ValueError("Cannot compute joint imitation loss on zero valid examples.")
    return F.cross_entropy(valid_logits, valid_labels)


def _metrics_from_outputs(
    outputs: JointSequencePolicyOutput,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_logits = outputs.logits[mask]
    valid_labels = labels[mask]
    return valid_logits, valid_labels


def train_one_joint_epoch(
    policy: CommunicatingCTMTemporalRecoveryPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    grad_clip_norm: float | None = None,
) -> ImitationMetrics:
    policy.train()
    for batch in loader:
        batch = batch.to(device)
        outputs = joint_sequence_policy_outputs(policy, batch)
        loss = masked_joint_cross_entropy(outputs.logits, batch.labels, batch.label_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
    return evaluate_joint_offline(policy, loader, device=device)


@torch.no_grad()
def evaluate_joint_offline(
    policy: CommunicatingCTMTemporalRecoveryPolicy,
    loader: DataLoader,
    *,
    device: torch.device | str,
    message_mode: str = "learned",
) -> ImitationMetrics:
    policy.eval()
    total_loss = 0.0
    total_examples = 0
    confusion = torch.zeros((2, 2), dtype=torch.long)

    for batch in loader:
        batch = batch.to(device)
        outputs = joint_sequence_policy_outputs(policy, batch, message_mode=message_mode)
        loss = masked_joint_cross_entropy(outputs.logits, batch.labels, batch.label_mask)
        valid_logits, valid_labels = _metrics_from_outputs(outputs, batch.labels, batch.label_mask)
        predictions = valid_logits.argmax(dim=-1)
        for label, prediction in zip(valid_labels.cpu(), predictions.cpu()):
            confusion[int(label), int(prediction)] += 1
        count = int(batch.label_mask.sum().item())
        total_loss += float(loss.item()) * count
        total_examples += count

    if total_examples == 0:
        raise ValueError("Cannot evaluate on zero examples.")
    wait_total = int(confusion[0].sum().item())
    follow_total = int(confusion[1].sum().item())
    wait_accuracy = float(confusion[0, 0].item() / wait_total) if wait_total else 0.0
    follow_accuracy = float(confusion[1, 1].item() / follow_total) if follow_total else 0.0
    accuracy = float(confusion.diag().sum().item() / total_examples)
    balanced_accuracy = (wait_accuracy + follow_accuracy) / 2
    return ImitationMetrics(
        loss=total_loss / total_examples,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        wait_accuracy=wait_accuracy,
        follow_accuracy=follow_accuracy,
        confusion=(
            (int(confusion[0, 0].item()), int(confusion[0, 1].item())),
            (int(confusion[1, 0].item()), int(confusion[1, 1].item())),
        ),
        examples=total_examples,
    )
