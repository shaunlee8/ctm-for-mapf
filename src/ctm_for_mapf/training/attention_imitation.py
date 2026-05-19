"""Supervised imitation training for tokenized attention CTM policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ctm_for_mapf.datasets import TokenSequenceBatch
from ctm_for_mapf.models import AttentionCTMPolicyOutput, AttentionCTMTemporalRecoveryPolicy
from ctm_for_mapf.training.imitation import ImitationMetrics, masked_ctm_style_cross_entropy


@dataclass(frozen=True, slots=True)
class TokenSequencePolicyOutput:
    logits: torch.Tensor
    all_logits: torch.Tensor
    certainties: torch.Tensor
    action_latents: torch.Tensor
    attention_weights: torch.Tensor | None


def token_sequence_policy_outputs(
    policy: AttentionCTMTemporalRecoveryPolicy,
    batch: TokenSequenceBatch,
) -> TokenSequencePolicyOutput:
    batch_size, sequence_length, _, _ = batch.tokens.shape
    state = policy.initial_state(batch_size, device=batch.tokens.device)
    logits_by_time: list[torch.Tensor] = []
    all_logits_by_time: list[torch.Tensor] = []
    certainties_by_time: list[torch.Tensor] = []
    action_latents_by_time: list[torch.Tensor] = []
    attention_by_time: list[torch.Tensor] = []

    for timestep in range(sequence_length):
        output = policy(batch.tokens[:, timestep], state)
        if not isinstance(output, AttentionCTMPolicyOutput):
            raise TypeError("Attention token sequence training requires AttentionCTMPolicyOutput.")
        logits_by_time.append(output.logits)
        all_logits_by_time.append(output.all_logits.transpose(1, 2))
        certainties_by_time.append(output.certainties)
        action_latents_by_time.append(output.action_latents)
        if output.attention_weights is not None:
            attention_by_time.append(output.attention_weights)
        state = output.state

    return TokenSequencePolicyOutput(
        logits=torch.stack(logits_by_time, dim=1),
        all_logits=torch.stack(all_logits_by_time, dim=1),
        certainties=torch.stack(certainties_by_time, dim=1),
        action_latents=torch.stack(action_latents_by_time, dim=1),
        attention_weights=(torch.stack(attention_by_time, dim=1) if attention_by_time else None),
    )


def masked_attention_imitation_loss(
    outputs: TokenSequencePolicyOutput,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss, _ = masked_ctm_style_cross_entropy(outputs.all_logits, outputs.certainties, labels, mask)
    return loss


def train_one_attention_epoch(
    policy: AttentionCTMTemporalRecoveryPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    grad_clip_norm: float | None = None,
) -> ImitationMetrics:
    policy.train()
    for batch in loader:
        batch = batch.to(device)
        outputs = token_sequence_policy_outputs(policy, batch)
        loss = masked_attention_imitation_loss(outputs, batch.labels, batch.mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
    return evaluate_attention_offline(policy, loader, device=device)


@torch.no_grad()
def evaluate_attention_offline(
    policy: AttentionCTMTemporalRecoveryPolicy,
    loader: DataLoader,
    *,
    device: torch.device | str,
) -> ImitationMetrics:
    policy.eval()
    total_loss = 0.0
    total_examples = 0
    confusion = torch.zeros((2, 2), dtype=torch.long)

    for batch in loader:
        batch = batch.to(device)
        outputs = token_sequence_policy_outputs(policy, batch)
        loss = masked_attention_imitation_loss(outputs, batch.labels, batch.mask)
        valid_logits = outputs.logits[batch.mask]
        valid_labels = batch.labels[batch.mask]
        predictions = valid_logits.argmax(dim=-1)
        for label, prediction in zip(valid_labels.cpu(), predictions.cpu()):
            confusion[int(label), int(prediction)] += 1
        count = int(batch.mask.sum().item())
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
