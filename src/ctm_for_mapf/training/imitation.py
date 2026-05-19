"""Supervised full-sequence imitation training for temporal recovery policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ctm_for_mapf.datasets import SequenceBatch
from ctm_for_mapf.models import AdaptiveCTMPolicyOutput, RecurrentTemporalRecoveryPolicy


@dataclass(frozen=True, slots=True)
class ImitationMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    wait_accuracy: float
    follow_accuracy: float
    confusion: tuple[tuple[int, int], tuple[int, int]]
    examples: int


@dataclass(frozen=True, slots=True)
class SequencePolicyOutput:
    """Policy outputs replayed over environment time.

    ``logits`` always stores the common final-tick logits with shape
    ``[batch, env_steps, decisions]``. Adaptive CTM policies additionally fill:

    - ``all_logits``: ``[batch, env_steps, decisions, internal_ticks]``
    - ``certainties``: ``[batch, env_steps, 2, internal_ticks]``
    """

    logits: torch.Tensor
    all_logits: torch.Tensor | None = None
    certainties: torch.Tensor | None = None

    @property
    def is_adaptive_ctm(self) -> bool:
        return self.all_logits is not None and self.certainties is not None


def sequence_policy_outputs(policy: RecurrentTemporalRecoveryPolicy, batch: SequenceBatch) -> SequencePolicyOutput:
    """Replay one padded full-sequence batch through a recurrent policy."""
    batch_size, sequence_length, _ = batch.features.shape
    state = policy.initial_state(batch_size, device=batch.features.device)
    logits_by_time: list[torch.Tensor] = []
    all_logits_by_time: list[torch.Tensor] = []
    certainties_by_time: list[torch.Tensor] = []
    adaptive_mode: bool | None = None

    for timestep in range(sequence_length):
        output = policy(batch.features[:, timestep], state)
        logits_by_time.append(output.logits)
        state = output.state

        is_adaptive_output = isinstance(output, AdaptiveCTMPolicyOutput)
        if adaptive_mode is None:
            adaptive_mode = is_adaptive_output
        elif adaptive_mode != is_adaptive_output:
            raise RuntimeError("Policy changed output structure within one sequence replay.")

        if is_adaptive_output:
            all_logits_by_time.append(output.all_logits.transpose(1, 2))
            certainties_by_time.append(output.certainties)

    logits = torch.stack(logits_by_time, dim=1)
    if adaptive_mode:
        return SequencePolicyOutput(
            logits=logits,
            all_logits=torch.stack(all_logits_by_time, dim=1),
            certainties=torch.stack(certainties_by_time, dim=1),
        )
    return SequencePolicyOutput(logits=logits)


def sequence_logits(policy: RecurrentTemporalRecoveryPolicy, batch: SequenceBatch) -> torch.Tensor:
    """Return final-tick logits over one padded full-sequence batch."""
    return sequence_policy_outputs(policy, batch).logits


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_step = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    return (per_step * mask.float()).sum() / mask.sum().clamp_min(1)


def masked_ctm_style_cross_entropy(
    all_logits: torch.Tensor,
    certainties: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    use_most_certain: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official-style CTM classification loss over valid environment timesteps."""
    if all_logits.ndim != 4:
        raise ValueError(f"Expected all_logits with shape [batch, env_steps, classes, internal_ticks], got {tuple(all_logits.shape)}.")
    if certainties.ndim != 4:
        raise ValueError(f"Expected certainties with shape [batch, env_steps, 2, internal_ticks], got {tuple(certainties.shape)}.")

    valid_logits = all_logits[mask]
    valid_certainties = certainties[mask]
    valid_labels = labels[mask]
    if valid_labels.numel() == 0:
        raise ValueError("Cannot compute CTM-style loss on zero valid examples.")

    targets_expanded = valid_labels.unsqueeze(-1).expand(-1, valid_logits.shape[-1])
    per_tick_losses = F.cross_entropy(valid_logits, targets_expanded, reduction="none")
    best_tick = per_tick_losses.argmin(dim=1)
    selected_tick = valid_certainties[:, 1].argmax(dim=-1)
    if not use_most_certain:
        selected_tick = torch.full_like(selected_tick, valid_logits.shape[-1] - 1)

    batch_index = torch.arange(valid_logits.shape[0], device=valid_logits.device)
    minimum_ce = per_tick_losses[batch_index, best_tick].mean()
    selected_ce = per_tick_losses[batch_index, selected_tick].mean()
    return (minimum_ce + selected_ce) / 2, selected_tick


def masked_imitation_loss(outputs: SequencePolicyOutput, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if outputs.is_adaptive_ctm:
        assert outputs.all_logits is not None and outputs.certainties is not None
        loss, _ = masked_ctm_style_cross_entropy(outputs.all_logits, outputs.certainties, labels, mask)
        return loss
    return masked_cross_entropy(outputs.logits, labels, mask)


def train_one_epoch(
    policy: RecurrentTemporalRecoveryPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    grad_clip_norm: float | None = None,
) -> ImitationMetrics:
    policy.train()
    for batch in loader:
        batch = batch.to(device)
        outputs = sequence_policy_outputs(policy, batch)
        loss = masked_imitation_loss(outputs, batch.labels, batch.mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
    return evaluate_offline(policy, loader, device=device)


@torch.no_grad()
def evaluate_offline(
    policy: RecurrentTemporalRecoveryPolicy,
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
        outputs = sequence_policy_outputs(policy, batch)
        loss = masked_imitation_loss(outputs, batch.labels, batch.mask)
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
