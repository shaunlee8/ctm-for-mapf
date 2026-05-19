"""Non-benchmark training check for the unified CTM MAPF recovery policy.

This verifies Milestone 7E: the unified model is not only runnable, but has a
fixed-budget differentiable path that can train action attention and one-step-delayed
communication through a short two-step unroll.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ctm_for_mapf.datasets import ImitationDataset
from ctm_for_mapf.models import RecoveryObservationTokenizer, UnifiedCTMRecoveryPolicy
from ctm_for_mapf.models.recovery_policy import UnifiedCTMPolicyOutput


@dataclass(frozen=True, slots=True)
class JointPairBatch:
    tokens_t0: torch.Tensor
    positions_t0: torch.Tensor
    labels_t0: torch.Tensor
    tokens_t1: torch.Tensor
    positions_t1: torch.Tensor
    labels_t1: torch.Tensor


@dataclass(frozen=True, slots=True)
class LossParts:
    loss: torch.Tensor
    final_loss: torch.Tensor
    all_tick_loss: torch.Tensor
    accuracy: torch.Tensor


def _row_observation(dataset: ImitationDataset, index: int) -> dict:
    return {
        "obstacles": dataset.obstacles[index],
        "agents": dataset.agents[index],
        "xy": dataset.xy[index],
        "target_xy": dataset.target_xy[index],
        "planned_next_action": dataset.planned_next_action[index],
        "planned_next_delta": dataset.planned_next_delta[index],
        "scheduled_next_action": dataset.scheduled_next_action[index],
        "lateness": dataset.lateness[index],
        "off_plan": dataset.off_plan[index],
        "remaining_plan_length": dataset.remaining_plan_length[index],
        "was_delayed_last_step": dataset.was_delayed_last_step[index],
    }


def _pair_groups(dataset: ImitationDataset, *, num_agents: int, batch_size: int) -> list[tuple[list[int], list[int]]]:
    groups: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for index, (episode_id, timestep, agent_id) in enumerate(
        zip(dataset.episode_ids, dataset.timesteps, dataset.agent_ids, strict=True)
    ):
        groups[(int(episode_id), int(timestep))][int(agent_id)] = index

    selected: list[tuple[list[int], list[int]]] = []
    for episode_id, timestep in sorted(groups):
        first = groups[(episode_id, timestep)]
        second = groups.get((episode_id, timestep + 1))
        if second is None:
            continue
        common_agent_ids = sorted(set(first).intersection(second))
        if len(common_agent_ids) < num_agents:
            continue
        chosen = common_agent_ids[:num_agents]
        selected.append(([first[agent_id] for agent_id in chosen], [second[agent_id] for agent_id in chosen]))
        if len(selected) >= batch_size:
            break
    if not selected:
        raise RuntimeError(f"No consecutive joint timestep pairs with at least {num_agents} active agents were found.")
    return selected


def _build_pair_batch(dataset: ImitationDataset, *, num_agents: int, batch_size: int, device: torch.device) -> JointPairBatch:
    tokenizer = RecoveryObservationTokenizer()
    pairs = _pair_groups(dataset, num_agents=num_agents, batch_size=batch_size)
    tokens_t0: list[np.ndarray] = []
    positions_t0: list[np.ndarray] = []
    labels_t0: list[np.ndarray] = []
    tokens_t1: list[np.ndarray] = []
    positions_t1: list[np.ndarray] = []
    labels_t1: list[np.ndarray] = []
    for first_indices, second_indices in pairs:
        first_observations = [_row_observation(dataset, index) for index in first_indices]
        second_observations = [_row_observation(dataset, index) for index in second_indices]
        tokens_t0.append(tokenizer.batch_numpy(first_observations))
        tokens_t1.append(tokenizer.batch_numpy(second_observations))
        positions_t0.append(dataset.xy[first_indices])
        positions_t1.append(dataset.xy[second_indices])
        labels_t0.append(dataset.labels[first_indices])
        labels_t1.append(dataset.labels[second_indices])

    return JointPairBatch(
        tokens_t0=torch.as_tensor(np.stack(tokens_t0, axis=0), dtype=torch.float32, device=device),
        positions_t0=torch.as_tensor(np.stack(positions_t0, axis=0), dtype=torch.long, device=device),
        labels_t0=torch.as_tensor(np.stack(labels_t0, axis=0), dtype=torch.long, device=device),
        tokens_t1=torch.as_tensor(np.stack(tokens_t1, axis=0), dtype=torch.float32, device=device),
        positions_t1=torch.as_tensor(np.stack(positions_t1, axis=0), dtype=torch.long, device=device),
        labels_t1=torch.as_tensor(np.stack(labels_t1, axis=0), dtype=torch.long, device=device),
    )


def _loss(output: UnifiedCTMPolicyOutput, labels: torch.Tensor, *, all_tick_weight: float) -> LossParts:
    final_loss = F.cross_entropy(output.logits.reshape(-1, 2), labels.reshape(-1))
    repeated_labels = labels.unsqueeze(-1).expand(*labels.shape, output.all_logits.shape[2])
    all_tick_loss = F.cross_entropy(output.all_logits.reshape(-1, 2), repeated_labels.reshape(-1))
    loss = final_loss + all_tick_weight * all_tick_loss
    accuracy = (output.logits.argmax(dim=-1) == labels).float().mean()
    return LossParts(loss=loss, final_loss=final_loss.detach(), all_tick_loss=all_tick_loss.detach(), accuracy=accuracy.detach())


def _grad_norm(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.detach().norm().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to an imitation .npz dataset.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--all-tick-weight", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-input", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--n-synch-out", type=int, default=8)
    parser.add_argument("--n-synch-action", type=int, default=8)
    parser.add_argument("--message-dim", type=int, default=8)
    parser.add_argument("--communication-radius", type=int, default=3)
    parser.add_argument("--memory-length", type=int, default=5)
    parser.add_argument("--memory-hidden-dims", type=int, default=16)
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.num_agents < 1:
        raise ValueError("--num-agents must be >= 1.")

    device = torch.device(args.device)
    dataset = ImitationDataset.load(args.dataset)
    batch = _build_pair_batch(dataset, num_agents=args.num_agents, batch_size=args.batch_size, device=device)
    batch_size, num_agents, num_tokens, token_dim = batch.tokens_t0.shape

    policy = UnifiedCTMRecoveryPolicy(
        token_dim=token_dim,
        message_dim=args.message_dim,
        communication_radius=args.communication_radius,
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        memory_length=args.memory_length,
        memory_hidden_dims=args.memory_hidden_dims,
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    print(f"dataset={args.dataset}")
    print(f"joint_pair_batch={batch_size}x{num_agents} pair_agent_observations={batch_size * num_agents * 2}")
    print(f"tokens_t0_shape={tuple(batch.tokens_t0.shape)} tokens_t1_shape={tuple(batch.tokens_t1.shape)}")
    print(f"labels_t0_hist={torch.bincount(batch.labels_t0.reshape(-1).detach().cpu(), minlength=2).tolist()}")
    print(f"labels_t1_hist={torch.bincount(batch.labels_t1.reshape(-1).detach().cpu(), minlength=2).tolist()}")

    for step in range(1, args.steps + 1):
        policy.train()
        state = policy.initial_joint_state(batch_size, num_agents, device=device)
        communication_state = policy.initial_communication_state(batch_size, num_agents, device=device)

        output_t0 = policy.forward_joint_tokens(batch.tokens_t0, batch.positions_t0, state, communication_state)
        assert output_t0.next_communication_state is not None
        output_t1 = policy.forward_joint_tokens(batch.tokens_t1, batch.positions_t1, output_t0.state, output_t0.next_communication_state)

        loss_t0 = _loss(output_t0, batch.labels_t0, all_tick_weight=args.all_tick_weight)
        loss_t1 = _loss(output_t1, batch.labels_t1, all_tick_weight=args.all_tick_weight)
        loss = loss_t0.loss + loss_t1.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_action = _grad_norm(policy.action_head.weight)
        grad_q = _grad_norm(policy.core.q_proj.weight)
        grad_message = _grad_norm(policy.message_head.weight)
        grad_message_token = _grad_norm(policy.message_token_projector.weight)
        optimizer.step()

        print(
            f"step={step} "
            f"loss={float(loss.detach().cpu()):.4f} "
            f"t0_acc={float(loss_t0.accuracy.cpu()):.3f} "
            f"t1_acc={float(loss_t1.accuracy.cpu()):.3f} "
            f"grad_action={grad_action:.3e} "
            f"grad_q={grad_q:.3e} "
            f"grad_message={grad_message:.3e} "
            f"grad_message_token={grad_message_token:.3e}"
        )

    policy.eval()
    with torch.inference_mode():
        state = policy.initial_joint_state(batch_size, num_agents, device=device)
        communication_state = policy.initial_communication_state(batch_size, num_agents, device=device)
        output = policy.forward_joint_tokens(batch.tokens_t0, batch.positions_t0, state, communication_state)
    assert output.attention_weights is not None
    assert output.outgoing_messages is not None
    assert output.next_communication_state is not None
    print(f"post_train_attention_shape={tuple(output.attention_weights.shape)}")
    print(f"post_train_messages_shape={tuple(output.outgoing_messages.shape)}")
    print(f"post_train_next_received_shape={tuple(output.next_communication_state.received_messages.shape)}")


if __name__ == "__main__":
    main()
