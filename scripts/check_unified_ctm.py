"""Check the unified CTM MAPF recovery policy on joint imitation batches.

This script is intentionally not a benchmark. It verifies that the required milestone-7D
mechanics can run at realistic batch sizes:

- tokenized local observations with a received-message token,
- CTM action-synchronization cross-attention,
- per-tick logits/certainties for train-time losses,
- synchronization-derived outgoing messages routed over a radius graph,
- adaptive inference with early stopping.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ctm_for_mapf.datasets import ImitationDataset
from ctm_for_mapf.models import RecoveryObservationTokenizer, UnifiedCTMRecoveryPolicy


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


def _joint_groups(dataset: ImitationDataset, *, num_agents: int, batch_size: int) -> list[list[int]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (episode_id, timestep) in enumerate(zip(dataset.episode_ids, dataset.timesteps, strict=True)):
        groups[(int(episode_id), int(timestep))].append(index)

    selected: list[list[int]] = []
    for _, indices in sorted(groups.items()):
        indices = sorted(indices, key=lambda idx: int(dataset.agent_ids[idx]))
        if len(indices) >= num_agents:
            selected.append(indices[:num_agents])
        if len(selected) >= batch_size:
            break
    if not selected:
        raise RuntimeError(f"No joint timesteps with at least {num_agents} active agents were found.")
    return selected


def _build_joint_batch(dataset: ImitationDataset, *, num_agents: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = RecoveryObservationTokenizer()
    groups = _joint_groups(dataset, num_agents=num_agents, batch_size=batch_size)
    token_batches = []
    position_batches = []
    for indices in groups:
        observations = [_row_observation(dataset, index) for index in indices]
        token_batches.append(tokenizer.batch_numpy(observations))
        position_batches.append(dataset.xy[indices])
    tokens = torch.as_tensor(np.stack(token_batches, axis=0), dtype=torch.float32, device=device)
    positions = torch.as_tensor(np.stack(position_batches, axis=0), dtype=torch.long, device=device)
    return tokens, positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to an imitation .npz dataset.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("--adaptive-threshold", type=float, default=0.0)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.num_agents < 1:
        raise ValueError("--num-agents must be >= 1.")

    device = torch.device(args.device)
    dataset = ImitationDataset.load(args.dataset)
    tokens, positions = _build_joint_batch(dataset, num_agents=args.num_agents, batch_size=args.batch_size, device=device)
    batch_size, num_agents, num_tokens, token_dim = tokens.shape

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
    policy.eval()
    state = policy.initial_joint_state(batch_size, num_agents, device=device)
    communication_state = policy.initial_communication_state(batch_size, num_agents, device=device)

    with torch.inference_mode():
        fixed = policy.forward_joint_tokens(tokens, positions, state, communication_state)
        adaptive = policy.act_joint_adaptive_tokens(
            tokens,
            positions,
            state,
            communication_state,
            certainty_threshold=args.adaptive_threshold,
            min_ticks=1,
            max_ticks=args.iterations,
        )
        fixed_budget = policy.act_joint_adaptive_tokens(
            tokens,
            positions,
            state,
            communication_state,
            certainty_threshold=None,
            min_ticks=1,
            max_ticks=min(2, args.iterations),
        )

    assert fixed.attention_weights is not None
    assert fixed.outgoing_messages is not None
    assert fixed.next_communication_state is not None
    assert fixed.neighbor_mask is not None

    neighbor_edges = fixed.neighbor_mask.sum().item()
    possible_edges = fixed.neighbor_mask.numel()
    print(f"dataset={args.dataset}")
    print(f"joint_batch={batch_size}x{num_agents} agent_observations={batch_size * num_agents}")
    print(f"tokens_shape={tuple(tokens.shape)} positions_shape={tuple(positions.shape)}")
    print(f"fixed_logits_shape={tuple(fixed.logits.shape)} all_logits_shape={tuple(fixed.all_logits.shape)}")
    print(f"certainties_shape={tuple(fixed.certainties.shape)} attention_shape={tuple(fixed.attention_weights.shape)}")
    print(f"messages_shape={tuple(fixed.outgoing_messages.shape)} next_received_shape={tuple(fixed.next_communication_state.received_messages.shape)}")
    print(f"neighbor_edges={int(neighbor_edges)}/{possible_edges}")
    print(
        "adaptive_ticks="
        f"min:{int(adaptive.ticks_used.min())} "
        f"mean:{float(adaptive.ticks_used.float().mean()):.2f} "
        f"max:{int(adaptive.ticks_used.max())}"
    )
    print(f"fixed_budget_ticks_unique={fixed_budget.ticks_used.unique(sorted=True).detach().cpu().tolist()}")
    print(f"fixed_decision_hist={torch.bincount(fixed.logits.argmax(dim=-1).reshape(-1).detach().cpu(), minlength=2).tolist()}")
    print(f"adaptive_decision_hist={torch.bincount(adaptive.decisions.reshape(-1).detach().cpu(), minlength=2).tolist()}")


if __name__ == "__main__":
    main()
