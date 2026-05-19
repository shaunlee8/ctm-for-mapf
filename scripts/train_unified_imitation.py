"""Train the unified CTM MAPF recovery policy on two-step imitation pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ctm_for_mapf.datasets import ImitationDataset, split_episode_ids
from ctm_for_mapf.models import RecoveryObservationTokenizer, UnifiedCTMRecoveryPolicy
from ctm_for_mapf.models.recovery_policy import UnifiedCTMPolicyOutput


@dataclass(frozen=True, slots=True)
class UnifiedPairSample:
    tokens_t0: torch.Tensor
    positions_t0: torch.Tensor
    labels_t0: torch.Tensor
    tokens_t1: torch.Tensor
    positions_t1: torch.Tensor
    labels_t1: torch.Tensor


@dataclass(frozen=True, slots=True)
class UnifiedPairBatch:
    tokens_t0: torch.Tensor
    positions_t0: torch.Tensor
    labels_t0: torch.Tensor
    tokens_t1: torch.Tensor
    positions_t1: torch.Tensor
    labels_t1: torch.Tensor

    def to(self, device: torch.device | str) -> "UnifiedPairBatch":
        return UnifiedPairBatch(
            tokens_t0=self.tokens_t0.to(device),
            positions_t0=self.positions_t0.to(device),
            labels_t0=self.labels_t0.to(device),
            tokens_t1=self.tokens_t1.to(device),
            positions_t1=self.positions_t1.to(device),
            labels_t1=self.labels_t1.to(device),
        )


@dataclass(frozen=True, slots=True)
class Metrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    wait_accuracy: float
    follow_accuracy: float
    confusion: tuple[tuple[int, int], tuple[int, int]]
    examples: int


class UnifiedPairDataset(Dataset[UnifiedPairSample]):
    def __init__(self, samples: list[UnifiedPairSample]) -> None:
        if not samples:
            raise ValueError("UnifiedPairDataset requires at least one sample.")
        self.samples = samples
        self.num_agents = int(samples[0].labels_t0.shape[0])
        self.num_tokens = int(samples[0].tokens_t0.shape[-2])
        self.token_dim = int(samples[0].tokens_t0.shape[-1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> UnifiedPairSample:
        return self.samples[index]

    @property
    def label_counts(self) -> tuple[int, int]:
        labels = torch.cat([sample.labels_t0.reshape(-1) for sample in self.samples] + [sample.labels_t1.reshape(-1) for sample in self.samples])
        counts = torch.bincount(labels, minlength=2)
        return int(counts[0]), int(counts[1])

    @classmethod
    def from_imitation_dataset(
        cls,
        dataset: ImitationDataset,
        *,
        episode_ids: set[int] | None,
        num_agents: int,
        max_pairs: int | None = None,
    ) -> "UnifiedPairDataset":
        tokenizer = RecoveryObservationTokenizer()
        groups: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
        for index, (episode_id, timestep, agent_id) in enumerate(
            zip(dataset.episode_ids, dataset.timesteps, dataset.agent_ids, strict=True)
        ):
            episode_id = int(episode_id)
            if episode_ids is not None and episode_id not in episode_ids:
                continue
            groups[(episode_id, int(timestep))][int(agent_id)] = index

        samples: list[UnifiedPairSample] = []
        for episode_id, timestep in sorted(groups):
            first = groups[(episode_id, timestep)]
            second = groups.get((episode_id, timestep + 1))
            if second is None:
                continue
            common_agent_ids = sorted(set(first).intersection(second))
            if len(common_agent_ids) < num_agents:
                continue
            chosen = common_agent_ids[:num_agents]
            first_indices = [first[agent_id] for agent_id in chosen]
            second_indices = [second[agent_id] for agent_id in chosen]
            first_observations = [_row_observation(dataset, index) for index in first_indices]
            second_observations = [_row_observation(dataset, index) for index in second_indices]
            samples.append(
                UnifiedPairSample(
                    tokens_t0=torch.as_tensor(tokenizer.batch_numpy(first_observations), dtype=torch.float32),
                    positions_t0=torch.as_tensor(dataset.xy[first_indices], dtype=torch.long),
                    labels_t0=torch.as_tensor(dataset.labels[first_indices], dtype=torch.long),
                    tokens_t1=torch.as_tensor(tokenizer.batch_numpy(second_observations), dtype=torch.float32),
                    positions_t1=torch.as_tensor(dataset.xy[second_indices], dtype=torch.long),
                    labels_t1=torch.as_tensor(dataset.labels[second_indices], dtype=torch.long),
                )
            )
            if max_pairs is not None and len(samples) >= max_pairs:
                break
        return cls(samples)


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


def collate_unified_pairs(samples: list[UnifiedPairSample]) -> UnifiedPairBatch:
    if not samples:
        raise ValueError("Cannot collate an empty unified pair batch.")
    return UnifiedPairBatch(
        tokens_t0=torch.stack([sample.tokens_t0 for sample in samples], dim=0),
        positions_t0=torch.stack([sample.positions_t0 for sample in samples], dim=0),
        labels_t0=torch.stack([sample.labels_t0 for sample in samples], dim=0),
        tokens_t1=torch.stack([sample.tokens_t1 for sample in samples], dim=0),
        positions_t1=torch.stack([sample.positions_t1 for sample in samples], dim=0),
        labels_t1=torch.stack([sample.labels_t1 for sample in samples], dim=0),
    )


def _class_weights(dataset: UnifiedPairDataset, *, device: torch.device | str) -> torch.Tensor:
    wait_count, follow_count = dataset.label_counts
    counts = torch.tensor([wait_count, follow_count], dtype=torch.float32, device=device).clamp_min(1.0)
    weights = counts.sum() / (2.0 * counts)
    return weights / weights.mean()


def _ctm_style_loss(
    output: UnifiedCTMPolicyOutput,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    use_most_certain: bool = True,
) -> torch.Tensor:
    logits = output.all_logits.reshape(-1, output.all_logits.shape[2], 2).transpose(1, 2)
    certainties = output.certainties.reshape(-1, 2, output.certainties.shape[-1])
    targets = labels.reshape(-1)
    expanded_targets = targets.unsqueeze(-1).expand(-1, logits.shape[-1])
    per_tick_losses = F.cross_entropy(logits, expanded_targets, weight=class_weights, reduction="none")
    best_tick = per_tick_losses.argmin(dim=1)
    selected_tick = certainties[:, 1].argmax(dim=-1)
    if not use_most_certain:
        selected_tick = torch.full_like(selected_tick, logits.shape[-1] - 1)
    example_index = torch.arange(logits.shape[0], device=logits.device)
    return (per_tick_losses[example_index, best_tick].mean() + per_tick_losses[example_index, selected_tick].mean()) / 2.0


def _final_loss(output: UnifiedCTMPolicyOutput, labels: torch.Tensor, *, class_weights: torch.Tensor | None) -> torch.Tensor:
    return F.cross_entropy(output.logits.reshape(-1, 2), labels.reshape(-1), weight=class_weights)


def _loss(
    output: UnifiedCTMPolicyOutput,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    loss_mode: str,
    ctm_loss_weight: float,
) -> torch.Tensor:
    if loss_mode == "final":
        return _final_loss(output, labels, class_weights=class_weights)
    if loss_mode == "ctm":
        return _ctm_style_loss(output, labels, class_weights=class_weights)
    if loss_mode == "hybrid":
        return _final_loss(output, labels, class_weights=class_weights) + ctm_loss_weight * _ctm_style_loss(
            output,
            labels,
            class_weights=class_weights,
        )
    raise ValueError("loss_mode must be one of {'final', 'ctm', 'hybrid'}.")


def _forward_pair(policy: UnifiedCTMRecoveryPolicy, batch: UnifiedPairBatch) -> tuple[UnifiedCTMPolicyOutput, UnifiedCTMPolicyOutput]:
    batch_size, num_agents = batch.labels_t0.shape
    state = policy.initial_joint_state(batch_size, num_agents, device=batch.tokens_t0.device)
    communication_state = policy.initial_communication_state(batch_size, num_agents, device=batch.tokens_t0.device)
    output_t0 = policy.forward_joint_tokens(batch.tokens_t0, batch.positions_t0, state, communication_state)
    assert output_t0.next_communication_state is not None
    output_t1 = policy.forward_joint_tokens(batch.tokens_t1, batch.positions_t1, output_t0.state, output_t0.next_communication_state)
    return output_t0, output_t1


def _update_confusion(confusion: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
    predictions = logits.argmax(dim=-1)
    for label, prediction in zip(labels.reshape(-1).detach().cpu(), predictions.reshape(-1).detach().cpu()):
        confusion[int(label), int(prediction)] += 1


def _metrics_from_confusion(loss_sum: float, examples: int, confusion: torch.Tensor) -> Metrics:
    wait_total = int(confusion[0].sum().item())
    follow_total = int(confusion[1].sum().item())
    wait_accuracy = float(confusion[0, 0].item() / wait_total) if wait_total else 0.0
    follow_accuracy = float(confusion[1, 1].item() / follow_total) if follow_total else 0.0
    accuracy = float(confusion.diag().sum().item() / examples) if examples else 0.0
    balanced_accuracy = (wait_accuracy + follow_accuracy) / 2.0
    return Metrics(
        loss=loss_sum / max(examples, 1),
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        wait_accuracy=wait_accuracy,
        follow_accuracy=follow_accuracy,
        confusion=(
            (int(confusion[0, 0].item()), int(confusion[0, 1].item())),
            (int(confusion[1, 0].item()), int(confusion[1, 1].item())),
        ),
        examples=examples,
    )


def train_one_epoch(
    policy: UnifiedCTMRecoveryPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    class_weights: torch.Tensor | None,
    loss_mode: str,
    ctm_loss_weight: float,
    grad_clip_norm: float | None,
) -> Metrics:
    policy.train()
    for batch in loader:
        batch = batch.to(device)
        output_t0, output_t1 = _forward_pair(policy, batch)
        loss_t0 = _loss(output_t0, batch.labels_t0, class_weights=class_weights, loss_mode=loss_mode, ctm_loss_weight=ctm_loss_weight)
        loss_t1 = _loss(output_t1, batch.labels_t1, class_weights=class_weights, loss_mode=loss_mode, ctm_loss_weight=ctm_loss_weight)
        loss = loss_t0 + loss_t1
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
    return evaluate(policy, loader, device=device, class_weights=class_weights, loss_mode=loss_mode, ctm_loss_weight=ctm_loss_weight)


@torch.no_grad()
def evaluate(
    policy: UnifiedCTMRecoveryPolicy,
    loader: DataLoader,
    *,
    device: torch.device | str,
    class_weights: torch.Tensor | None,
    loss_mode: str,
    ctm_loss_weight: float,
) -> Metrics:
    policy.eval()
    loss_sum = 0.0
    examples = 0
    confusion = torch.zeros((2, 2), dtype=torch.long)
    for batch in loader:
        batch = batch.to(device)
        output_t0, output_t1 = _forward_pair(policy, batch)
        loss_t0 = _loss(output_t0, batch.labels_t0, class_weights=class_weights, loss_mode=loss_mode, ctm_loss_weight=ctm_loss_weight)
        loss_t1 = _loss(output_t1, batch.labels_t1, class_weights=class_weights, loss_mode=loss_mode, ctm_loss_weight=ctm_loss_weight)
        count = int(batch.labels_t0.numel() + batch.labels_t1.numel())
        loss_sum += float((loss_t0 + loss_t1).item()) * count
        examples += count
        _update_confusion(confusion, output_t0.logits, batch.labels_t0)
        _update_confusion(confusion, output_t1.logits, batch.labels_t1)
    if examples == 0:
        raise ValueError("Cannot evaluate on zero examples.")
    return _metrics_from_confusion(loss_sum, examples, confusion)


def _metrics_dict(metrics: Metrics) -> dict:
    return {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "wait_accuracy": metrics.wait_accuracy,
        "follow_accuracy": metrics.follow_accuracy,
        "confusion": metrics.confusion,
        "examples": metrics.examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the unified CTM MAPF recovery policy.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to an imitation .npz dataset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and training metadata.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--loss-mode", choices=("final", "ctm", "hybrid"), default="hybrid")
    parser.add_argument("--ctm-loss-weight", type=float, default=0.5)
    parser.add_argument("--no-balanced-loss", action="store_true")

    parser.add_argument("--ctm-iterations", type=int, default=4)
    parser.add_argument("--ctm-d-model", type=int, default=64)
    parser.add_argument("--ctm-d-input", type=int, default=32)
    parser.add_argument("--ctm-heads", type=int, default=4)
    parser.add_argument("--ctm-n-synch-out", type=int, default=8)
    parser.add_argument("--ctm-n-synch-action", type=int, default=8)
    parser.add_argument("--ctm-message-dim", type=int, default=8)
    parser.add_argument("--ctm-communication-radius", type=int, default=3)
    parser.add_argument("--ctm-memory-length", type=int, default=5)
    parser.add_argument("--ctm-memory-hidden-dims", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = ImitationDataset.load(args.dataset)
    train_episode_ids, val_episode_ids = split_episode_ids(dataset, train_fraction=args.train_fraction, seed=args.split_seed)
    train_dataset = UnifiedPairDataset.from_imitation_dataset(
        dataset,
        episode_ids=train_episode_ids,
        num_agents=args.num_agents,
        max_pairs=args.max_train_pairs,
    )
    val_dataset = UnifiedPairDataset.from_imitation_dataset(
        dataset,
        episode_ids=val_episode_ids,
        num_agents=args.num_agents,
        max_pairs=args.max_val_pairs,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_unified_pairs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_unified_pairs)

    model_config = {
        "token_dim": train_dataset.token_dim,
        "message_dim": args.ctm_message_dim,
        "communication_radius": args.ctm_communication_radius,
        "iterations": args.ctm_iterations,
        "d_model": args.ctm_d_model,
        "d_input": args.ctm_d_input,
        "heads": args.ctm_heads,
        "n_synch_out": args.ctm_n_synch_out,
        "n_synch_action": args.ctm_n_synch_action,
        "memory_length": args.ctm_memory_length,
        "memory_hidden_dims": args.ctm_memory_hidden_dims,
    }
    policy = UnifiedCTMRecoveryPolicy(**model_config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    class_weights = None if args.no_balanced_loss else _class_weights(train_dataset, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = args.output_dir / "best.pt"
    latest_checkpoint_path = args.output_dir / "latest.pt"
    history_path = args.output_dir / "history.json"
    config_path = args.output_dir / "config.json"

    run_config = {
        "dataset": str(args.dataset),
        "train_episode_ids": sorted(train_episode_ids),
        "val_episode_ids": sorted(val_episode_ids),
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "train_label_counts": train_dataset.label_counts,
        "val_label_counts": val_dataset.label_counts,
        "num_agents": args.num_agents,
        "num_tokens": train_dataset.num_tokens,
        "token_dim": train_dataset.token_dim,
        "loss_mode": args.loss_mode,
        "balanced_loss": not args.no_balanced_loss,
        "class_weights": None if class_weights is None else [float(x) for x in class_weights.detach().cpu()],
        "model_config": model_config,
    }
    config_path.write_text(json.dumps(run_config, indent=2))

    print(f"dataset={args.dataset}")
    print(f"output_dir={args.output_dir}")
    print(f"train_pairs={len(train_dataset)} val_pairs={len(val_dataset)}")
    print(f"train_label_counts={train_dataset.label_counts} val_label_counts={val_dataset.label_counts}")
    print(f"num_agents={args.num_agents} num_tokens={train_dataset.num_tokens} token_dim={train_dataset.token_dim}")
    print(f"loss_mode={args.loss_mode} balanced_loss={not args.no_balanced_loss} class_weights={run_config['class_weights']}")

    best_val_loss = float("inf")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            policy,
            train_loader,
            optimizer,
            device=device,
            class_weights=class_weights,
            loss_mode=args.loss_mode,
            ctm_loss_weight=args.ctm_loss_weight,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = evaluate(
            policy,
            val_loader,
            device=device,
            class_weights=class_weights,
            loss_mode=args.loss_mode,
            ctm_loss_weight=args.ctm_loss_weight,
        )
        row = {"epoch": epoch, "train": _metrics_dict(train_metrics), "val": _metrics_dict(val_metrics)}
        history.append(row)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.3f} train_bal_acc={train_metrics.balanced_accuracy:.3f} "
            f"train_wait={train_metrics.wait_accuracy:.3f} train_follow={train_metrics.follow_accuracy:.3f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.3f} val_bal_acc={val_metrics.balanced_accuracy:.3f} "
            f"val_wait={val_metrics.wait_accuracy:.3f} val_follow={val_metrics.follow_accuracy:.3f}"
        )
        checkpoint = {
            "model_type": "unified_ctm",
            "model_config": model_config,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "run_config": run_config,
        }
        torch.save(checkpoint, latest_checkpoint_path)
        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            torch.save(checkpoint, best_checkpoint_path)
        history_path.write_text(json.dumps({"history": history, "best_val_loss": best_val_loss}, indent=2))

    print(f"best_checkpoint={best_checkpoint_path}")
    print(f"latest_checkpoint={latest_checkpoint_path}")
    print(f"history={history_path}")


if __name__ == "__main__":
    main()
