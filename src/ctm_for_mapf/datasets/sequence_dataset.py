"""Full per-agent sequence views over flat imitation datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ctm_for_mapf.datasets.imitation_rollouts import ImitationDataset
from ctm_for_mapf.models.observation_vectorizer import RecoveryObservationVectorizer


@dataclass(frozen=True, slots=True)
class AgentSequence:
    features: torch.Tensor
    labels: torch.Tensor
    physical_actions: torch.Tensor
    episode_id: int
    agent_id: int
    timesteps: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    features: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    episode_ids: torch.Tensor
    agent_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "SequenceBatch":
        return SequenceBatch(
            features=self.features.to(device),
            labels=self.labels.to(device),
            mask=self.mask.to(device),
            lengths=self.lengths.to(device),
            episode_ids=self.episode_ids.to(device),
            agent_ids=self.agent_ids.to(device),
        )


class AgentSequenceDataset(Dataset[AgentSequence]):
    """Dataset of full time-ordered sequences for each `(episode_id, agent_id)` pair."""

    def __init__(self, sequences: list[AgentSequence]) -> None:
        if not sequences:
            raise ValueError("AgentSequenceDataset requires at least one sequence.")
        self.sequences = sequences
        self.feature_dim = int(sequences[0].features.shape[-1])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> AgentSequence:
        return self.sequences[index]

    @classmethod
    def from_imitation_dataset(
        cls,
        dataset: ImitationDataset,
        *,
        vectorizer: RecoveryObservationVectorizer | None = None,
        episode_ids: set[int] | None = None,
    ) -> "AgentSequenceDataset":
        vectorizer = vectorizer or RecoveryObservationVectorizer()
        groups: dict[tuple[int, int], list[int]] = {}
        for row_idx, (episode_id, agent_id) in enumerate(zip(dataset.episode_ids.tolist(), dataset.agent_ids.tolist())):
            episode_id = int(episode_id)
            if episode_ids is not None and episode_id not in episode_ids:
                continue
            groups.setdefault((episode_id, int(agent_id)), []).append(row_idx)

        sequences: list[AgentSequence] = []
        for (episode_id, agent_id), row_indices in sorted(groups.items()):
            row_indices = sorted(row_indices, key=lambda idx: int(dataset.timesteps[idx]))
            observations = [_observation_from_row(dataset, idx) for idx in row_indices]
            features = torch.as_tensor(vectorizer.batch_numpy(observations), dtype=torch.float32)
            labels = torch.as_tensor(dataset.labels[row_indices], dtype=torch.long)
            physical_actions = torch.as_tensor(dataset.physical_actions[row_indices], dtype=torch.long)
            timesteps = torch.as_tensor(dataset.timesteps[row_indices], dtype=torch.long)
            sequences.append(
                AgentSequence(
                    features=features,
                    labels=labels,
                    physical_actions=physical_actions,
                    episode_id=episode_id,
                    agent_id=agent_id,
                    timesteps=timesteps,
                )
            )
        return cls(sequences)


def split_episode_ids(dataset: ImitationDataset, *, train_fraction: float = 0.8, seed: int = 0) -> tuple[set[int], set[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1.")
    episode_ids = np.unique(dataset.episode_ids)
    if episode_ids.size < 2:
        raise ValueError("Need at least two episodes to split train and validation sets.")
    rng = np.random.default_rng(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)
    num_train = min(max(1, int(round(len(shuffled) * train_fraction))), len(shuffled) - 1)
    return set(map(int, shuffled[:num_train])), set(map(int, shuffled[num_train:]))


def collate_agent_sequences(sequences: list[AgentSequence]) -> SequenceBatch:
    if not sequences:
        raise ValueError("Cannot collate an empty sequence batch.")
    batch_size = len(sequences)
    max_length = max(sequence.length for sequence in sequences)
    feature_dim = int(sequences[0].features.shape[-1])

    features = torch.zeros((batch_size, max_length, feature_dim), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    episode_ids = torch.zeros(batch_size, dtype=torch.long)
    agent_ids = torch.zeros(batch_size, dtype=torch.long)

    for batch_idx, sequence in enumerate(sequences):
        length = sequence.length
        features[batch_idx, :length] = sequence.features
        labels[batch_idx, :length] = sequence.labels
        mask[batch_idx, :length] = True
        lengths[batch_idx] = length
        episode_ids[batch_idx] = sequence.episode_id
        agent_ids[batch_idx] = sequence.agent_id

    return SequenceBatch(
        features=features,
        labels=labels,
        mask=mask,
        lengths=lengths,
        episode_ids=episode_ids,
        agent_ids=agent_ids,
    )


def _observation_from_row(dataset: ImitationDataset, idx: int) -> dict:
    return {
        "obstacles": dataset.obstacles[idx],
        "agents": dataset.agents[idx],
        "xy": dataset.xy[idx],
        "target_xy": dataset.target_xy[idx],
        "planned_next_action": dataset.planned_next_action[idx],
        "planned_next_delta": dataset.planned_next_delta[idx],
        "scheduled_next_action": dataset.scheduled_next_action[idx],
        "lateness": dataset.lateness[idx],
        "off_plan": dataset.off_plan[idx],
        "remaining_plan_length": dataset.remaining_plan_length[idx],
        "was_delayed_last_step": dataset.was_delayed_last_step[idx],
    }
