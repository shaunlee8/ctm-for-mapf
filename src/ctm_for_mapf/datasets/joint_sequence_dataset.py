"""Joint multi-agent episode sequence views over flat imitation datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ctm_for_mapf.datasets.imitation_rollouts import ImitationDataset
from ctm_for_mapf.datasets.sequence_dataset import _observation_from_row
from ctm_for_mapf.models.observation_vectorizer import RecoveryObservationVectorizer


@dataclass(frozen=True, slots=True)
class JointEpisodeSequence:
    """One episode arranged as ``[time, agents, ...]`` for communication training."""

    features: torch.Tensor
    labels: torch.Tensor
    physical_actions: torch.Tensor
    positions: torch.Tensor
    label_mask: torch.Tensor
    agent_mask: torch.Tensor
    timesteps: torch.Tensor
    episode_id: int
    agent_ids: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.timesteps.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.agent_ids.shape[0])


@dataclass(frozen=True, slots=True)
class JointSequenceBatch:
    """Padded batch of joint episodes.

    Shapes:

    - ``features``: ``[batch, time, agents, feature_dim]``
    - ``labels``: ``[batch, time, agents]``
    - ``positions``: ``[batch, time, agents, 2]``
    - ``label_mask``: examples that should contribute to supervised loss
    - ``agent_mask``: agent slots that exist in the episode
    - ``timestep_mask``: timesteps that exist before batch padding
    - ``presence_mask``: rows with an observed feature record in the flat corpus

    ``label_mask`` and ``presence_mask`` are currently the same since the saved corpus
    stores only active examples. A dataset problem for later.
    """

    features: torch.Tensor
    labels: torch.Tensor
    physical_actions: torch.Tensor
    positions: torch.Tensor
    label_mask: torch.Tensor
    agent_mask: torch.Tensor
    timestep_mask: torch.Tensor
    presence_mask: torch.Tensor
    lengths: torch.Tensor
    num_agents: torch.Tensor
    episode_ids: torch.Tensor
    agent_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "JointSequenceBatch":
        return JointSequenceBatch(
            features=self.features.to(device),
            labels=self.labels.to(device),
            physical_actions=self.physical_actions.to(device),
            positions=self.positions.to(device),
            label_mask=self.label_mask.to(device),
            agent_mask=self.agent_mask.to(device),
            timestep_mask=self.timestep_mask.to(device),
            presence_mask=self.presence_mask.to(device),
            lengths=self.lengths.to(device),
            num_agents=self.num_agents.to(device),
            episode_ids=self.episode_ids.to(device),
            agent_ids=self.agent_ids.to(device),
        )


class JointEpisodeSequenceDataset(Dataset[JointEpisodeSequence]):
    """Dataset of full time-ordered joint episodes for communication training."""

    def __init__(self, episodes: list[JointEpisodeSequence]) -> None:
        if not episodes:
            raise ValueError("JointEpisodeSequenceDataset requires at least one episode.")
        self.episodes = episodes
        self.feature_dim = int(episodes[0].features.shape[-1])
        self.max_agents = max(episode.num_agents for episode in episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> JointEpisodeSequence:
        return self.episodes[index]

    @classmethod
    def from_imitation_dataset(
        cls,
        dataset: ImitationDataset,
        *,
        vectorizer: RecoveryObservationVectorizer | None = None,
        episode_ids: set[int] | None = None,
    ) -> "JointEpisodeSequenceDataset":
        vectorizer = vectorizer or RecoveryObservationVectorizer()
        rows_by_episode: dict[int, list[int]] = {}
        for row_idx, episode_id in enumerate(dataset.episode_ids.tolist()):
            episode_id = int(episode_id)
            if episode_ids is not None and episode_id not in episode_ids:
                continue
            rows_by_episode.setdefault(episode_id, []).append(row_idx)

        episodes: list[JointEpisodeSequence] = []
        for episode_id, row_indices in sorted(rows_by_episode.items()):
            timesteps = sorted({int(dataset.timesteps[idx]) for idx in row_indices})
            agent_ids = sorted({int(dataset.agent_ids[idx]) for idx in row_indices})
            timestep_to_index = {timestep: idx for idx, timestep in enumerate(timesteps)}
            agent_to_index = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
            length = len(timesteps)
            num_agents = len(agent_ids)

            sample_observation = _observation_from_row(dataset, row_indices[0])
            feature_dim = int(vectorizer.vectorize(sample_observation).shape[0])
            features = torch.zeros((length, num_agents, feature_dim), dtype=torch.float32)
            labels = torch.zeros((length, num_agents), dtype=torch.long)
            physical_actions = torch.zeros((length, num_agents), dtype=torch.long)
            positions = torch.zeros((length, num_agents, 2), dtype=torch.long)
            label_mask = torch.zeros((length, num_agents), dtype=torch.bool)

            observations = [_observation_from_row(dataset, idx) for idx in row_indices]
            vectorized = vectorizer.batch_numpy(observations)
            for row_idx, flat_idx in enumerate(row_indices):
                time_idx = timestep_to_index[int(dataset.timesteps[flat_idx])]
                agent_idx = agent_to_index[int(dataset.agent_ids[flat_idx])]
                features[time_idx, agent_idx] = torch.as_tensor(vectorized[row_idx], dtype=torch.float32)
                labels[time_idx, agent_idx] = int(dataset.labels[flat_idx])
                physical_actions[time_idx, agent_idx] = int(dataset.physical_actions[flat_idx])
                positions[time_idx, agent_idx] = torch.as_tensor(dataset.xy[flat_idx], dtype=torch.long)
                label_mask[time_idx, agent_idx] = bool(dataset.active_examples[flat_idx])

            episodes.append(
                JointEpisodeSequence(
                    features=features,
                    labels=labels,
                    physical_actions=physical_actions,
                    positions=positions,
                    label_mask=label_mask,
                    agent_mask=torch.ones(num_agents, dtype=torch.bool),
                    timesteps=torch.as_tensor(timesteps, dtype=torch.long),
                    episode_id=episode_id,
                    agent_ids=torch.as_tensor(agent_ids, dtype=torch.long),
                )
            )
        return cls(episodes)


def collate_joint_episode_sequences(episodes: list[JointEpisodeSequence]) -> JointSequenceBatch:
    if not episodes:
        raise ValueError("Cannot collate an empty joint episode batch.")
    batch_size = len(episodes)
    max_length = max(episode.length for episode in episodes)
    max_agents = max(episode.num_agents for episode in episodes)
    feature_dim = int(episodes[0].features.shape[-1])

    features = torch.zeros((batch_size, max_length, max_agents, feature_dim), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_length, max_agents), dtype=torch.long)
    physical_actions = torch.zeros((batch_size, max_length, max_agents), dtype=torch.long)
    positions = torch.zeros((batch_size, max_length, max_agents, 2), dtype=torch.long)
    label_mask = torch.zeros((batch_size, max_length, max_agents), dtype=torch.bool)
    agent_mask = torch.zeros((batch_size, max_agents), dtype=torch.bool)
    timestep_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    presence_mask = torch.zeros((batch_size, max_length, max_agents), dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    num_agents = torch.zeros(batch_size, dtype=torch.long)
    episode_ids = torch.zeros(batch_size, dtype=torch.long)
    agent_ids = torch.full((batch_size, max_agents), fill_value=-1, dtype=torch.long)

    for batch_idx, episode in enumerate(episodes):
        length = episode.length
        count = episode.num_agents
        features[batch_idx, :length, :count] = episode.features
        labels[batch_idx, :length, :count] = episode.labels
        physical_actions[batch_idx, :length, :count] = episode.physical_actions
        positions[batch_idx, :length, :count] = episode.positions
        label_mask[batch_idx, :length, :count] = episode.label_mask
        presence_mask[batch_idx, :length, :count] = episode.label_mask
        agent_mask[batch_idx, :count] = episode.agent_mask
        timestep_mask[batch_idx, :length] = True
        lengths[batch_idx] = length
        num_agents[batch_idx] = count
        episode_ids[batch_idx] = episode.episode_id
        agent_ids[batch_idx, :count] = episode.agent_ids

    return JointSequenceBatch(
        features=features,
        labels=labels,
        physical_actions=physical_actions,
        positions=positions,
        label_mask=label_mask,
        agent_mask=agent_mask,
        timestep_mask=timestep_mask,
        presence_mask=presence_mask,
        lengths=lengths,
        num_agents=num_agents,
        episode_ids=episode_ids,
        agent_ids=agent_ids,
    )
