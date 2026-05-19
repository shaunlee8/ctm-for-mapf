"""Tokenized per-agent sequence views over flat imitation datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from ctm_for_mapf.datasets.imitation_rollouts import ImitationDataset
from ctm_for_mapf.datasets.sequence_dataset import _observation_from_row
from ctm_for_mapf.models.observation_vectorizer import RecoveryObservationTokenizer


@dataclass(frozen=True, slots=True)
class TokenAgentSequence:
    tokens: torch.Tensor
    labels: torch.Tensor
    physical_actions: torch.Tensor
    episode_id: int
    agent_id: int
    timesteps: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True, slots=True)
class TokenSequenceBatch:
    tokens: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    lengths: torch.Tensor
    episode_ids: torch.Tensor
    agent_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "TokenSequenceBatch":
        return TokenSequenceBatch(
            tokens=self.tokens.to(device),
            labels=self.labels.to(device),
            mask=self.mask.to(device),
            lengths=self.lengths.to(device),
            episode_ids=self.episode_ids.to(device),
            agent_ids=self.agent_ids.to(device),
        )


class TokenAgentSequenceDataset(Dataset[TokenAgentSequence]):
    """Dataset of full time-ordered token sequences for each agent."""

    def __init__(self, sequences: list[TokenAgentSequence]) -> None:
        if not sequences:
            raise ValueError("TokenAgentSequenceDataset requires at least one sequence.")
        self.sequences = sequences
        self.num_tokens = int(sequences[0].tokens.shape[-2])
        self.token_dim = int(sequences[0].tokens.shape[-1])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> TokenAgentSequence:
        return self.sequences[index]

    @classmethod
    def from_imitation_dataset(
        cls,
        dataset: ImitationDataset,
        *,
        tokenizer: RecoveryObservationTokenizer | None = None,
        episode_ids: set[int] | None = None,
    ) -> "TokenAgentSequenceDataset":
        tokenizer = tokenizer or RecoveryObservationTokenizer()
        groups: dict[tuple[int, int], list[int]] = {}
        for row_idx, (episode_id, agent_id) in enumerate(zip(dataset.episode_ids.tolist(), dataset.agent_ids.tolist())):
            episode_id = int(episode_id)
            if episode_ids is not None and episode_id not in episode_ids:
                continue
            groups.setdefault((episode_id, int(agent_id)), []).append(row_idx)

        sequences: list[TokenAgentSequence] = []
        for (episode_id, agent_id), row_indices in sorted(groups.items()):
            row_indices = sorted(row_indices, key=lambda idx: int(dataset.timesteps[idx]))
            observations = [_observation_from_row(dataset, idx) for idx in row_indices]
            tokens = torch.as_tensor(tokenizer.batch_numpy(observations), dtype=torch.float32)
            labels = torch.as_tensor(dataset.labels[row_indices], dtype=torch.long)
            physical_actions = torch.as_tensor(dataset.physical_actions[row_indices], dtype=torch.long)
            timesteps = torch.as_tensor(dataset.timesteps[row_indices], dtype=torch.long)
            sequences.append(
                TokenAgentSequence(
                    tokens=tokens,
                    labels=labels,
                    physical_actions=physical_actions,
                    episode_id=episode_id,
                    agent_id=agent_id,
                    timesteps=timesteps,
                )
            )
        return cls(sequences)


def collate_token_agent_sequences(sequences: list[TokenAgentSequence]) -> TokenSequenceBatch:
    if not sequences:
        raise ValueError("Cannot collate an empty token sequence batch.")
    batch_size = len(sequences)
    max_length = max(sequence.length for sequence in sequences)
    num_tokens = int(sequences[0].tokens.shape[-2])
    token_dim = int(sequences[0].tokens.shape[-1])

    tokens = torch.zeros((batch_size, max_length, num_tokens, token_dim), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_length), dtype=torch.long)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    episode_ids = torch.zeros(batch_size, dtype=torch.long)
    agent_ids = torch.zeros(batch_size, dtype=torch.long)

    for batch_idx, sequence in enumerate(sequences):
        length = sequence.length
        tokens[batch_idx, :length] = sequence.tokens
        labels[batch_idx, :length] = sequence.labels
        mask[batch_idx, :length] = True
        lengths[batch_idx] = length
        episode_ids[batch_idx] = sequence.episode_id
        agent_ids[batch_idx] = sequence.agent_id

    return TokenSequenceBatch(
        tokens=tokens,
        labels=labels,
        mask=mask,
        lengths=lengths,
        episode_ids=episode_ids,
        agent_ids=agent_ids,
    )
