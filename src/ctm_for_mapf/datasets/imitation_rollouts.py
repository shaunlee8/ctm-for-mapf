"""
Copyright (c) 2026 Michael Mulder, Shaun Christopher Lee.

Generate temporal-recovery imitation data from expert rollouts.

Third-party dependencies used by this module:
- POGEMA:
  MISSING!!!
- PyTorch, PyTorch Foundation / contributors:
  https://pytorch.org/
- NumPy, NumPy developers:
  https://numpy.org/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from pogema import GridConfig

from ctm_for_mapf.envs import DisturbanceModel, ForcedWaitDelayModel, RecoveryEnv
from ctm_for_mapf.models import SafeTemporalRecoveryExpert, TemporalExpertOutput


class TemporalRecoveryExpert(Protocol):
    def act(self, tracker, actual_positions) -> TemporalExpertOutput:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ImitationDataset:
    obstacles: np.ndarray
    agents: np.ndarray
    xy: np.ndarray
    target_xy: np.ndarray
    planned_next_action: np.ndarray
    planned_next_delta: np.ndarray
    scheduled_next_action: np.ndarray
    lateness: np.ndarray
    off_plan: np.ndarray
    remaining_plan_length: np.ndarray
    was_delayed_last_step: np.ndarray
    labels: np.ndarray
    physical_actions: np.ndarray
    episode_ids: np.ndarray
    timesteps: np.ndarray
    agent_ids: np.ndarray
    active_examples: np.ndarray

    @property
    def num_examples(self) -> int:
        return int(self.labels.shape[0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            obstacles=self.obstacles,
            agents=self.agents,
            xy=self.xy,
            target_xy=self.target_xy,
            planned_next_action=self.planned_next_action,
            planned_next_delta=self.planned_next_delta,
            scheduled_next_action=self.scheduled_next_action,
            lateness=self.lateness,
            off_plan=self.off_plan,
            remaining_plan_length=self.remaining_plan_length,
            was_delayed_last_step=self.was_delayed_last_step,
            labels=self.labels,
            physical_actions=self.physical_actions,
            episode_ids=self.episode_ids,
            timesteps=self.timesteps,
            agent_ids=self.agent_ids,
            active_examples=self.active_examples,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ImitationDataset":
        with np.load(path) as data:
            return cls(**{key: data[key] for key in data.files})


def generate_imitation_dataset(
    *,
    num_episodes: int,
    grid_config: GridConfig,
    delay_probability: float,
    max_steps: int | None = None,
    seed: int = 0,
    expert: TemporalRecoveryExpert | None = None,
    delay_model_factory: Callable[[], DisturbanceModel] | None = None,
) -> ImitationDataset:
    """Roll out a temporal recovery expert and collect supervised examples."""
    rng = np.random.default_rng(seed)
    expert = expert or SafeTemporalRecoveryExpert()
    records: dict[str, list[Any]] = {
        "obstacles": [],
        "agents": [],
        "xy": [],
        "target_xy": [],
        "planned_next_action": [],
        "planned_next_delta": [],
        "scheduled_next_action": [],
        "lateness": [],
        "off_plan": [],
        "remaining_plan_length": [],
        "was_delayed_last_step": [],
        "labels": [],
        "physical_actions": [],
        "episode_ids": [],
        "timesteps": [],
        "agent_ids": [],
        "active_examples": [],
    }

    episode_id = 0
    attempts = 0
    max_attempts = max(num_episodes * 10, num_episodes)
    while episode_id < num_episodes:
        if attempts >= max_attempts:
            raise RuntimeError(f"Failed to generate {num_episodes} valid episodes within {max_attempts} attempts.")
        attempts += 1
        env_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        env = RecoveryEnv(
            grid_config.copy(update={"seed": env_seed}),
            delay_model=(delay_model_factory() if delay_model_factory is not None else ForcedWaitDelayModel(delay_probability=delay_probability)),
            seed=env_seed,
        )
        try:
            observations, _ = env.reset()
        except RuntimeError:
            # Random MAPF instances are occasionally unsolvable for the simple PP. Skip.
            continue
        horizon = max_steps or grid_config.max_episode_steps

        for _ in range(horizon):
            actual_positions = env.base_env.get_agents_xy(ignore_borders=True)
            expert_output = expert.act(env.tracker, actual_positions)
            _append_examples(records, observations, expert_output.decisions, expert_output.actions, episode_id, env.tracker.timestep)
            observations, _, terminated, truncated, _ = env.step(expert_output.actions)
            if all(terminated) or all(truncated):
                break
        episode_id += 1

    return ImitationDataset(
        obstacles=np.asarray(records["obstacles"], dtype=np.float32),
        agents=np.asarray(records["agents"], dtype=np.float32),
        xy=np.asarray(records["xy"], dtype=np.int32),
        target_xy=np.asarray(records["target_xy"], dtype=np.int32),
        planned_next_action=np.asarray(records["planned_next_action"], dtype=np.float32),
        planned_next_delta=np.asarray(records["planned_next_delta"], dtype=np.float32),
        scheduled_next_action=np.asarray(records["scheduled_next_action"], dtype=np.float32),
        lateness=np.asarray(records["lateness"], dtype=np.float32),
        off_plan=np.asarray(records["off_plan"], dtype=np.float32),
        remaining_plan_length=np.asarray(records["remaining_plan_length"], dtype=np.float32),
        was_delayed_last_step=np.asarray(records["was_delayed_last_step"], dtype=np.float32),
        labels=np.asarray(records["labels"], dtype=np.int64),
        physical_actions=np.asarray(records["physical_actions"], dtype=np.int64),
        episode_ids=np.asarray(records["episode_ids"], dtype=np.int64),
        timesteps=np.asarray(records["timesteps"], dtype=np.int64),
        agent_ids=np.asarray(records["agent_ids"], dtype=np.int64),
        active_examples=np.asarray(records["active_examples"], dtype=np.bool_),
    )


def concatenate_imitation_datasets(datasets: list[ImitationDataset]) -> ImitationDataset:
    """Concatenate compatible imitation datasets while reindexing episode ids."""
    if not datasets:
        raise ValueError("At least one dataset is required.")
    fields = tuple(ImitationDataset.__dataclass_fields__)
    episode_offset = 0
    pieces: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    for dataset in datasets:
        for field in fields:
            values = getattr(dataset, field)
            if field == "episode_ids":
                values = values + episode_offset
            pieces[field].append(values)
        episode_offset += int(dataset.episode_ids.max()) + 1 if dataset.episode_ids.size else 0
    return ImitationDataset(**{field: np.concatenate(parts, axis=0) for field, parts in pieces.items()})


def _append_examples(
    records: dict[str, list[Any]],
    observations: list[dict],
    decisions: tuple[int, ...],
    physical_actions: tuple[int, ...],
    episode_id: int,
    timestep: int,
) -> None:
    for agent_id, (observation, decision, physical_action) in enumerate(zip(observations, decisions, physical_actions)):
        active_example = bool(observation["remaining_plan_length"] > 0)
        if not active_example:
            continue
        for key in (
            "obstacles",
            "agents",
            "xy",
            "target_xy",
            "planned_next_action",
            "planned_next_delta",
            "scheduled_next_action",
            "lateness",
            "off_plan",
            "remaining_plan_length",
            "was_delayed_last_step",
        ):
            records[key].append(observation[key])
        records["labels"].append(decision)
        records["physical_actions"].append(physical_action)
        records["episode_ids"].append(episode_id)
        records["timesteps"].append(timestep)
        records["agent_ids"].append(agent_id)
        records["active_examples"].append(active_example)
