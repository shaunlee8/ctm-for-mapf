"""Scripted coordination-stress scenarios for recovery experiments. Roleplay here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pogema import GridConfig

from ctm_for_mapf.envs.disturbances import BurstDelayModel, DisturbanceModel


@dataclass(frozen=True, slots=True)
class StressScenario:
    """A named recovery stress scenario with a recommended structured delay."""

    name: str
    grid_config: GridConfig
    disturbance_factory: Callable[[], DisturbanceModel]
    description: str
    family: str = "legacy"
    variant: str = "base"
    split: str = "legacy"


def corridor_convoy_variant(
    *,
    corridor_length: int = 9,
    delay_start: int = 1,
    delay_duration: int = 3,
    obs_radius: int = 1,
    max_episode_steps: int = 24,
    variant: str | None = None,
    split: str = "custom",
) -> StressScenario:
    """Three agents follow one corridor. A lead-agent suddenly rushes backwards."""
    if corridor_length < 9:
        raise ValueError("corridor_length must be >= 9.")
    lead_start = corridor_length // 2
    followers = (lead_start - 2, lead_start - 4)
    if followers[-1] < 0:
        raise ValueError("corridor_length leaves insufficient spacing for three agents.")
    name_variant = variant or f"l{corridor_length}_s{delay_start}_d{delay_duration}"
    return StressScenario(
        name=f"corridor_convoy__{name_variant}",
        family="corridor_convoy",
        variant=name_variant,
        split=split,
        grid_config=GridConfig(
            observation_type="POMAPF",
            on_target="nothing",
            collision_system="block_both",
            map="\n".join(("#" * corridor_length, "." * corridor_length, "#" * corridor_length)),
            agents_xy=[(1, lead_start), (1, followers[0]), (1, followers[1])],
            targets_xy=[(1, corridor_length - 1), (1, corridor_length - 2), (1, corridor_length - 3)],
            obs_radius=obs_radius,
            max_episode_steps=max_episode_steps,
        ),
        disturbance_factory=lambda: BurstDelayModel(agent_ids=(0,), start_timestep=delay_start, duration=delay_duration),
        description="A remote burst on the lead agent forces trailing convoy agents to reschedule.",
    )


def intersection_variant(
    *,
    arm_length: int = 3,
    delay_start: int = 1,
    delay_duration: int = 1,
    obs_radius: int = 1,
    max_episode_steps: int = 24,
    variant: str | None = None,
    split: str = "custom",
) -> StressScenario:
    """Two agents cross a single-cell intersection from perpendicular corridors."""
    if arm_length < 3:
        raise ValueError("arm_length must be >= 3.")
    size = 2 * arm_length + 1
    center = arm_length
    rows = []
    for row in range(size):
        chars = ["#"] * size
        if row == center:
            chars = ["."] * size
        else:
            chars[center] = "."
        rows.append("".join(chars))
    name_variant = variant or f"a{arm_length}_s{delay_start}_d{delay_duration}"
    return StressScenario(
        name=f"intersection__{name_variant}",
        family="intersection",
        variant=name_variant,
        split=split,
        grid_config=GridConfig(
            observation_type="POMAPF",
            on_target="nothing",
            collision_system="block_both",
            map="\n".join(rows),
            agents_xy=[(center, 0), (0, center)],
            targets_xy=[(center, size - 1), (size - 1, center)],
            obs_radius=obs_radius,
            max_episode_steps=max_episode_steps,
        ),
        disturbance_factory=lambda: BurstDelayModel(agent_ids=(0,), start_timestep=delay_start, duration=delay_duration),
        description="A delayed entrant shifts the order in which agents should use the intersection.",
    )


def merge_variant(
    *,
    shared_width: int = 9,
    delay_start: int = 0,
    delay_duration: int = 3,
    obs_radius: int = 1,
    max_episode_steps: int = 24,
    variant: str | None = None,
    split: str = "custom",
) -> StressScenario:
    """Two branch agents merge into a shared corridor behind an existing lead agent."""
    if shared_width < 9:
        raise ValueError("shared_width must be >= 9.")
    branch_col = shared_width // 2 - 1
    center_row = 2
    rows = []
    for row in range(5):
        chars = ["#"] * shared_width
        if row == center_row:
            chars[0] = "#"
            chars[-1] = "#"
            for col in range(1, shared_width - 1):
                chars[col] = "."
        else:
            chars[branch_col] = "."
        rows.append("".join(chars))
    name_variant = variant or f"w{shared_width}_s{delay_start}_d{delay_duration}"
    return StressScenario(
        name=f"merge__{name_variant}",
        family="merge",
        variant=name_variant,
        split=split,
        grid_config=GridConfig(
            observation_type="POMAPF",
            on_target="nothing",
            collision_system="block_both",
            map="\n".join(rows),
            agents_xy=[(center_row, branch_col + 2), (0, branch_col), (4, branch_col)],
            targets_xy=[(center_row, shared_width - 2), (center_row, shared_width - 3), (center_row, max(1, branch_col + 1))],
            obs_radius=obs_radius,
            max_episode_steps=max_episode_steps,
        ),
        disturbance_factory=lambda: BurstDelayModel(agent_ids=(0,), start_timestep=delay_start, duration=delay_duration),
        description="Branch agents must coordinate around a slowed lead agent in the shared corridor.",
    )


def corridor_convoy_scenario(*, obs_radius: int = 1, max_episode_steps: int = 24) -> StressScenario:
    return corridor_convoy_variant(obs_radius=obs_radius, max_episode_steps=max_episode_steps, variant="base", split="legacy")


def intersection_scenario(*, obs_radius: int = 1, max_episode_steps: int = 24) -> StressScenario:
    return intersection_variant(obs_radius=obs_radius, max_episode_steps=max_episode_steps, variant="base", split="legacy")


def merge_scenario(*, obs_radius: int = 1, max_episode_steps: int = 24) -> StressScenario:
    return merge_variant(obs_radius=obs_radius, max_episode_steps=max_episode_steps, variant="base", split="legacy")


def coordination_stress_scenarios(*, obs_radius: int = 1, max_episode_steps: int = 24) -> tuple[StressScenario, ...]:
    return (
        corridor_convoy_scenario(obs_radius=obs_radius, max_episode_steps=max_episode_steps),
        intersection_scenario(obs_radius=obs_radius, max_episode_steps=max_episode_steps),
        merge_scenario(obs_radius=obs_radius, max_episode_steps=max_episode_steps),
    )


def communication_stress_train_scenarios(*, obs_radius: int = 2, max_episode_steps: int = 40) -> tuple[StressScenario, ...]:
    """Parameterised stress variants used to train communication policies."""
    return (
        corridor_convoy_variant(corridor_length=9, delay_duration=2, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="train"),
        corridor_convoy_variant(corridor_length=11, delay_duration=3, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="train"),
        intersection_variant(arm_length=3, delay_start=1, delay_duration=1, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="train"),
        intersection_variant(arm_length=4, delay_start=2, delay_duration=1, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="train"),
        merge_variant(shared_width=9, delay_duration=2, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="train"),
        merge_variant(shared_width=11, delay_duration=3, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="train"),
    )


def communication_stress_test_scenarios(*, obs_radius: int = 2, max_episode_steps: int = 48) -> tuple[StressScenario, ...]:
    """Held-out communication-demand variants reserved for evaluation."""
    return (
        corridor_convoy_variant(corridor_length=13, delay_duration=4, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="test"),
        corridor_convoy_variant(corridor_length=15, delay_start=2, delay_duration=3, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="test"),
        intersection_variant(arm_length=5, delay_start=0, delay_duration=1, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="test"),
        intersection_variant(arm_length=6, delay_start=3, delay_duration=1, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="test"),
        merge_variant(shared_width=13, delay_duration=4, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="test"),
        merge_variant(shared_width=15, delay_start=1, delay_duration=3, obs_radius=obs_radius, max_episode_steps=max_episode_steps, split="test"),
    )
