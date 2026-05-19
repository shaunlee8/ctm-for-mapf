"""Generate communication-training data with held-out stress families reserved for evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pogema import GridConfig

from ctm_for_mapf.datasets import concatenate_imitation_datasets, generate_imitation_dataset
from ctm_for_mapf.envs import communication_stress_train_scenarios
from ctm_for_mapf.models import CentralizedTemporalRepairExpert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate communication-training imitation data.")
    parser.add_argument("--random-episodes", type=int, default=256)
    parser.add_argument("--stress-repeats", type=int, default=64)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--obs-radius", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--delay-probability", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expert = CentralizedTemporalRepairExpert()
    datasets = [
        generate_imitation_dataset(
            num_episodes=args.random_episodes,
            grid_config=GridConfig(
                observation_type="POMAPF",
                on_target="nothing",
                collision_system="block_both",
                num_agents=args.num_agents,
                size=args.size,
                density=args.density,
                obs_radius=args.obs_radius,
                max_episode_steps=args.max_steps,
            ),
            delay_probability=args.delay_probability,
            max_steps=args.max_steps,
            seed=args.seed,
            expert=expert,
        )
    ]
    for scenario in communication_stress_train_scenarios(obs_radius=args.obs_radius, max_episode_steps=args.max_steps):
        datasets.append(
            generate_imitation_dataset(
                num_episodes=args.stress_repeats,
                grid_config=scenario.grid_config,
                delay_probability=0.0,
                max_steps=args.max_steps,
                seed=args.seed,
                expert=expert,
                delay_model_factory=scenario.disturbance_factory,
            )
        )
    dataset = concatenate_imitation_datasets(datasets)
    dataset.save(args.output)
    unique, counts = np.unique(dataset.labels, return_counts=True)
    print(f"saved={args.output}")
    print(f"examples={dataset.num_examples}")
    print(f"label_counts={{{', '.join(f'{int(k)}: {int(v)}' for k, v in zip(unique, counts))}}}")
    print(f"episodes={len(np.unique(dataset.episode_ids))}")
    print(f"train_stress_variants={len(communication_stress_train_scenarios(obs_radius=args.obs_radius, max_episode_steps=args.max_steps))}")


if __name__ == "__main__":
    main()
