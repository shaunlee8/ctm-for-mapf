"""Closed-loop rollout evaluation for temporal recovery policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
from pogema import GridConfig

from ctm_for_mapf.envs import DisturbanceModel, ForcedWaitDelayModel, RecoveryEnv
from ctm_for_mapf.models import (
    AdaptiveCTMTemporalRecoveryPolicy,
    CentralizedTemporalRepairExpert,
    CommunicatingCTMTemporalRecoveryPolicy,
    FOLLOW_PLAN_DECISION,
    FollowPlanPolicy,
    RecoveryObservationVectorizer,
    RecurrentTemporalRecoveryPolicy,
    SafeTemporalRecoveryExpert,
)


@dataclass(frozen=True, slots=True)
class AdaptiveInferenceConfig:
    """Inference-time compute policy for adaptive CTMs.

    ``certainty_threshold=None`` means a fixed-budget run that always uses
    ``max_ticks``. Otherwise, agents stop early once certainty crosses the threshold,
    subject to ``min_ticks`` and ``max_ticks``.
    """

    certainty_threshold: float | None
    min_ticks: int = 1
    max_ticks: int | None = None


@dataclass(frozen=True, slots=True)
class RolloutResult:
    success: bool
    nominal_makespan: int
    actual_makespan: int
    makespan_inflation: int
    injected_delays: int
    off_plan_agent_steps: int
    learned_decision_agent_steps: int = 0
    total_internal_ticks: int = 0
    tick_histogram: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    episodes: int
    success_rate: float
    mean_nominal_makespan: float
    mean_actual_makespan: float
    mean_makespan_inflation: float
    mean_injected_delays: float
    mean_off_plan_agent_steps: float
    mean_internal_ticks_per_decision: float | None
    tick_histogram: tuple[int, ...]


ControllerKind = Literal["follow_plan", "expert", "repair_expert", "learned"]


def evaluate_controller(
    *,
    controller: ControllerKind,
    grid_config: GridConfig,
    delay_probability: float,
    seeds: list[int],
    learned_policy: RecurrentTemporalRecoveryPolicy | None = None,
    adaptive_inference: AdaptiveInferenceConfig | None = None,
    communication_mode: Literal["learned", "zero", "shuffled"] = "learned",
    delay_model_factory: Callable[[], DisturbanceModel] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[RolloutSummary, list[RolloutResult]]:
    results = [
        run_rollout(
            controller=controller,
            grid_config=grid_config,
            delay_probability=delay_probability,
            seed=seed,
            learned_policy=learned_policy,
            adaptive_inference=adaptive_inference,
            communication_mode=communication_mode,
            delay_model_factory=delay_model_factory,
            device=device,
        )
        for seed in seeds
    ]
    return summarize_rollouts(results), results


def run_rollout(
    *,
    controller: ControllerKind,
    grid_config: GridConfig,
    delay_probability: float,
    seed: int,
    learned_policy: RecurrentTemporalRecoveryPolicy | None = None,
    adaptive_inference: AdaptiveInferenceConfig | None = None,
    communication_mode: Literal["learned", "zero", "shuffled"] = "learned",
    delay_model_factory: Callable[[], DisturbanceModel] | None = None,
    device: torch.device | str = "cpu",
) -> RolloutResult:
    env = RecoveryEnv(
        grid_config.copy(update={"seed": seed}),
        delay_model=(delay_model_factory() if delay_model_factory is not None else ForcedWaitDelayModel(delay_probability=delay_probability)),
        seed=seed,
    )
    observations, reset_info = env.reset()
    follow_plan = FollowPlanPolicy()
    expert = SafeTemporalRecoveryExpert()
    repair_expert = CentralizedTemporalRepairExpert()
    vectorizer = RecoveryObservationVectorizer()
    state = None
    communication_state = None
    if controller == "learned":
        if learned_policy is None:
            raise ValueError("learned_policy is required when controller='learned'.")
        learned_policy.eval()
        if isinstance(learned_policy, CommunicatingCTMTemporalRecoveryPolicy):
            if adaptive_inference is not None:
                raise TypeError("adaptive_inference is not supported for CommunicatingCTMTemporalRecoveryPolicy.")
            state = learned_policy.initial_joint_state(1, env.num_agents, device=device)
            communication_state = learned_policy.initial_communication_state(1, env.num_agents, device=device)
        else:
            state = learned_policy.initial_state(env.num_agents, device=device)

    off_plan_agent_steps = 0
    learned_decision_agent_steps = 0
    total_internal_ticks = 0
    tick_histogram: list[int] = []
    final_info = None
    terminated = [False] * env.num_agents
    truncated = [False] * env.num_agents
    for _ in range(grid_config.max_episode_steps):
        actual_positions = env.base_env.get_agents_xy(ignore_borders=True)
        if controller == "follow_plan":
            actions = follow_plan.act(env.tracker, actual_positions)
        elif controller == "expert":
            actions = list(expert.act(env.tracker, actual_positions).actions)
        elif controller == "repair_expert":
            actions = list(repair_expert.act(env.tracker, actual_positions).actions)
        elif controller == "learned":
            assert learned_policy is not None and state is not None
            batch = vectorizer.batch_tensor(observations, device=device)
            with torch.no_grad():
                if isinstance(learned_policy, CommunicatingCTMTemporalRecoveryPolicy):
                    assert communication_state is not None
                    joint_batch = batch.unsqueeze(0)
                    joint_positions = torch.as_tensor(actual_positions, dtype=torch.long, device=device).unsqueeze(0)
                    decisions_joint, state, communication_state = learned_policy.act_joint(
                        joint_batch,
                        joint_positions,
                        state,
                        communication_state,
                        message_mode=communication_mode,
                    )
                    decisions = decisions_joint.squeeze(0)
                    fixed_ticks = int(learned_policy.core.iterations)
                    ticks_used = torch.full((batch.shape[0],), fixed_ticks, dtype=torch.long, device=batch.device)
                elif adaptive_inference is not None:
                    if not isinstance(learned_policy, AdaptiveCTMTemporalRecoveryPolicy):
                        raise TypeError("adaptive_inference requires AdaptiveCTMTemporalRecoveryPolicy.")
                    adaptive_output = learned_policy.act_adaptive(
                        batch,
                        state,
                        certainty_threshold=adaptive_inference.certainty_threshold,
                        min_ticks=adaptive_inference.min_ticks,
                        max_ticks=adaptive_inference.max_ticks,
                    )
                    decisions = adaptive_output.decisions
                    state = adaptive_output.state
                    ticks_used = adaptive_output.ticks_used
                else:
                    decisions, state = learned_policy.act(batch, state)
                    fixed_ticks = int(getattr(getattr(learned_policy, "core", None), "iterations", 0))
                    ticks_used = torch.full((batch.shape[0],), fixed_ticks, dtype=torch.long, device=batch.device)
            learned_decision_agent_steps += int(batch.shape[0])
            total_internal_ticks += int(ticks_used.sum().item())
            max_tick = int(ticks_used.max().item()) if ticks_used.numel() else 0
            if len(tick_histogram) < max_tick:
                tick_histogram.extend([0] * (max_tick - len(tick_histogram)))
            for tick, count in enumerate(torch.bincount(ticks_used.cpu(), minlength=max_tick + 1).tolist()):
                if tick == 0:
                    continue
                tick_histogram[tick - 1] += int(count)
            actions = [
                0 if int(decision) != FOLLOW_PLAN_DECISION else env.tracker.path_next_action(agent_id, tuple(position))
                for agent_id, (decision, position) in enumerate(zip(decisions.cpu().tolist(), actual_positions))
            ]
        else:
            raise ValueError(f"Unknown controller {controller!r}.")

        observations, _, terminated, truncated, final_info = env.step(actions)
        off_plan_agent_steps += sum(final_info.off_plan)
        if all(terminated) or all(truncated):
            break

    actual_makespan = env.tracker.timestep
    injected_delays = final_info.injected_delay_count if final_info is not None else 0
    nominal_makespan = int(reset_info["nominal_makespan"])
    return RolloutResult(
        success=all(terminated),
        nominal_makespan=nominal_makespan,
        actual_makespan=actual_makespan,
        makespan_inflation=actual_makespan - nominal_makespan,
        injected_delays=injected_delays,
        off_plan_agent_steps=off_plan_agent_steps,
        learned_decision_agent_steps=learned_decision_agent_steps,
        total_internal_ticks=total_internal_ticks,
        tick_histogram=tuple(tick_histogram),
    )


def summarize_rollouts(results: list[RolloutResult]) -> RolloutSummary:
    if not results:
        raise ValueError("Cannot summarize zero rollouts.")
    learned_decisions = sum(result.learned_decision_agent_steps for result in results)
    total_ticks = sum(result.total_internal_ticks for result in results)
    max_hist_len = max((len(result.tick_histogram) for result in results), default=0)
    histogram = [0] * max_hist_len
    for result in results:
        for idx, count in enumerate(result.tick_histogram):
            histogram[idx] += count
    return RolloutSummary(
        episodes=len(results),
        success_rate=float(np.mean([result.success for result in results])),
        mean_nominal_makespan=float(np.mean([result.nominal_makespan for result in results])),
        mean_actual_makespan=float(np.mean([result.actual_makespan for result in results])),
        mean_makespan_inflation=float(np.mean([result.makespan_inflation for result in results])),
        mean_injected_delays=float(np.mean([result.injected_delays for result in results])),
        mean_off_plan_agent_steps=float(np.mean([result.off_plan_agent_steps for result in results])),
        mean_internal_ticks_per_decision=(total_ticks / learned_decisions) if learned_decisions else None,
        tick_histogram=tuple(histogram),
    )
