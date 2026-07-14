"""Deterministic, task-wise evaluation with explicit seed accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass(frozen=True)
class TaskEvaluation:
    task: str
    episodes: int
    seed: int
    mean_return: float
    std_return: float
    returns: list[float]


def evaluate_by_task(
    model: BaseAlgorithm,
    task_names: Iterable[str],
    make_env: Callable[[str, int], gym.Env],
    *,
    episodes: int,
    seed: int,
) -> list[dict[str, object]]:
    """Evaluate every task independently using deterministic, recorded seeds."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    reports: list[dict[str, object]] = []
    for task_index, task_name in enumerate(task_names):
        env = make_env(task_name, task_index)
        returns: list[float] = []
        try:
            for episode in range(episodes):
                episode_seed = seed + task_index * 10_000 + episode
                observation, _ = env.reset(seed=episode_seed)
                terminated = truncated = False
                total = 0.0
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                returns.append(total)
        finally:
            env.close()
        report = TaskEvaluation(
            task=task_name,
            episodes=episodes,
            seed=seed + task_index * 10_000,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            returns=returns,
        )
        reports.append(asdict(report))
    return reports
