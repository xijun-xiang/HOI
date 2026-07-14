"""Task-wise evaluation that preserves numeric environment diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm


def _numeric_info(info: dict[str, object]) -> dict[str, float]:
    """Keep scalar diagnostics only; arrays and nested payloads are not silently averaged."""
    return {
        key: float(value)
        for key, value in info.items()
        if isinstance(value, (bool, int, float, np.number))
    }


def evaluate_rich_by_task(
    model: BaseAlgorithm,
    task_names: Iterable[str],
    make_env: Callable[[str, int], gym.Env],
    *,
    episodes: int,
    seed: int,
) -> list[dict[str, object]]:
    """Evaluate returns plus final/mean/min/max numeric ``info`` diagnostics.

    The function does not infer success semantics.  It records the environment
    keys verbatim, making proxies such as ``door_opening`` auditable rather
    than relabelling them as a solved task.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    reports: list[dict[str, object]] = []
    for task_index, task_name in enumerate(task_names):
        env = make_env(task_name, task_index)
        returns: list[float] = []
        episode_metrics: list[dict[str, dict[str, float]]] = []
        try:
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed + task_index * 10_000 + episode)
                terminated = truncated = False
                total_return = 0.0
                values: dict[str, list[float]] = defaultdict(list)
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_return += float(reward)
                    for key, value in _numeric_info(info).items():
                        values[key].append(value)
                returns.append(total_return)
                episode_metrics.append(
                    {
                        key: {
                            "final": samples[-1],
                            "mean": float(np.mean(samples)),
                            "min": float(np.min(samples)),
                            "max": float(np.max(samples)),
                        }
                        for key, samples in values.items()
                        if samples
                    }
                )
        finally:
            env.close()

        aggregate: dict[str, dict[str, float]] = {}
        all_keys = sorted({key for metrics in episode_metrics for key in metrics})
        for key in all_keys:
            aggregate[key] = {
                stat: float(
                    np.mean([metrics[key][stat] for metrics in episode_metrics if key in metrics])
                )
                for stat in ("final", "mean", "min", "max")
            }
        reports.append(
            {
                "task": task_name,
                "episodes": episodes,
                "seed": seed + task_index * 10_000,
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "returns": returns,
                "info": aggregate,
            }
        )
    return reports
