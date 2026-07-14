"""Observation alignment and explicit task identity for shared policies."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def build_task_onehot(task_index: int, num_tasks: int) -> np.ndarray:
    """Return a float32 one-hot task descriptor with validated dimensions."""
    if num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    if not 0 <= task_index < num_tasks:
        raise ValueError(f"task_index must be in [0, {num_tasks}), got {task_index}")
    vector = np.zeros(num_tasks, dtype=np.float32)
    vector[task_index] = 1.0
    return vector


class TaskAlignedObservation(gym.ObservationWrapper):
    """Pad a vector observation and optionally append a task one-hot vector.

    HumanoidBench tasks expose state vectors with different lengths. A shared
    network needs a common shape, so every observation is right-padded to
    ``target_dim``. Appending a one-hot identity is optional: it distinguishes
    the task-id and task-token controls from the task-agnostic baseline.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        task_index: int,
        num_tasks: int,
        target_dim: int,
        append_task_id: bool,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("TaskAlignedObservation requires a Box observation space")
        self.task_id = build_task_onehot(task_index, num_tasks)
        self.target_dim = int(target_dim)
        self.append_task_id = bool(append_task_id)
        source_dim = int(np.prod(env.observation_space.shape))
        if self.target_dim < source_dim:
            raise ValueError(
                f"target_dim ({self.target_dim}) is smaller than source dim ({source_dim})"
            )
        output_dim = self.target_dim + (num_tasks if self.append_task_id else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(output_dim,), dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        source = np.asarray(observation, dtype=np.float32).reshape(-1)
        if source.size > self.target_dim:
            raise ValueError("received observation exceeds configured target_dim")
        aligned = np.zeros(self.target_dim, dtype=np.float32)
        aligned[: source.size] = source
        if not self.append_task_id:
            return aligned
        return np.concatenate((aligned, self.task_id), dtype=np.float32)
