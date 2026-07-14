"""Demonstration data, quality filtering, and task-balanced mini-batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def normalise_task_weights(values: list[float] | np.ndarray, num_tasks: int) -> np.ndarray:
    """Validate non-negative task weights and return a probability vector."""
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights.size != num_tasks:
        raise ValueError(f"expected {num_tasks} task weights, got {weights.size}")
    if np.any(weights < 0):
        raise ValueError("task weights must be non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("task weights must have positive sum")
    return (weights / total).astype(np.float64)


@dataclass(frozen=True)
class Demonstrations:
    """Validated vector observations/actions with optional task provenance."""

    observations: np.ndarray
    actions: np.ndarray
    task_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.observations.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("observations and actions must both be rank-2")
        if self.observations.shape[0] != self.actions.shape[0] or self.observations.shape[0] < 1:
            raise ValueError("observations/actions must have the same non-zero sample count")
        if self.task_ids is not None and self.task_ids.reshape(-1).shape[0] != self.observations.shape[0]:
            raise ValueError("task_ids must contain one entry per sample")

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    @classmethod
    def load(cls, path: Path, *, observation_dim: int, action_dim: int) -> "Demonstrations":
        """Load the public ``obs_vec``, ``act_vec``, ``task_id`` NPZ schema."""
        with np.load(path) as raw:
            if "obs_vec" not in raw or "act_vec" not in raw:
                raise ValueError("demonstration NPZ must contain obs_vec and act_vec")
            observations = np.asarray(raw["obs_vec"], dtype=np.float32)
            actions = np.asarray(raw["act_vec"], dtype=np.float32)
            task_ids = (
                np.asarray(raw["task_id"], dtype=np.int64).reshape(-1)
                if "task_id" in raw
                else None
            )
        dataset = cls(observations=observations, actions=actions, task_ids=task_ids)
        if observations.shape[1] != observation_dim:
            raise ValueError(f"demo observation dim {observations.shape[1]} != {observation_dim}")
        if actions.shape[1] != action_dim:
            raise ValueError(f"demo action dim {actions.shape[1]} != {action_dim}")
        return dataset

    def excluding_tasks(self, excluded: set[int]) -> "Demonstrations":
        """Return a filtered copy; provenance is mandatory for task filtering."""
        if not excluded:
            return self
        if self.task_ids is None:
            raise ValueError("cannot filter demonstrations by task without task_ids")
        keep = ~np.isin(self.task_ids, list(excluded))
        if not np.any(keep):
            raise ValueError("task filter removed every demonstration")
        return Demonstrations(self.observations[keep], self.actions[keep], self.task_ids[keep])


class TaskBatchSampler:
    """Uniform, balanced, or prior-weighted sampling over a demo dataset."""

    def __init__(
        self,
        dataset: Demonstrations,
        *,
        num_tasks: int,
        mode: str = "uniform",
        task_weights: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        if mode not in {"uniform", "balanced", "weighted"}:
            raise ValueError(f"unknown sampling mode: {mode}")
        self.dataset = dataset
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.by_task: dict[int, np.ndarray] = {}
        self.available: np.ndarray | None = None
        self.probabilities: np.ndarray | None = None
        if mode != "uniform":
            if dataset.task_ids is None:
                raise ValueError(f"{mode} sampling requires task_ids")
            for task in range(num_tasks):
                indices = np.flatnonzero(dataset.task_ids == task)
                if indices.size:
                    self.by_task[task] = indices.astype(np.int64)
            if not self.by_task:
                raise ValueError("no demonstrations match configured task IDs")
            self.available = np.asarray(sorted(self.by_task), dtype=np.int64)
            if mode == "balanced":
                self.probabilities = np.full(self.available.size, 1 / self.available.size)
            else:
                if task_weights is None:
                    raise ValueError("weighted sampling requires task_weights")
                weights = normalise_task_weights(task_weights, num_tasks)[self.available]
                available_total = float(weights.sum())
                if available_total <= 0:
                    raise ValueError("weighted sampling assigned zero mass to every available task")
                self.probabilities = weights / available_total

    def indices(self, batch_size: int) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.mode == "uniform":
            return self.rng.integers(0, self.dataset.size, size=batch_size, dtype=np.int64)
        assert self.available is not None and self.probabilities is not None
        selected_tasks = self.rng.choice(self.available, size=batch_size, p=self.probabilities)
        result = np.empty(batch_size, dtype=np.int64)
        for task in np.unique(selected_tasks):
            mask = selected_tasks == task
            result[mask] = self.rng.choice(self.by_task[int(task)], size=int(mask.sum()))
        return result

    def batch(self, batch_size: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        picked = self.indices(batch_size)
        observations = torch.as_tensor(self.dataset.observations[picked], device=device)
        actions = torch.as_tensor(self.dataset.actions[picked], device=device)
        return observations, actions
