"""Task-scale-aware RL loss weighting with an auditable adaptive schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import normalise_task_weights


@dataclass(frozen=True)
class AdaptiveWeightConfig:
    """Controls a blend from a declared task prior toward observed difficulty."""

    mode: str = "static"
    ema_alpha: float = 0.05
    adaptive_mix: float = 0.3
    temperature: float = 1.0
    logit_clip: float = 4.0
    normalise_sample_weights: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"static", "adaptive"}:
            raise ValueError("mode must be static or adaptive")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0 <= self.adaptive_mix <= 1:
            raise ValueError("adaptive_mix must be in [0, 1]")
        if self.temperature <= 0 or self.logit_clip <= 0:
            raise ValueError("temperature and logit_clip must be positive")


class TaskWeightController:
    """Maps a final task one-hot suffix to static or adaptive per-sample weights."""

    def __init__(
        self,
        *,
        task_dim: int,
        base_weights: list[float] | np.ndarray | None = None,
        config: AdaptiveWeightConfig = AdaptiveWeightConfig(),
        device: torch.device,
    ) -> None:
        if task_dim < 1:
            raise ValueError("task_dim must be positive")
        self.task_dim = task_dim
        self.config = config
        prior = (
            np.ones(task_dim, dtype=np.float64) / task_dim
            if base_weights is None
            else normalise_task_weights(base_weights, task_dim)
        )
        self.base = torch.as_tensor(prior, dtype=torch.float32, device=device)
        self.current = self.base.clone()
        self.reward_ema = torch.zeros(task_dim, dtype=torch.float32, device=device)

    def task_onehot(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] < self.task_dim:
            raise ValueError("observations do not contain the configured task one-hot suffix")
        onehot = observations[:, -self.task_dim :]
        if torch.any(onehot.sum(dim=1) <= 0):
            raise ValueError("task one-hot suffix contains an unassigned sample")
        return onehot

    @torch.no_grad()
    def update(self, observations: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Update difficulty weights; lower reward EMA means higher difficulty."""
        onehot = self.task_onehot(observations)
        if self.config.mode == "static":
            self.current = self.base.clone()
            return self.current
        mass = onehot.sum(dim=0)
        reward_sum = (onehot * rewards.reshape(-1, 1)).sum(dim=0)
        seen = mass > 0
        if torch.any(seen):
            means = reward_sum[seen] / mass[seen].clamp_min(1)
            self.reward_ema[seen] = (
                (1 - self.config.ema_alpha) * self.reward_ema[seen]
                + self.config.ema_alpha * means
            )
        difficulty = -self.reward_ema
        z_scores = (difficulty - difficulty.mean()) / difficulty.std(unbiased=False).clamp_min(1e-6)
        adaptive = torch.softmax(
            torch.clamp(z_scores / self.config.temperature, -self.config.logit_clip, self.config.logit_clip),
            dim=0,
        )
        mixed = (1 - self.config.adaptive_mix) * self.base + self.config.adaptive_mix * adaptive
        self.current = mixed / mixed.sum().clamp_min(1e-6)
        return self.current

    def sample_weights(self, observations: torch.Tensor) -> torch.Tensor:
        onehot = self.task_onehot(observations)
        weights = (onehot * self.current.unsqueeze(0)).sum(dim=1, keepdim=True).clamp_min(1e-6)
        if self.config.normalise_sample_weights:
            weights = weights / weights.mean().detach().clamp_min(1e-6)
        return weights

    def diagnostics(self) -> dict[str, float | list[float]]:
        entropy = -(self.current * self.current.clamp_min(1e-8).log()).sum()
        return {
            "weights": self.current.detach().cpu().tolist(),
            "minimum": float(self.current.min().detach().cpu()),
            "maximum": float(self.current.max().detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }
