"""Task-conditioned critic heads for interpretable shared-policy ablations."""

from __future__ import annotations

from typing import Any, Optional

import torch
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import SACPolicy
from torch import nn


class TaskResidualCritic(ContinuousCritic):
    """Twin Q-functions plus small task-specific residual heads.

    The shared critic remains the primary estimator.  Each residual sees the
    learned state/action representation and the final task one-hot suffix,
    which makes the inductive bias easy to switch off or ablate.
    """

    def __init__(
        self,
        *args: Any,
        task_dim: int,
        task_head_mode: str = "mlp",
        task_head_hidden: int = 128,
        task_head_rank: int = 32,
        residual_scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        if task_dim < 1:
            raise ValueError("task_dim must be positive")
        if task_head_mode not in {"mlp", "bilinear"}:
            raise ValueError("task_head_mode must be 'mlp' or 'bilinear'")
        if task_head_hidden < 1 or task_head_rank < 1:
            raise ValueError("task head dimensions must be positive")
        super().__init__(*args, **kwargs)
        action_dim = int(self.action_space.shape[0])
        input_dim = self.features_extractor.features_dim + action_dim
        self.task_dim = task_dim
        self.task_head_mode = task_head_mode
        self.residual_scale = float(residual_scale)
        if task_head_mode == "mlp":
            self.residual_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(input_dim + task_dim, task_head_hidden),
                        nn.ReLU(),
                        nn.Linear(task_head_hidden, 1),
                    )
                    for _ in range(self.n_critics)
                ]
            )
        else:
            self.residual_heads = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "state_action": nn.Linear(input_dim, task_head_rank),
                            "task": nn.Linear(task_dim, task_head_rank, bias=False),
                            "output": nn.Linear(task_head_rank, 1),
                        }
                    )
                    for _ in range(self.n_critics)
                ]
            )

    def _task_identity(self, observations: torch.Tensor) -> torch.Tensor:
        flat = observations.float().reshape(observations.shape[0], -1)
        if flat.shape[1] < self.task_dim:
            raise ValueError("critic observations are missing the task one-hot suffix")
        task = flat[:, -self.task_dim :]
        if torch.any(task.sum(dim=1) <= 0):
            raise ValueError("critic received an observation without a task assignment")
        return task

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(observations, self.features_extractor)
        state_action = torch.cat([features, actions], dim=1)
        task = self._task_identity(observations)
        values: list[torch.Tensor] = []
        for base_q, head in zip(self.q_networks, self.residual_heads):
            base_value = base_q(state_action)
            if self.task_head_mode == "mlp":
                residual = head(torch.cat([state_action, task], dim=1))
            else:
                assert isinstance(head, nn.ModuleDict)
                residual = head["output"](
                    head["state_action"](state_action) * head["task"](task)
                )
            values.append(base_value + self.residual_scale * residual)
        return tuple(values)


class TaskHeadSACPolicy(SACPolicy):
    """SAC policy whose twin critic uses :class:`TaskResidualCritic`."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        lr_schedule: Schedule,
        *,
        task_dim: int,
        task_head_mode: str = "mlp",
        task_head_hidden: int = 128,
        task_head_rank: int = 32,
        residual_scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.task_dim = task_dim
        self.task_head_mode = task_head_mode
        self.task_head_hidden = task_head_hidden
        self.task_head_rank = task_head_rank
        self.residual_scale = residual_scale
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> TaskResidualCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return TaskResidualCritic(
            **critic_kwargs,
            task_dim=self.task_dim,
            task_head_mode=self.task_head_mode,
            task_head_hidden=self.task_head_hidden,
            task_head_rank=self.task_head_rank,
            residual_scale=self.residual_scale,
        ).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        parameters = super()._get_constructor_parameters()
        parameters.update(
            {
                "task_dim": self.task_dim,
                "task_head_mode": self.task_head_mode,
                "task_head_hidden": self.task_head_hidden,
                "task_head_rank": self.task_head_rank,
                "residual_scale": self.residual_scale,
            }
        )
        return parameters
