"""Policy feature extractors used by the task-conditioned controls."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class TaskTokenTransformer(BaseFeaturesExtractor):
    """Encode aligned state tokens together with a learned task token.

    The final ``task_dim`` coordinates of an observation are a one-hot task
    identity. They are projected into a learned prefix token; the remaining
    coordinates are projected into state tokens and encoded by self-attention.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        features_dim: int = 256,
        task_dim: int,
        token_dim: int = 128,
        num_state_tokens: int = 8,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        if len(observation_space.shape) != 1:
            raise ValueError("TaskTokenTransformer expects a flat observation")
        if task_dim < 1:
            raise ValueError("task_dim must be positive for task-token conditioning")
        if token_dim % heads:
            raise ValueError("token_dim must be divisible by heads")
        total_dim = int(np.prod(observation_space.shape))
        state_dim = total_dim - task_dim
        if state_dim < 1:
            raise ValueError("observation must contain state coordinates before task identity")
        super().__init__(observation_space, features_dim)
        self.state_dim = state_dim
        self.task_dim = task_dim
        self.token_dim = token_dim
        self.state_projection = nn.Linear(state_dim, num_state_tokens * token_dim)
        self.task_projection = nn.Linear(task_dim, token_dim)
        self.position = nn.Parameter(torch.zeros(1, num_state_tokens + 1, token_dim))
        block = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.float().reshape(observations.shape[0], -1)
        state = observations[:, : self.state_dim]
        task = observations[:, self.state_dim :]
        state_tokens = self.state_projection(state).reshape(
            observations.shape[0], -1, self.token_dim
        )
        task_token = self.task_projection(task).unsqueeze(1)
        tokens = torch.cat((task_token, state_tokens), dim=1) + self.position
        return self.head(self.encoder(tokens)[:, 0])
