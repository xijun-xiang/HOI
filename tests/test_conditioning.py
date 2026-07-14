import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from hoi.conditioning import TaskAlignedObservation, build_task_onehot
from hoi.models import TaskTokenTransformer


class TinyEnvironment(gym.Env):
    observation_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.array([1.0, 2.0, 3.0], dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0.0, True, False, {}


def test_onehot_and_observation_alignment():
    wrapper = TaskAlignedObservation(
        TinyEnvironment(), task_index=1, num_tasks=3, target_dim=5, append_task_id=True
    )
    observation, _ = wrapper.reset()
    assert wrapper.observation_space.contains(observation)
    assert observation.tolist() == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    np.testing.assert_array_equal(build_task_onehot(0, 2), np.array([1.0, 0.0]))


def test_task_token_transformer_shape():
    space = spaces.Box(-1, 1, shape=(11,), dtype=np.float32)
    extractor = TaskTokenTransformer(
        space,
        features_dim=16,
        task_dim=3,
        token_dim=8,
        num_state_tokens=2,
        layers=1,
        heads=2,
    )
    output = extractor(torch.randn(4, 11))
    assert output.shape == (4, 16)
