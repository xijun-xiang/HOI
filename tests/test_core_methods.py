import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from hoi.algorithms import DemonstrationRegularization, TaskBalancedSAC
from hoi.balancing import AdaptiveWeightConfig, TaskWeightController
from hoi.data import Demonstrations, TaskBatchSampler
from hoi.policies import TaskHeadSACPolicy
from hoi.rich_evaluation import evaluate_rich_by_task


def test_task_weight_controller_tracks_difficulty_and_normalises_samples():
    controller = TaskWeightController(
        task_dim=2,
        base_weights=[1.0, 3.0],
        config=AdaptiveWeightConfig(mode="static"),
        device=torch.device("cpu"),
    )
    observations = torch.tensor([[0.2, 1.0, 0.0], [0.1, 0.0, 1.0]])
    controller.update(observations, torch.tensor([[0.0], [0.0]]))
    weights = controller.sample_weights(observations).squeeze(1)
    assert torch.allclose(weights, torch.tensor([0.5, 1.5]))

    adaptive = TaskWeightController(
        task_dim=2,
        config=AdaptiveWeightConfig(mode="adaptive", ema_alpha=1.0, adaptive_mix=1.0),
        device=torch.device("cpu"),
    )
    adaptive.update(observations, torch.tensor([[-5.0], [5.0]]))
    assert adaptive.current[0] > adaptive.current[1]


def test_demo_loader_filter_and_balanced_sampler(tmp_path):
    path = tmp_path / "demos.npz"
    np.savez(
        path,
        obs_vec=np.arange(18, dtype=np.float32).reshape(6, 3),
        act_vec=np.arange(6, dtype=np.float32).reshape(6, 1),
        task_id=np.array([0, 0, 0, 0, 1, 1]),
    )
    demos = Demonstrations.load(path, observation_dim=3, action_dim=1).excluding_tasks({1})
    sampler = TaskBatchSampler(demos, num_tasks=2, mode="balanced", seed=3)
    observations, actions = sampler.batch(4, device=torch.device("cpu"))
    assert observations.shape == (4, 3)
    assert actions.shape == (4, 1)


def test_task_head_policy_constructs_and_scores_task_conditioned_observations():
    observation_space = spaces.Box(-1, 1, shape=(5,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    policy = TaskHeadSACPolicy(
        observation_space,
        action_space,
        lambda _: 3e-4,
        task_dim=2,
        task_head_mode="bilinear",
        net_arch=[16],
    )
    observations = torch.tensor([[0.0, 0.1, 0.2, 1.0, 0.0], [0.2, 0.1, 0.0, 0.0, 1.0]])
    values = policy.critic(observations, torch.zeros(2, 2))
    assert len(values) == 2
    assert all(value.shape == (2, 1) for value in values)


class TinyTaskEnvironment(gym.Env):
    observation_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        return np.array([0.0, 1.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        observation = np.array([float(np.clip(action[0], -1, 1)), 0.0, 1.0], dtype=np.float32)
        return observation, 1.0, self.step_count >= 2, False, {"proxy": self.step_count}

    def close(self):
        pass


def test_task_balanced_sac_runs_weighted_and_demo_regularised_update():
    env = DummyVecEnv([TinyTaskEnvironment])
    model = TaskBalancedSAC(
        "MlpPolicy",
        env,
        learning_starts=1,
        buffer_size=32,
        batch_size=4,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [16]},
        seed=0,
    )
    controller = TaskWeightController(
        task_dim=2,
        config=AdaptiveWeightConfig(mode="adaptive"),
        device=model.device,
    )
    model.configure_task_weighting(controller)
    demos = Demonstrations(
        observations=np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (8, 1)),
        actions=np.zeros((8, 1), dtype=np.float32),
        task_ids=np.zeros(8, dtype=np.int64),
    )
    model.configure_demonstrations(
        TaskBatchSampler(demos, num_tasks=2, mode="balanced"),
        regularization=DemonstrationRegularization(
            behavior_coefficient=0.1,
            anchor_coefficient=0.1,
            anchor_kl_coefficient=0.1,
            warmup_updates=1,
            decay_updates=4,
            minimum_scale=0.2,
            update_interval=2,
            max_update_kl=1e-12,
        ),
    )
    assert len(model.pretrain_actor(steps=2, batch_size=4, learning_rate=1e-3)) == 2
    model.learn(total_timesteps=12)
    assert model._n_updates > 0
    env.close()


class MetricEnvironment(TinyTaskEnvironment):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info.update({"opening": 0.25 * self.step_count, "note": "ignored"})
        return observation, reward, terminated, truncated, info


class ZeroPolicy:
    def predict(self, observation, deterministic=True):
        return np.zeros(1, dtype=np.float32), None


def test_rich_evaluation_preserves_numeric_info_without_claiming_success():
    reports = evaluate_rich_by_task(
        ZeroPolicy(), ["tiny"], lambda _task, _index: MetricEnvironment(), episodes=2, seed=7
    )
    assert reports[0]["info"]["opening"]["final"] == 0.5
    assert reports[0]["info"]["proxy"]["mean"] == 1.5
