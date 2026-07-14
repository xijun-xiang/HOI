"""Train concise shared-policy SAC controls on a list of HumanoidBench tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from .conditioning import TaskAlignedObservation
from .evaluation import evaluate_by_task
from .models import TaskTokenTransformer


def comma_separated(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=comma_separated, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=positive_int, default=30_000)
    parser.add_argument("--num-envs", type=positive_int, default=None)
    parser.add_argument("--episodes", type=positive_int, default=10)
    parser.add_argument("--eval-seed", type=int, default=27_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--architecture", choices=("mlp", "task-token"), default="mlp")
    parser.add_argument("--task-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--train-freq", type=positive_int, default=1)
    parser.add_argument("--gradient-steps", type=positive_int, default=1)
    parser.add_argument("--buffer-size", type=positive_int, default=200_000)
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--token-dim", type=positive_int, default=128)
    parser.add_argument("--tokens", type=positive_int, default=8)
    parser.add_argument("--layers", type=positive_int, default=2)
    parser.add_argument("--heads", type=positive_int, default=4)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    if args.architecture == "task-token" and not args.task_id:
        parser.error("--architecture task-token requires --task-id")
    return args


def load_humanoid_bench() -> None:
    try:
        import humanoid_bench  # noqa: F401  # Registers Gymnasium environments.
    except ModuleNotFoundError as error:
        raise SystemExit(
            "HumanoidBench is required. Install it first, for example: "
            "pip install git+https://github.com/carlosferrazza/humanoid-bench.git"
        ) from error


def inspect_tasks(task_names: list[str]) -> tuple[int, tuple[int, ...]]:
    """Return maximum state dimension and a common action shape."""
    dimensions: list[int] = []
    action_shape: tuple[int, ...] | None = None
    for task_name in task_names:
        env = gym.make(task_name, render_mode=None)
        try:
            if not isinstance(env.observation_space, spaces.Box):
                raise TypeError(f"{task_name} does not provide a Box observation space")
            if not isinstance(env.action_space, spaces.Box):
                raise TypeError(f"{task_name} does not provide a Box action space")
            dimensions.append(int(np.prod(env.observation_space.shape)))
            if action_shape is None:
                action_shape = env.action_space.shape
            elif env.action_space.shape != action_shape:
                raise ValueError(
                    f"{task_name} action shape {env.action_space.shape} differs from {action_shape}"
                )
        finally:
            env.close()
    if action_shape is None:
        raise ValueError("no tasks supplied")
    return max(dimensions), action_shape


def environment_factory(
    task_name: str,
    task_index: int,
    *,
    num_tasks: int,
    target_dim: int,
    append_task_id: bool,
    seed: int,
) -> Callable[[], gym.Env]:
    def make() -> gym.Env:
        env = gym.make(task_name, render_mode=None)
        env = TaskAlignedObservation(
            env,
            task_index=task_index,
            num_tasks=num_tasks,
            target_dim=target_dim,
            append_task_id=append_task_id,
        )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return make


def main() -> None:
    args = parse_args()
    load_humanoid_bench()
    set_random_seed(args.seed)
    target_dim, _ = inspect_tasks(args.tasks)
    num_envs = args.num_envs or len(args.tasks)
    factories = [
        environment_factory(
            args.tasks[index % len(args.tasks)],
            index % len(args.tasks),
            num_tasks=len(args.tasks),
            target_dim=target_dim,
            append_task_id=args.task_id,
            seed=args.seed + index,
        )
        for index in range(num_envs)
    ]
    vec_env = DummyVecEnv(factories)

    policy_kwargs: dict[str, object] = {"net_arch": [256, 256]}
    if args.architecture == "task-token":
        policy_kwargs.update(
            features_extractor_class=TaskTokenTransformer,
            features_extractor_kwargs={
                "features_dim": 256,
                "task_dim": len(args.tasks),
                "token_dim": args.token_dim,
                "num_state_tokens": args.tokens,
                "layers": args.layers,
                "heads": args.heads,
            },
        )
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_kwargs=policy_kwargs,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.learn(total_timesteps=args.steps, progress_bar=False)
        model.save(args.out_dir / "model")

        def make_eval_env(task_name: str, task_index: int) -> gym.Env:
            return environment_factory(
                task_name,
                task_index,
                num_tasks=len(args.tasks),
                target_dim=target_dim,
                append_task_id=args.task_id,
                seed=args.seed + 50_000 + task_index,
            )()

        evaluations = evaluate_by_task(
            model, args.tasks, make_eval_env, episodes=args.episodes, seed=args.eval_seed
        )
        summary = {
            "config": {
                **vars(args),
                "out_dir": str(args.out_dir),
                "target_observation_dim": target_dim,
                "num_envs": num_envs,
            },
            "evaluation": evaluations,
            "mean_return": float(np.mean([entry["mean_return"] for entry in evaluations])),
        }
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
