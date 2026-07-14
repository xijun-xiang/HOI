"""Train concise shared-policy SAC controls on a list of HumanoidBench tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from .algorithms import DemonstrationRegularization, TaskBalancedSAC
from .balancing import AdaptiveWeightConfig, TaskWeightController
from .conditioning import TaskAlignedObservation
from .data import Demonstrations, TaskBatchSampler
from .evaluation import evaluate_by_task
from .models import TaskTokenTransformer
from .policies import TaskHeadSACPolicy
from .rich_evaluation import evaluate_rich_by_task


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


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def comma_separated_floats(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return values


def comma_separated_ints(value: str) -> set[int]:
    try:
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integer task IDs") from error


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
    parser.add_argument("--task-head-critic", action="store_true")
    parser.add_argument("--task-head-mode", choices=("mlp", "bilinear"), default="mlp")
    parser.add_argument("--task-head-hidden", type=positive_int, default=128)
    parser.add_argument("--task-head-rank", type=positive_int, default=32)
    parser.add_argument("--critic-residual-scale", type=non_negative_float, default=0.1)
    parser.add_argument("--rl-weight-mode", choices=("none", "static", "adaptive"), default="none")
    parser.add_argument("--rl-task-weights", type=comma_separated_floats)
    parser.add_argument("--rl-ema-alpha", type=float, default=0.05)
    parser.add_argument("--rl-adaptive-mix", type=float, default=0.3)
    parser.add_argument("--rl-temperature", type=float, default=1.0)
    parser.add_argument("--demo-npz", type=Path)
    parser.add_argument("--demo-sampling", choices=("uniform", "balanced", "weighted"), default="uniform")
    parser.add_argument("--demo-task-weights", type=comma_separated_floats)
    parser.add_argument("--exclude-demo-tasks", type=comma_separated_ints, default=set())
    parser.add_argument("--bc-pretrain-steps", type=int, default=0)
    parser.add_argument("--bc-pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-bc-coefficient", type=non_negative_float, default=0.0)
    parser.add_argument("--actor-anchor-coefficient", type=non_negative_float, default=0.0)
    parser.add_argument("--actor-anchor-kl-coefficient", type=non_negative_float, default=0.0)
    parser.add_argument("--actor-regularization-decay-updates", type=int, default=0)
    parser.add_argument("--actor-regularization-min-scale", type=float, default=0.0)
    parser.add_argument("--actor-regularization-warmup-updates", type=int, default=0)
    parser.add_argument("--actor-freeze-updates", type=int, default=0)
    parser.add_argument("--actor-update-interval", type=positive_int, default=1)
    parser.add_argument("--actor-max-update-kl", type=non_negative_float, default=0.0)
    parser.add_argument("--rich-eval", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    if args.architecture == "task-token" and not args.task_id:
        parser.error("--architecture task-token requires --task-id")
    if (args.task_head_critic or args.rl_weight_mode != "none") and not args.task_id:
        parser.error("task-head critics and task-weighted RL require --task-id")
    if min(
        args.bc_pretrain_steps,
        args.actor_regularization_decay_updates,
        args.actor_regularization_warmup_updates,
        args.actor_freeze_updates,
    ) < 0:
        parser.error("pretraining and regularization schedules must be non-negative")
    if args.bc_pretrain_learning_rate <= 0:
        parser.error("--bc-pretrain-learning-rate must be positive")
    if not 0 < args.rl_ema_alpha <= 1:
        parser.error("--rl-ema-alpha must be in (0, 1]")
    if not 0 <= args.rl_adaptive_mix <= 1:
        parser.error("--rl-adaptive-mix must be in [0, 1]")
    if args.rl_temperature <= 0:
        parser.error("--rl-temperature must be positive")
    if not 0 <= args.actor_regularization_min_scale <= 1:
        parser.error("--actor-regularization-min-scale must be in [0, 1]")
    regularization_requested = (
        args.bc_pretrain_steps > 0
        or args.actor_bc_coefficient > 0
        or args.actor_anchor_coefficient > 0
        or args.actor_anchor_kl_coefficient > 0
    )
    if regularization_requested and args.demo_npz is None:
        parser.error("--demo-npz is required for demonstration pretraining or regularization")
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
    policy: str | type[TaskHeadSACPolicy] = "MlpPolicy"
    if args.task_head_critic:
        policy = TaskHeadSACPolicy
        policy_kwargs.update(
            {
                "task_dim": len(args.tasks),
                "task_head_mode": args.task_head_mode,
                "task_head_hidden": args.task_head_hidden,
                "task_head_rank": args.task_head_rank,
                "residual_scale": args.critic_residual_scale,
            }
        )
    model = TaskBalancedSAC(
        policy,
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
    run_notes: dict[str, object] = {}
    if args.rl_weight_mode != "none":
        controller = TaskWeightController(
            task_dim=len(args.tasks),
            base_weights=args.rl_task_weights,
            config=AdaptiveWeightConfig(
                mode=args.rl_weight_mode,
                ema_alpha=args.rl_ema_alpha,
                adaptive_mix=args.rl_adaptive_mix,
                temperature=args.rl_temperature,
            ),
            device=torch.device(model.device),
        )
        model.configure_task_weighting(controller)
        run_notes["task_weighting"] = {
            "mode": args.rl_weight_mode,
            "base_weights": controller.base.detach().cpu().tolist(),
        }
    actor_controls = DemonstrationRegularization(
        behavior_coefficient=args.actor_bc_coefficient,
        anchor_coefficient=args.actor_anchor_coefficient,
        anchor_kl_coefficient=args.actor_anchor_kl_coefficient,
        decay_updates=args.actor_regularization_decay_updates,
        minimum_scale=args.actor_regularization_min_scale,
        warmup_updates=args.actor_regularization_warmup_updates,
        freeze_updates=args.actor_freeze_updates,
        update_interval=args.actor_update_interval,
        max_update_kl=args.actor_max_update_kl,
    )
    if args.demo_npz is not None:
        observation_dim = int(np.prod(vec_env.observation_space.shape))
        action_dim = int(np.prod(vec_env.action_space.shape))
        demonstrations = Demonstrations.load(
            args.demo_npz, observation_dim=observation_dim, action_dim=action_dim
        ).excluding_tasks(args.exclude_demo_tasks)
        sampler = TaskBatchSampler(
            demonstrations,
            num_tasks=len(args.tasks),
            mode=args.demo_sampling,
            task_weights=(
                np.asarray(args.demo_task_weights, dtype=np.float64)
                if args.demo_task_weights is not None
                else None
            ),
            seed=args.seed,
        )
        model.configure_demonstrations(sampler, regularization=actor_controls)
        run_notes["demonstrations"] = {
            "path": str(args.demo_npz),
            "samples": demonstrations.size,
            "sampling": args.demo_sampling,
            "excluded_task_ids": sorted(args.exclude_demo_tasks),
            "regularization": vars(actor_controls),
        }
        if args.bc_pretrain_steps:
            losses = model.pretrain_actor(
                steps=args.bc_pretrain_steps,
                batch_size=args.batch_size,
                learning_rate=args.bc_pretrain_learning_rate,
            )
            run_notes["bc_pretraining"] = {
                "steps": args.bc_pretrain_steps,
                "mean_loss": float(np.mean(losses)),
                "final_loss": losses[-1],
            }
    else:
        model.configure_demonstrations(None, regularization=actor_controls)
    run_notes["actor_update_controls"] = vars(actor_controls)
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
        rich_evaluations = (
            evaluate_rich_by_task(
                model, args.tasks, make_eval_env, episodes=args.episodes, seed=args.eval_seed
            )
            if args.rich_eval
            else None
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
            "run_notes": run_notes,
        }
        if rich_evaluations is not None:
            summary["rich_evaluation"] = rich_evaluations
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
