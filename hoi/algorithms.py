"""SAC extensions for task-balanced RL and conservative IL-to-RL transfer."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac import SAC
from torch.nn import functional as functional

from .balancing import TaskWeightController
from .data import TaskBatchSampler


@dataclass(frozen=True)
class DemonstrationRegularization:
    """Demonstration retention and conservative actor-update controls."""

    behavior_coefficient: float = 0.0
    anchor_coefficient: float = 0.0
    anchor_kl_coefficient: float = 0.0
    decay_updates: int = 0
    minimum_scale: float = 0.0
    warmup_updates: int = 0
    freeze_updates: int = 0
    update_interval: int = 1
    max_update_kl: float = 0.0

    def __post_init__(self) -> None:
        if min(self.behavior_coefficient, self.anchor_coefficient, self.anchor_kl_coefficient) < 0:
            raise ValueError("regularization coefficients must be non-negative")
        if min(self.decay_updates, self.warmup_updates, self.freeze_updates) < 0:
            raise ValueError("regularization schedules must be non-negative")
        if not 0 <= self.minimum_scale <= 1:
            raise ValueError("minimum_scale must be in [0, 1]")
        if self.update_interval < 1:
            raise ValueError("update_interval must be positive")
        if self.max_update_kl < 0:
            raise ValueError("max_update_kl must be non-negative")

    def scale(self, updates: int) -> float:
        """Use a warmup, then linearly decay toward the declared floor."""
        if updates < self.warmup_updates or self.decay_updates == 0:
            return 1.0
        progress = min(1.0, (updates - self.warmup_updates) / self.decay_updates)
        return max(self.minimum_scale, 1.0 - progress)

    def actor_due(self, updates: int) -> bool:
        """Delay or freeze actor optimisation without changing critic updates."""
        if updates < self.freeze_updates:
            return False
        return (updates - self.freeze_updates) % self.update_interval == 0


def diagonal_gaussian_kl(
    mean_p: torch.Tensor,
    log_std_p: torch.Tensor,
    mean_q: torch.Tensor,
    log_std_q: torch.Tensor,
) -> torch.Tensor:
    """Per-sample ``KL(N_p || N_q)`` for diagonal Gaussian actor outputs."""
    variance_p = torch.exp(2 * log_std_p)
    variance_q = torch.exp(2 * log_std_q).clamp_min(1e-8)
    return torch.sum(
        log_std_q - log_std_p
        + (variance_p + (mean_p - mean_q).pow(2)) / (2 * variance_q)
        - 0.5,
        dim=1,
        keepdim=True,
    )


class TaskBalancedSAC(SAC):
    """SAC with optional task-weighted losses and demonstration retention.

    Default construction is ordinary SAC loss aggregation. The additions are
    configured explicitly after construction so a run manifest shows every
    non-baseline intervention.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.task_weight_controller: TaskWeightController | None = None
        self.demo_sampler: TaskBatchSampler | None = None
        self.demo_regularization = DemonstrationRegularization()
        self._actor_anchor: torch.nn.Module | None = None

    def configure_task_weighting(self, controller: TaskWeightController | None) -> None:
        """Enable task-balanced replay loss weighting, or restore uniform loss."""
        self.task_weight_controller = controller

    def configure_demonstrations(
        self,
        sampler: TaskBatchSampler | None,
        *,
        regularization: DemonstrationRegularization = DemonstrationRegularization(),
    ) -> None:
        """Set a public demonstration sampler for BC and policy-anchor terms."""
        needs_demos = any(
            coefficient > 0
            for coefficient in (
                regularization.behavior_coefficient,
                regularization.anchor_coefficient,
                regularization.anchor_kl_coefficient,
            )
        )
        if sampler is None and needs_demos:
            raise ValueError("demonstrations are required when regularization is non-zero")
        self.demo_sampler = sampler
        self.demo_regularization = regularization
        self._actor_anchor = None
        if sampler is not None and (
            regularization.anchor_coefficient > 0 or regularization.anchor_kl_coefficient > 0
        ):
            self.snapshot_actor_anchor()

    def snapshot_actor_anchor(self) -> None:
        """Freeze the current actor as the reference for conservative RL updates."""
        # Rebuild through the policy rather than ``deepcopy``: SB3 action
        # distributions retain recent non-leaf tensors after a forward pass.
        self._actor_anchor = self.policy.make_actor().to(self.device)
        self._actor_anchor.load_state_dict(self.actor.state_dict())
        self._actor_anchor.eval()
        for parameter in self._actor_anchor.parameters():
            parameter.requires_grad_(False)

    def pretrain_actor(self, *, steps: int, batch_size: int, learning_rate: float) -> list[float]:
        """Run deterministic action-regression pretraining on configured demos."""
        if self.demo_sampler is None:
            raise ValueError("configure demonstrations before actor pretraining")
        if steps < 1 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("steps, batch_size, and learning_rate must be positive")
        optimizer = self.actor.optimizer.__class__(self.actor.parameters(), lr=learning_rate)
        self.actor.set_training_mode(True)
        losses: list[float] = []
        for _ in range(steps):
            observations, actions = self.demo_sampler.batch(batch_size, device=self.device)
            loss = functional.mse_loss(self.actor(observations, deterministic=True), actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if (
            self.demo_regularization.anchor_coefficient > 0
            or self.demo_regularization.anchor_kl_coefficient > 0
        ):
            self.snapshot_actor_anchor()
        return losses

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "task_weight_controller",
            "demo_sampler",
            "_actor_anchor",
        ]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Optimise SAC with optional task and demonstration loss terms."""
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        behavior_losses: list[float] = []
        anchor_losses: list[float] = []
        anchor_kl_losses: list[float] = []
        actor_update_kls: list[float] = []
        actor_updates = 0
        actor_rejections = 0

        for gradient_step in range(gradient_steps):
            update_index = self._n_updates + gradient_step
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if self.use_sde:
                self.actor.reset_noise()

            if self.task_weight_controller is None:
                sample_weights = torch.ones_like(replay_data.rewards)
            else:
                self.task_weight_controller.update(replay_data.observations, replay_data.rewards)
                sample_weights = self.task_weight_controller.sample_weights(replay_data.observations)

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(float(ent_coef_loss.detach().cpu()))
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.detach().cpu()))
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * (
                    next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                (sample_weights * (current_q - target_q_values).pow(2)).mean()
                for current_q in current_q_values
            )
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(float(critic_loss.detach().cpu()))

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (sample_weights * (ent_coef * log_prob - min_qf_pi)).mean()

            scale = self.demo_regularization.scale(update_index)
            if self.demo_sampler is not None and self.demo_regularization.behavior_coefficient > 0:
                demo_observations, demo_actions = self.demo_sampler.batch(batch_size, device=self.device)
                behavior_loss = functional.mse_loss(
                    self.actor(demo_observations, deterministic=True), demo_actions
                )
                actor_loss = actor_loss + scale * self.demo_regularization.behavior_coefficient * behavior_loss
                behavior_losses.append(float(behavior_loss.detach().cpu()))
            if self._actor_anchor is not None and self.demo_regularization.anchor_coefficient > 0:
                with torch.no_grad():
                    anchored_actions = self._actor_anchor(replay_data.observations, deterministic=True)
                anchor_loss = functional.mse_loss(
                    self.actor(replay_data.observations, deterministic=True), anchored_actions
                )
                actor_loss = actor_loss + scale * self.demo_regularization.anchor_coefficient * anchor_loss
                anchor_losses.append(float(anchor_loss.detach().cpu()))
            if self._actor_anchor is not None and self.demo_regularization.anchor_kl_coefficient > 0:
                current_mean, current_log_std, _ = self.actor.get_action_dist_params(
                    replay_data.observations
                )
                with torch.no_grad():
                    anchor_mean, anchor_log_std, _ = self._actor_anchor.get_action_dist_params(
                        replay_data.observations
                    )
                anchor_kl_loss = (
                    sample_weights
                    * diagonal_gaussian_kl(
                        current_mean, current_log_std, anchor_mean, anchor_log_std
                    )
                ).mean()
                actor_loss = (
                    actor_loss
                    + scale * self.demo_regularization.anchor_kl_coefficient * anchor_kl_loss
                )
                anchor_kl_losses.append(float(anchor_kl_loss.detach().cpu()))

            if self.demo_regularization.actor_due(update_index):
                old_mean = old_log_std = None
                actor_state = optimizer_state = None
                if self.demo_regularization.max_update_kl > 0:
                    with torch.no_grad():
                        old_mean, old_log_std, _ = self.actor.get_action_dist_params(
                            replay_data.observations
                        )
                    actor_state = {
                        key: value.detach().clone() for key, value in self.actor.state_dict().items()
                    }
                    optimizer_state = copy.deepcopy(self.actor.optimizer.state_dict())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                rejected = False
                if old_mean is not None and old_log_std is not None:
                    with torch.no_grad():
                        new_mean, new_log_std, _ = self.actor.get_action_dist_params(
                            replay_data.observations
                        )
                        update_kl = diagonal_gaussian_kl(
                            old_mean, old_log_std, new_mean, new_log_std
                        ).mean()
                    actor_update_kls.append(float(update_kl.detach().cpu()))
                    if float(update_kl.detach().cpu()) > self.demo_regularization.max_update_kl:
                        assert actor_state is not None and optimizer_state is not None
                        self.actor.load_state_dict(actor_state)
                        self.actor.optimizer.load_state_dict(optimizer_state)
                        rejected = True
                if rejected:
                    actor_rejections += 1
                else:
                    actor_updates += 1
            actor_losses.append(float(actor_loss.detach().cpu()))

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if behavior_losses:
            self.logger.record("train/demo_behavior_loss", np.mean(behavior_losses))
        if anchor_losses:
            self.logger.record("train/actor_anchor_loss", np.mean(anchor_losses))
        if anchor_kl_losses:
            self.logger.record("train/actor_anchor_kl_loss", np.mean(anchor_kl_losses))
        self.logger.record("train/actor_update_interval", self.demo_regularization.update_interval)
        self.logger.record("train/actor_update_ratio", actor_updates / gradient_steps)
        self.logger.record("train/actor_update_reject_ratio", actor_rejections / gradient_steps)
        if actor_update_kls:
            self.logger.record("train/actor_update_kl", np.mean(actor_update_kls))
        if self.task_weight_controller is not None:
            diagnostics = self.task_weight_controller.diagnostics()
            self.logger.record("train/task_weight_min", diagnostics["minimum"])
            self.logger.record("train/task_weight_max", diagnostics["maximum"])
            self.logger.record("train/task_weight_entropy", diagnostics["entropy"])
