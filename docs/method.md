# Method

## Problem

The release studies a single SAC policy across six HumanoidBench
whole-body-object-interaction tasks: push, kitchen, door, package, cabinet,
and truck. The central difficulty is that their state vectors have different
lengths and their return scales are not comparable.

## Shared-policy interface

`TaskAlignedObservation` pads every vector state to the largest task dimension.
The two task-aware controls append a one-hot task identity:

1. **Vanilla MLP-SAC**: aligned state only.
2. **Task-id MLP-SAC**: aligned state plus a task one-hot vector.
3. **Task-token Transformer-SAC**: the one-hot vector is projected into a
   learned prefix token and attended jointly with projected state tokens.

The implementation deliberately keeps the interface explicit: each task is
wrapped independently, all actions must share the same shape, and the final
policy input is always a flat `float32` vector.

## Task-conditioned critic residual

`TaskHeadSACPolicy` retains the ordinary pair of SAC Q-functions and adds a
small residual head to each. Each head receives the critic's state/action
representation and the final task one-hot suffix:

`Q_i(s, a, t) = Q_i_shared(s, a) + lambda * Delta_i(s, a, t)`.

The residual can be an MLP or a low-rank bilinear interaction. `lambda` is
explicit (`--critic-residual-scale`, default `0.1`), so the shared critic is
still the primary estimator. This is an ablation-friendly conditioning choice,
not a claim that every task needs its own independent critic.

## Task-balanced RL losses

`TaskWeightController` reads only the final one-hot task suffix. In static
mode it applies a declared prior. In adaptive mode it keeps a reward EMA per
task, maps lower reward to higher difficulty through a temperature-controlled
softmax, then mixes that distribution with the prior. The resulting sample
weights are normalized to have batch mean one before they weight both critic
squared error and actor SAC loss.

This mechanism addresses loss-scale domination; it does not make returns
comparable. Raw and task-normalized evaluation remain separate diagnostics.

## Conservative IL-to-RL transfer

The optional demonstration path accepts only a simple public NPZ schema. It
supports deterministic action-regression pretraining and three explicitly
logged RL-time retention terms:

1. action MSE on sampled demonstrations;
2. action MSE to a frozen actor snapshot; and
3. KL from the current diagonal-Gaussian actor to that snapshot.

The regularization scale can warm up, decay linearly to a floor, or remain
constant. Actor updates can be delayed, initially frozen, and reverted when
the post-update distributional KL exceeds `--actor-max-update-kl`. Critic and
entropy updates continue on every gradient step. These are stabilisation
controls to be tested in matched ablations, not defaults hidden in the
baseline.

## Why task-scale-aware evaluation

Raw mean return is a necessary benchmark score, but one task with a much larger
return magnitude can dominate it. `hoi.analyze` therefore also reports each
task delta divided by the absolute baseline return for that task (with a floor
of one). This diagnostic is reported alongside—not instead of—raw returns.
