# Evaluation protocol

Evaluation is deterministic and task-wise. For task index `i` and episode
`e`, the evaluator uses `base_seed + 10_000 * i + e`; every seed is recorded
in the result JSON. This makes it possible to compare matching train/eval seed
pairs instead of relying on a single aggregate run.

Use the same task list, horizon, checkpoint budget, and episode count for every
control. Report raw return first, then task-normalized deltas, per-task results,
and any environment-specific behavioural proxies separately. A proxy such as
door-handle proximity must not be presented as task completion unless opening
and passage metrics confirm it.
