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

## Rich proxy audit

Pass `--rich-eval` to additionally save `rich_evaluation` in `summary.json`.
For every numeric scalar in an environment's `info` dictionary, the evaluator
records the episode-final, mean, minimum, and maximum value, then averages
those statistics task-wise. Strings, arrays, and nested values are excluded
rather than silently coerced.

Metric names and semantics remain the environment's own. In particular, a
positive `opening`, proximity, or contact proxy is evidence about that proxy;
it is not automatically renamed to `success`. Report an explicit completion
metric separately when the environment provides one.
