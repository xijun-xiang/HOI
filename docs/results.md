# Step-matched 30k results

The curated result table in `results/step_matched_30k_summary.csv` summarizes
the corrected 30,000-step evaluation across training seeds 0, 1, and 2 on the
six-task suite. Each checkpoint family used the same corrected eval10 protocol.

| Control | Mean raw return | Task-normalized delta | Interpretation |
|---|---:|---:|---|
| Vanilla MLP-SAC | -1130.631 | 0.00000 | Strongest raw-return control. |
| Task-id MLP-SAC | -1131.392 | +0.03328 | Nearly ties raw control; task IDs alone explain part of the signal. |
| Reference Transformer | -1143.532 | +0.04049 | Positive scale-adjusted signal, lower raw return. |
| Task-token Transformer | -1140.645 | +0.07611 | Largest normalized, door-shaped signal; not a raw-return win. |

This is **not** a state-of-the-art claim. Rich proxy evaluation found higher
average door hand-hatch proximity for the task-token control, but door opening
and passage remained effectively zero. Package returns also worsened on every
paired seed in the primary audit. The useful contribution is therefore the
controlled comparison and diagnostic protocol, not a claim that task tokens
solve humanoid object interaction.
