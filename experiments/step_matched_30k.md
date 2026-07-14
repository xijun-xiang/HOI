# Step-matched 30k experiment ledger

This ledger records the core comparison summarized in
[`results/step_matched_30k_summary.csv`](../results/step_matched_30k_summary.csv).
It is a compact reproducibility map, not a claim of bit-exact replay across
hardware or HumanoidBench releases.

## Shared protocol

- Tasks: `h1hand-push-v0`, `h1hand-kitchen-v0`, `h1hand-door-v0`,
  `h1hand-package-v0`, `h1hand-cabinet-v0`, `h1hand-truck-v0`
- Training seeds: `0, 1, 2`
- Training budget: 30,000 environment steps per run
- Replay buffer / batch / target parameters: `200000 / 256 / tau=0.005`
- Discount: `gamma=0.99`
- Evaluation: deterministic, task-wise, corrected `eval10` protocol

## Controls and rationale

| Control | Task ID | Architecture / schedule | Purpose |
|---|---:|---|---|
| Vanilla MLP-SAC | no | 256x256 MLP; LR `3e-5`; starts `1000`; train frequency `1` | Shared-policy raw-return baseline. |
| Task-id MLP-SAC | yes | Same MLP baseline | Isolates the value of an explicit identity input. |
| Reference Transformer | yes | Transformer family; LR `1e-5`; starts `3000`; train frequency `8`; delayed actor updates | Capacity-matched structured-conditioning reference. |
| Task-token Transformer | yes | 8 state tokens; token dim `256`; 8 layers; 8 heads; dropout `0.1` | Tests whether a learned task prefix changes the multi-task signal. |

The original working scripts contained additional implementation controls (for
example actor KL caps and adaptive task weights). The public runner exposes
those controls as opt-in flags rather than applying them implicitly to any row
above. See [method.md](../docs/method.md) and [highlights.md](../docs/highlights.md)
for their precise public semantics.

## Interpretation boundary

Use the CSV and [results.md](../docs/results.md) together. The token control's
best task-normalized diagnostic is mainly door-shaped, while raw return remains
best for the vanilla MLP and the rich audit does not support a door-completion
claim. Package is a documented failure mode, not omitted noise.
