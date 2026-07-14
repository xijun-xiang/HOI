# What this release extracts

This is a curated research artifact, not a wholesale export of the original
experiment workspace. The selection below prioritizes code that is both central
to the dissertation's reasoning and independently auditable.

| Highlight | Public implementation | Evidence status |
|---|---|---|
| Heterogeneous-task shared SAC | aligned vector observations and task-ID controls | evaluated in the committed 30k comparison |
| Structured task conditioning | learned task-token Transformer | evaluated in the committed 30k comparison |
| Task-specific value correction | residual MLP/bilinear critic heads | implementation and component tests; evaluate as a new ablation |
| Interference-aware optimisation | static/adaptive task-balanced replay losses | implementation and component tests; evaluate as a new ablation |
| Demonstration-to-RL transfer | validated NPZ data, BC pretraining, action/KL anchoring | implementation and component tests; no private data released |
| Conservative actor updates | warmup/decay/freeze/delay and KL rejection | implementation and integration smoke test |
| Claim-disciplined evaluation | fixed task/seed evaluator and raw/normalized comparison | evaluated in the committed 30k comparison |
| Behavioural proxy audit | numeric `info` aggregation without success relabelling | implementation and component test |

The results table in [results.md](results.md) applies only to the controls
named there. Optional methods are deliberately not described as validated
improvements until their matched task, seed, compute, and rich-evaluation
ablations are added.
