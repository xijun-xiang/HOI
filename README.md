# HOI: Task-Conditioned SAC for Humanoid Object Interaction

This repository is the compact, open-source research artifact for a
compute-conscious study of multi-task humanoid object interaction on
[HumanoidBench](https://github.com/carlosferrazza/humanoid-bench). It keeps the
parts that are useful to inspect and reproduce:

- a shared-policy SAC training loop for heterogeneous humanoid tasks;
- explicit task-aligned observations and task-ID controls;
- a task-token Transformer feature extractor;
- deterministic, task-wise evaluation with recorded seeds; and
- raw and task-scale-aware result analysis.

It intentionally excludes private writing, large experiment directories,
checkpoints, W&B logs, and external demonstration datasets. See
[NOTICE.md](NOTICE.md) for dependencies and attribution.

## Research question and result

Can structured task conditioning reduce interference in six HumanoidBench
hand-object tasks (push, kitchen, door, package, cabinet, truck) under a fixed
compute budget?

The answer supported by the 30k-step, three-seed comparison is nuanced:

- vanilla MLP-SAC remains the strongest **raw-return** control;
- task-ID MLP-SAC nearly matches it, so a simple identity input is essential;
- task-token conditioning has the largest **task-normalized** diagnostic delta,
  driven mainly by door-shaped reward; and
- this signal does not establish task completion: rich evaluation found no
  meaningful door opening/passage and a consistent package-task failure mode.

This is a method-and-evaluation contribution, not a state-of-the-art claim.
The exact result summary and its limitations are in [docs/results.md](docs/results.md).

## Install

Create a Python 3.10+ environment, install this package, then install the
benchmark separately:

```bash
pip install -e ".[dev]"
pip install git+https://github.com/carlosferrazza/humanoid-bench.git
```

HumanoidBench has its own MuJoCo, JAX, and GPU setup requirements. Follow its
installation guide for the target platform before starting a long run.

## Reproduce a compact control

All commands use a six-task suite; lower `--steps` first to validate a machine.

```bash
TASKS=h1hand-push-v0,h1hand-kitchen-v0,h1hand-door-v0,h1hand-package-v0,h1hand-cabinet-v0,h1hand-truck-v0

# Task-agnostic, shared MLP control
hoi-train --tasks "$TASKS" --out-dir artifacts/vanilla --no-task-id --architecture mlp --steps 30000 --seed 0

# Simple task-ID control
hoi-train --tasks "$TASKS" --out-dir artifacts/task-id --task-id --architecture mlp --steps 30000 --seed 0

# Explicit learned task-token control
hoi-train --tasks "$TASKS" --out-dir artifacts/task-token --task-id --architecture task-token --steps 30000 --seed 0
```

Each run saves `model.zip` and a task-wise `summary.json`. Compare two matching
summaries without hiding return-scale differences:

```bash
hoi-analyze \
  --baseline artifacts/vanilla/summary.json \
  --candidate artifacts/task-token/summary.json \
  --out artifacts/task-token/comparison.json
```

For the complete method, evaluation protocol, and reported 30k result:

- [Method](docs/method.md)
- [Evaluation protocol](docs/evaluation.md)
- [Results and claim boundaries](docs/results.md)

## Repository layout

```text
hoi/
  conditioning.py  # observation padding and task IDs
  models.py        # task-token Transformer feature extractor
  train.py         # shared-policy SAC runner
  evaluation.py    # deterministic, task-wise evaluator
  analysis.py      # raw and task-normalized comparisons
docs/              # method, protocol, and transparent results
results/           # small, human-readable aggregate table only
tests/             # dependency-light component tests
```

## Citation

If this repository supports your work, please cite the associated dissertation
and the HumanoidBench benchmark. A machine-readable starting point is provided
in [CITATION.cff](CITATION.cff); replace the dissertation title and archival
identifier with the final approved metadata before a formal release.

## License

The code in this repository is available under the [MIT License](LICENSE).
HumanoidBench and Stable-Baselines3 are separate MIT-licensed dependencies and
must be cited according to their respective repositories.
