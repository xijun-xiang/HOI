# Public demonstration interface

The repository intentionally does not ship large demonstration archives. To
run the IL-to-RL components, provide an NPZ with this compact schema:

| Key | Shape | Required | Meaning |
|---|---|---:|---|
| `obs_vec` | `(N, observation_dim)` | yes | Final, task-aligned policy inputs. The last `task_dim` entries must be the task one-hot when task-aware methods are used. |
| `act_vec` | `(N, action_dim)` | yes | Actions in the policy's normalized action space (normally `[-1, 1]`). |
| `task_id` | `(N,)` | no | Integer task provenance. Required for balanced/weighted demo sampling or `--exclude-demo-tasks`. |

`Demonstrations.load` checks all ranks, sample counts, and vector dimensions
before training. It does not inspect private paths, transitions, rewards, or
unlisted arrays. That is deliberate: action-supervision experiments should
start from data whose alignment and action scaling are explicit.

## Example

```bash
hoi-train --tasks "$TASKS" --out-dir artifacts/il-rl \
  --demo-npz data/public_demos.npz --demo-sampling balanced \
  --bc-pretrain-steps 5000 --bc-pretrain-learning-rate 3e-4 \
  --actor-bc-coefficient 0.1 --actor-anchor-coefficient 0.05 \
  --actor-anchor-kl-coefficient 0.01 \
  --actor-regularization-warmup-updates 1000 \
  --actor-regularization-decay-updates 10000 \
  --actor-regularization-min-scale 0.1 \
  --actor-update-interval 2 --actor-max-update-kl 0.01
```

The output `summary.json` records the data path, sample count, sampling mode,
declared exclusions, pretraining loss summary, and all regularization controls.
Datasets and frozen actor snapshots are intentionally not embedded in
`model.zip`; reconfigure them explicitly when resuming an IL-to-RL study.
