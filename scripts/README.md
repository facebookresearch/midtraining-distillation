# scripts/

Launch scripts. The public entry points are `launch_midtrain.sh` (mid-training
from the 1B student init) and `launch_pretrain.sh` (from scratch); the rest is
plumbing.

## Files

- `launch_midtrain.sh` — small `sbatch` wrapper. Takes `RECIPE=<name>` as an
  env var, resolves the matching YAML under
  `../apps/main/configs/midtrain_recipes/<name>.yaml`, and submits a 4-node H200 sbatch
  job. Supports `DRY_RUN=1` to print the sbatch command without submitting.
- `launch_pretrain.sh` — same, for `../apps/main/configs/pretrain_recipes/`.
- `_sbatch_inner.sh` — the script that actually runs inside the sbatch
  allocation. Activates the `${LINGUA_CONDA_ENV}` conda env, sets the standard wandb /
  CUDA env, derives `NPROC_PER_NODE` / `MASTER_ADDR` / `MASTER_PORT` /
  `NODE_RANK` from SLURM, and `torchrun`s `apps.main.train` with the chosen
  recipe YAML.

## Common usage

```bash
# DRY_RUN: print sbatch invocation, don't submit.
RECIPE=ntp_baseline DRY_RUN=1 bash scripts/launch_midtrain.sh

# Submit a 4-node mid-training run for the headline entropy-switched recipe.
RECIPE=switch_distill bash scripts/launch_midtrain.sh

# Override nodes (e.g. for a 1-node smoke test, though wallclock will balloon).
RECIPE=fkd NNODES=1 bash scripts/launch_midtrain.sh

# Override total steps for a quick sanity check (default 28800).
RECIPE=ntp_baseline STEPS_OVERRIDE=200 bash scripts/launch_midtrain.sh
```

## Recipe discovery

```bash
ls ../apps/main/configs/midtrain_recipes/ | sed 's/\.yaml$//'
```

## Other scripts

- `env.sh` — single source of truth for every path. Source it in each shell.
- `fetch_models.sh` — downloads the teachers + student init and converts the
  student to Lingua DCP, so it exists in both HF and DCP layouts.
- `smoke_test.py` — asserts the HF→Lingua-DCP conversion is numerically
  faithful (matching top-5 predictions on a fixed prompt). Run it after
  `fetch_models.sh`.
- `lingua_to_hf.sh` — DCP → consolidated → HF for a trained checkpoint.
- `hf_generate.py` — sample completions from a converted HF dir, to confirm the
  conversion round-tripped.
