# Post-training: SFT → DPO → RLVR1 → RLVR2

Reproduces the OLMo-2 1B post-training pipeline on top of a mid-training
checkpoint produced by this repo.


## 1. Set up open-instruct

```bash
git clone https://github.com/allenai/open-instruct.git
cd open-instruct
git checkout 63834749cbd3119666f40931d41a41fcffed4972    # the SHA the paper used
export OPEN_INSTRUCT_ROOT=$PWD
bash /path/to/midtraining-distillation/post_training/scripts/setup_env.sh
```

`setup_env.sh` creates a `uv` venv (`uv sync --frozen`) under
`${OPEN_INSTRUCT_ROOT}/.venv` and smoke-tests the imports. Run it once, on a
login node.


## 2. Configure

The scripts read these paths from the environment:

| Variable | Meaning |
|---|---|
| `OPEN_INSTRUCT_ROOT` | your patched open-instruct checkout (required) |
| `POST_TRAINING_ROOT` | where run outputs go (required) |
| `SLURM_LOG_DIR` | where sbatch `.out`/`.err` land (required) |
| `WANDB_API_KEY` | optional; runs default to `WANDB_MODE=offline` |
| `OLMES_ROOT` | optional; only for the auto-eval hooks — see §4 |

Every `#SBATCH --account` / `--qos` / `--partition` line in `scripts/*.sh` reads
`CHANGE_ME`. **You must edit those for your cluster before submitting.** Also
check `NCCL_SOCKET_IFNAME` in `scripts/train_olmo2_1b_sft_4node.sh` — it defaults to
`eth0`, which is wrong on most InfiniBand sites (`ip link` will tell you).

## 3. Run

We use a single command that chains all four stages with `--dependency=afterok`:

```bash
RUN_TAG=switch_distill \
BASE_CKPT=${MIDTRAIN_ROOT}/switch_distill/checkpoints/0000028800/hf \
bash scripts/pipeline_post_training.sh
```

`DRY_RUN=1` prints the sbatch invocations without submitting. Partial runs:
`SKIP_RLVR2=1` (stop after RLVR1), `SKIP_RLVR=1` (stop after DPO),
`SKIP_DPO=1` (SFT only), or `SKIP_SFT=1 SFT_CKPT=<dir>` to resume mid-chain.
`NO_WATCHDOG=1` and `NO_EVAL=1` disable the sidecar jobs.

Output layout under `${POST_TRAINING_ROOT}/${RUN_TAG}/`:

```
${RUN_TAG}_sft/${RUN_TAG}_sft/     ← leaf HF dir
${RUN_TAG}_dpo/${RUN_TAG}_dpo/     ← leaf HF dir
${RUN_TAG}_rlvr1/                  ← parent; grpo nests __1__<ts>_checkpoints/step_N
${RUN_TAG}_rlvr2/                  ← same nesting
```

SFT and DPO leaf paths are deterministic and computed up front, so the chain can
be submitted in one shot. RLVR2 cannot know RLVR1's `__1__<timestamp>` directory
at submit time, so it resolves it at runtime and takes the highest-numbered
`step_*` checkpoint that exists.

### Stage geometry and hyperparameters

| Stage | Nodes | LR | Notes |
|---|---|---|---|
| SFT | 4 × 8 | **5e-6** | `tulu-3-sft-olmo-2-mixture`, 2 epochs, effective batch 128 |
| DPO | 1 × 8 | 2.5e-6 | `olmo-2-0425-1b-preference-mix`, `dpo_norm`, β=5, 1 epoch |
| RLVR1 | 4 × 8 | 5e-7 | 24 learners + 4 vLLM engines (TP=2); `RLVR-GSM-MATH-IF-Mixed-Constraints` |
| RLVR2 | 4 × 8 | 5e-7 | same geometry; `allenai/RLVR-MATH` |

`train_watchdog.sh` resubmits a stage on `afternotok` (resuming from
`--checkpoint_state_dir`) up to `MAX_RETRIES`.

## 4. Evaluation

The `AUTO_EVAL` hooks in each stage script shell out to OLMES wrappers via
`${OLMES_ROOT}`. Those wrappers are **not part of this release** — see
`../evaluation/` for the eval setup and task configuration instead. Run with
`NO_EVAL=1` (or leave `OLMES_ROOT` unset; the hooks no-op with a warning) and
evaluate separately.
