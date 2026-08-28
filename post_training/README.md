# Post-training: SFT → DPO → RLVR1 → RLVR2

Reproduces the OLMo-2 1B post-training pipeline on top of a mid-training
checkpoint produced by this repo, so the paper's "do the KD gains survive
post-training?" question can be answered for any recipe.

These are launch scripts, not a fork. You clone upstream open-instruct, make
the small set of changes listed below, and run the Slurm scripts here.

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

> **Local modifications are not distributed.** Our runs used a lightly patched
> open-instruct. Those patches are not part of this release. Upstream assumes
> Ai2's Beaker infrastructure and Hub-resident models, so a few changes are
> needed to run on plain Slurm from a local checkpoint. The important ones:
>
> | File | Change | Why |
> |---|---|---|
> | `finetune.py` | Skip `snapshot_download` when `model_name_or_path` is a local directory | `snapshot_download` only accepts Hub repo ids, so **training from a local mid-trained checkpoint fails outright without this**. The one change you cannot skip. |
> | `finetune.py` | Additionally save an HF-format snapshot per epoch (e.g. `hf_epoch_{n}`) | `accelerator.save_state` writes resume-state only; it is not loadable by vLLM / `from_pretrained`, so intermediate epochs cannot be evaluated. Use a prefix that `clean_last_n_checkpoints` will not try to parse as an int. |
> | `utils.py` | Make `import beaker` optional; have `maybe_use_ai2_wandb_entity` read `$WANDB_ENTITY` instead of calling `wandb.login()` to look up the `ai2-llm` team | Removes a hard Beaker dependency and a network round-trip on the import path. |
> | `grpo_fast.py` | Add `VIRTUAL_ENV`, `PYTHONHOME`, `PYTHONPATH`, `UV_*` to Ray's excluded env vars, and exclude `.venv/` from the runtime env | Otherwise Ray ships the uv venv to workers and they come up with a broken interpreter. Required for the RLVR stages. |
> | `grpo_fast.py` | After each RL checkpoint save, rewrite the OLMo2 `config.json` to flat `rope_theta` / `rope_scaling` (dropping `rope_parameters`) | Otherwise vLLM silently falls back to `rope_theta=10000` and eval output degenerates. Derive `bos`/`eos`/`pad` from the saved tokenizer rather than hardcoding them. The same fix is available standalone as `scripts/normalize_rope_config.py`, which the SFT and DPO scripts already call. |
> | `model_utils.py` | Guard `importlib.util.find_spec("flash_attn.cute")` in `try/except ModuleNotFoundError`; rewrite `tokenizer_class: "TokenizersBackend"` to `"GPT2Tokenizer"` on save | `find_spec` *raises* when the parent package is missing; `TokenizersBackend` is not a real HF class and breaks vLLM. |
> | `vllm_utils.py` | Snapshot `self.active_tasks.values()` before iterating | Other threads mutate the dict during iteration. |

## 2. Configure

The scripts read these from the environment — there are no hardcoded paths:

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

One command chains all four stages with `--dependency=afterok`:

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

**The SFT learning rate is the single most important knob here, and it is not
the Tulu-3 default.** Tulu-3 ships 3e-5, which is calibrated for an
NTP-initialized model. On a KD-distilled mid-train checkpoint it erases most of
the retained-capability gain (≈8 macro points at SFT; DROP −10.8, TriviaQA −5.2)
and the damage compounds through DPO. We use **5e-6**; anything ≤5e-6
essentially eliminates the regression. See §5.1 of the paper.

The RL learning rates are hardcoded in the stage scripts rather than inherited,
deliberately: `--export=ALL` otherwise leaks the SFT LR into the RL stages.

`train_watchdog.sh` resubmits a stage on `afternotok` (resuming from
`--checkpoint_state_dir`) up to `MAX_RETRIES`, which matters for the ~43 h RLVR1
stage.

## 4. Evaluation

The `AUTO_EVAL` hooks in each stage script shell out to OLMES wrappers via
`${OLMES_ROOT}`. Those wrappers are **not part of this release** — see
`../evaluation/` for the eval setup and task configuration instead. Run with
`NO_EVAL=1` (or leave `OLMES_ROOT` unset; the hooks no-op with a warning) and
evaluate separately.

## Not included

Three superseded scripts from our tree were deliberately left out:
`train_olmo2_1b_rlvr1.sh` and `train_olmo2_1b_rlvr1_4node.sh` (both were `exit 1`
stubs) and the single-node `train_olmo2_1b_sft.sh` (superseded by the 4-node
variant, which uses grad-accum 2 instead of 8 to reach the same effective batch).
No model weights are released.
