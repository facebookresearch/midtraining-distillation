#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# OLMo-2 1B DPO on top of the SFT checkpoint produced by train_olmo2_1b_sft.sh.
#
# Recipe (1B DPO block, on-policy preference mix) from docs/olmo2.md:
#   - Dataset: allenai/olmo-2-0425-1b-preference-mix
#   - 1 node x 8 H100, ~2h
#   - DeepSpeed Stage 2, per_device_batch=8, grad_accum=2, lr=2.5e-6, 1 epoch
#   - dpo_norm, beta=5, --add_bos, --chat_template_name tulu
#
# Submit:
#   SFT_CKPT=/path/to/sft/output \
#   sbatch post_training/scripts/train_olmo2_1b_dpo.sh

#SBATCH --job-name=olmo2_1b_dpo
#SBATCH --output=${SLURM_LOG_DIR}/olmo2_1b_dpo-%j.out
#SBATCH --error=${SLURM_LOG_DIR}/olmo2_1b_dpo-%j.err
#SBATCH --time=06:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --requeue

# Directory holding these release scripts (this file's own directory), so
# normalize_rope_config.py and siblings resolve without assuming a layout
# inside the open-instruct checkout.
POST_TRAINING_SCRIPTS="${POST_TRAINING_SCRIPTS:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

set -euo pipefail

# ----- env -----
VENV_PATH="${OPEN_INSTRUCT_ROOT:?set OPEN_INSTRUCT_ROOT}/.venv"
source "$VENV_PATH/bin/activate"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=olmo2_1b_post_training
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# Point WANDB_BASE_URL at a self-hosted W&B if you use one.
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300

# Override weka paths to use local checkpoint storage
export REFERENCE_LOGPROBS_CACHE_PATH=${OPEN_INSTRUCT_ROOT}/.cache/reference_logprobs_cache

# Use all 8 GPUs (adjust NUM_GPUS for debugging)
NUM_GPUS=8

# ----- inputs -----
OUT_ROOT=${OUT_ROOT:-${MIDTRAIN_CKPT}/post_training}
SFT_CKPT=${SFT_CKPT:-${OUT_ROOT}/olmo2_1b_baseline28800_sft/olmo2_1b_baseline28800_sft}
EXP_NAME=${EXP_NAME:-olmo2_1b_baseline28800_dpo}
OUTPUT_DIR="${OUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"
# Offline wandb logs land alongside the output dir for easy `wandb sync` later.
export WANDB_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"

if [[ ! -d "${SFT_CKPT}" ]]; then
    echo "ERROR: SFT_CKPT does not exist: ${SFT_CKPT}"; exit 1
fi

cd ${OPEN_INSTRUCT_ROOT}

echo "=== DPO ==="
echo "SFT_CKPT=${SFT_CKPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}

EFFECTIVE_BATCH=$((NUM_GPUS * PER_DEVICE_BATCH * GRAD_ACCUM))
echo "=== Batch size config ==="
echo "  per_device_batch_size: ${PER_DEVICE_BATCH}"
echo "  gradient_accumulation_steps: ${GRAD_ACCUM}"
echo "  num_gpus: ${NUM_GPUS}"
echo "  EFFECTIVE BATCH SIZE: ${EFFECTIVE_BATCH}"
echo "========================"

# Shared DPO args for both cache prepass and real run
DPO_ARGS=(
    --exp_name "${EXP_NAME}"
    --model_name_or_path "${SFT_CKPT}"
    --tokenizer_name_or_path "${SFT_CKPT}"
    --use_slow_tokenizer False
    --add_bos
    --chat_template_name tulu
    --mixer_list allenai/olmo-2-0425-1b-preference-mix 1.0
    --max_seq_length 2048
    --per_device_train_batch_size ${PER_DEVICE_BATCH}
    --gradient_accumulation_steps ${GRAD_ACCUM}
    --learning_rate 2.5e-6
    --lr_scheduler_type linear
    --warmup_ratio 0.1
    --weight_decay 0.0
    --num_epochs 1
    --output_dir "${OUTPUT_DIR}"
    --use_lora False
    --loss_type dpo_norm
    --beta 5
    --activation_memory_budget 0.5
    --report_to wandb
    --with_tracking
    --logging_steps 1
    --seed 111
    --push_to_hub False
    --try_launch_beaker_eval_jobs False
)

echo "=== DPO: prebuilding dataset cache (single process) ==="
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/dpo_tune_cache.py \
    "${DPO_ARGS[@]}" \
    --cache_dataset_only

echo "=== DPO: launching distributed training ==="
accelerate launch \
    --mixed_precision bf16 \
    --num_processes ${NUM_GPUS} \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage2_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard \
    open_instruct/dpo_tune_cache.py \
    "${DPO_ARGS[@]}"


echo "=== Normalizing RoPE config schema for older inference stacks ==="
python ${POST_TRAINING_SCRIPTS}/normalize_rope_config.py "${OUTPUT_DIR}/${EXP_NAME}" \
    || { echo "ERROR: normalize_rope_config.py failed; eval will produce degenerate output" >&2; exit 1; }

echo "=== DPO done -> ${OUTPUT_DIR} ==="

# Auto-launch the 10-task OLMES eval on the DPO output. EVAL_OUTPUT_DIR can be
# overridden; default is olmes/results/posttrain/${EXP_NAME}. Set AUTO_EVAL=0
# to opt out.
if [ "${AUTO_EVAL:-1}" = "1" ]; then
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OLMES_ROOT}/results/posttrain/${EXP_NAME}}"
    echo "=== Auto-submitting OLMES eval -> ${EVAL_OUTPUT_DIR} ==="
    env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
        PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        HOME="${HOME}" USER="${USER}" \
        MODEL_PATH="${OUTPUT_DIR}/${EXP_NAME}" OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
        /usr/bin/sbatch \
        ${OLMES_ROOT}/eval_single_posttrain.sh \
        || echo "WARNING: eval sbatch failed (training output is still saved at ${OUTPUT_DIR}/${EXP_NAME})"
fi
