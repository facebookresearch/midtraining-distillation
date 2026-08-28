#!/usr/bin/env bash
# OLMo-2 1B SFT — 4-node H200 multi-node version.
#
# Same recipe as train_olmo2_1b_sft.sh (effective batch 128, LR 3e-5, 2 epochs).
# Different parallelism: 32 GPUs across 4 nodes via accelerate's DeepSpeed
# multi-node launcher. Per-device batch and grad_accum recalculated so the
# effective batch matches the single-node baseline.
#
# Expected wallclock: ~1.5-2h (vs ~6h single-node), but actual speedup will
# be sub-linear due to cross-node gradient sync.
#
# Submit:
#   sbatch post_training/scripts/train_olmo2_1b_sft_4node.sh
#
# Output dir lives under $DOLMINO_BASELINE_DIR/post_training/<exp_name>/.
# The DPO and RLVR wrappers consume that path.

#SBATCH --job-name=olmo2_1b_sft_4node
#SBATCH --output=${SLURM_LOG_DIR}/olmo2_1b_sft_4node-%j.out
#SBATCH --error=${SLURM_LOG_DIR}/olmo2_1b_sft_4node-%j.err
#SBATCH --time=06:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=4
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
# Compute nodes can't reliably reach huggingface.co -> the tulu-3-sft dataset HEAD
# check times out (5 retries then FAIL). The dataset is cached under HF_HOME, so
# force offline to skip the Hub round-trip. (Hub flakiness has killed SFT jobs
# outright, so this is on by default.)
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_PROJECT=olmo2_1b_post_training
# Override on the sbatch CLI: --export=ALL,WANDB_API_KEY
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
# wandb offline (proxy unreachable from compute nodes; see single-node script note)
export WANDB_MODE=offline
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300

# ----- multi-node topology -----
NNODES=${SLURM_JOB_NUM_NODES}              # 4 from sbatch header
GPUS_PER_NODE=8
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))     # 32

# Head node = first in the SLURM allocation
mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE=${ALL_NODES[0]}
HEAD_NODE_IP=$(getent hosts "${HEAD_NODE}" | awk '{print $1}')
# Pick a high random port to avoid collisions with sibling jobs.
MAIN_PORT=$((20000 + RANDOM % 10000))

# Standard NCCL / torch.distributed env that DeepSpeed expects.
export MASTER_ADDR="${HEAD_NODE_IP}"
export MASTER_PORT="${MAIN_PORT}"
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22
# Cluster-specific: interface used for inter-node NCCL. `eth0` suits a
# plain-Ethernet cluster; InfiniBand sites usually need something else
# (check `ip link`). Unset entirely to let NCCL auto-detect.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

# ----- inputs -----
BASE_CKPT=${BASE_CKPT:-${MIDTRAIN_CKPT}/checkpoints/0000028800/hf}
EXP_NAME=${EXP_NAME:-olmo2_1b_baseline28800_sft_4node}
OUT_ROOT=${OUT_ROOT:-${MIDTRAIN_CKPT}/post_training}
OUTPUT_DIR="${OUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"

cd ${OPEN_INSTRUCT_ROOT}
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== SFT 4-NODE ==="
echo "BASE_CKPT=${BASE_CKPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HEAD_NODE=${HEAD_NODE}  HEAD_NODE_IP=${HEAD_NODE_IP}  MAIN_PORT=${MAIN_PORT}"
echo "NNODES=${NNODES}  TOTAL_GPUS=${TOTAL_GPUS}"

# Match the published effective batch (128) by re-balancing across more GPUs.
#   Single-node: 8 GPUs × per_device=2 × grad_accum=8 = 128
#   4-node:     32 GPUs × per_device=2 × grad_accum=2 = 128
# Per-device batch stays at 2 (same memory footprint), grad_accum drops 8 → 2.
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-2}
GRAD_ACCUM=${GRAD_ACCUM:-2}
# SFT LR default 5e-6 -- NOT the Tulu-3 default of 3e-5. See post_training/README.md.
# Was 3e-5 — a stray env override trained idx200 (q20-13B) SFT at 5e-5 and
# permanently cratered recall/MC (triviaqa 51.4->41.7, arc-c 58.8->45.7), caught
# 2026-07-29. The pipeline (pipeline_post_training.sh) also hard-pins 5e-6.
LEARNING_RATE=${LEARNING_RATE:-5e-6}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-2}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-epoch}
EFFECTIVE_BATCH=$((TOTAL_GPUS * PER_DEVICE_BATCH * GRAD_ACCUM))
echo "=== Batch size config ==="
echo "  per_device_batch_size: ${PER_DEVICE_BATCH}"
echo "  gradient_accumulation_steps: ${GRAD_ACCUM}"
echo "  total_gpus: ${TOTAL_GPUS}"
echo "  EFFECTIVE BATCH SIZE: ${EFFECTIVE_BATCH}"
echo "  learning_rate: ${LEARNING_RATE}"
echo "  num_train_epochs: ${NUM_TRAIN_EPOCHS}"
echo "  checkpointing_steps: ${CHECKPOINTING_STEPS}"
echo "  (Published baseline: 8 GPUs × 2 batch × 8 accum = 128)"
echo "  (This run:          ${TOTAL_GPUS} GPUs × ${PER_DEVICE_BATCH} batch × ${GRAD_ACCUM} accum = ${EFFECTIVE_BATCH})"
echo "========================"
if [ "${EFFECTIVE_BATCH}" -ne 128 ]; then
    echo "WARNING: effective batch ${EFFECTIVE_BATCH} differs from published 128 — LR may need rescaling" >&2
fi

# Shared finetune.py args for both the cache prepass and the real run.
FT_ARGS=(
    --exp_name "${EXP_NAME}"
    --model_name_or_path "${BASE_CKPT}"
    --tokenizer_name "${BASE_CKPT}"
    --use_slow_tokenizer False
    --add_bos
    --chat_template_name tulu
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0
    --max_seq_length 4096
    --per_device_train_batch_size ${PER_DEVICE_BATCH}
    --gradient_accumulation_steps ${GRAD_ACCUM}
    --learning_rate ${LEARNING_RATE}
    --lr_scheduler_type linear
    --warmup_ratio 0.03
    --weight_decay 0.0
    --num_train_epochs ${NUM_TRAIN_EPOCHS}
    --output_dir "${OUTPUT_DIR}"
    --checkpointing_steps ${CHECKPOINTING_STEPS}
    --report_to wandb
    --with_tracking
    --logging_steps 1
    --seed 1
    --push_to_hub False
    --try_launch_beaker_eval_jobs False
)

# Cache prepass: single process on the head node only. The dataset cache
# lives on shared FS so all training workers can read it.
echo "=== SFT 4-node: prebuilding dataset cache (single process, head node) ==="
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    open_instruct/finetune.py \
    "${FT_ARGS[@]}" \
    --cache_dataset_only

# Stage FT_ARGS to disk so the srun heredoc can re-load it as a real array.
# Writing one arg per line (NUL-safe via -d '') would be ideal, but no arg
# contains a newline so plain `printf "%s\n"` is enough — and mapfile -t reads
# it back cleanly. This avoids the text-serialization bug where unquoted
# ${FT_ARGS[@]} inside a "..." srun heredoc word-splits values mid-flag.
FT_ARGS_FILE="${OUTPUT_DIR}/.ft_args_${SLURM_JOB_ID}.txt"
printf '%s\n' "${FT_ARGS[@]}" > "${FT_ARGS_FILE}"
echo "=== SFT 4-node: staged FT_ARGS to ${FT_ARGS_FILE} (${#FT_ARGS[@]} elements) ==="

# Multi-node training launch. srun fans out one launcher per node, and
# `accelerate launch` on each node spawns its 8 worker processes with the
# correct global rank derived from --machine_rank=$SLURM_NODEID.
echo "=== SFT 4-node: launching distributed training across ${NNODES} nodes ==="
srun --nodes=${NNODES} --ntasks=${NNODES} --ntasks-per-node=1 \
    bash -lc "
        source ${VENV_PATH}/bin/activate
        cd ${OPEN_INSTRUCT_ROOT}
        export PYTHONPATH=\"\$(pwd):\${PYTHONPATH:-}\"
        mapfile -t FT_ARGS < '${FT_ARGS_FILE}'
        accelerate launch \\
            --mixed_precision bf16 \\
            --num_machines ${NNODES} \\
            --num_processes ${TOTAL_GPUS} \\
            --machine_rank \${SLURM_NODEID} \\
            --main_process_ip ${HEAD_NODE_IP} \\
            --main_process_port ${MAIN_PORT} \\
            --use_deepspeed \\
            --deepspeed_config_file configs/ds_configs/stage2_no_offloading_accelerate.conf \\
            --deepspeed_multinode_launcher standard \\
            open_instruct/finetune.py \\
            \"\${FT_ARGS[@]}\" \\
            --timeout 14400
    "

rm -f "${FT_ARGS_FILE}"

echo "=== Copying special_tokens_map.json from base checkpoint ==="
cp "${BASE_CKPT}/special_tokens_map.json" "${OUTPUT_DIR}/${EXP_NAME}/" || echo "WARNING: Could not copy special_tokens_map.json"

# RoPE schema normalization — see single-node script for the full why.
echo "=== Normalizing RoPE config schema for older inference stacks ==="
python ${POST_TRAINING_SCRIPTS}/normalize_rope_config.py "${OUTPUT_DIR}/${EXP_NAME}" \
    || { echo "ERROR: normalize_rope_config.py failed; eval will produce degenerate output" >&2; exit 1; }

echo "=== SFT 4-node done -> ${OUTPUT_DIR} ==="

# Auto-launch the 10-task OLMES eval (same pattern as single-node).
if [ "${AUTO_EVAL:-1}" = "1" ]; then
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OLMES_ROOT}/results/posttrain/${EXP_NAME}}"
    echo "=== Auto-submitting OLMES eval -> ${EVAL_OUTPUT_DIR} ==="
    # NO `sbatch --export=`: under cgroup v2 an explicit --export (NONE or ALL)
    # makes slurmd harvest the login env via `su`, which hangs and holds the job at
    # Priority=0 (user_env_retrieval_failed_requeued_held) forever. The old --export=NONE
    # was here to keep this job's open-instruct venv PATH out of the child eval, so we
    # reproduce that by scrubbing the venv vars and pinning a system PATH in sbatch's own
    # environment, then letting the default --export=ALL propagate it. /usr/bin/sbatch is
    # absolute because the pinned PATH no longer contains whatever provided sbatch.
    env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
        PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        HOME="${HOME}" USER="${USER}" \
        MODEL_PATH="${OUTPUT_DIR}/${EXP_NAME}" OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
        /usr/bin/sbatch \
        ${OLMES_ROOT}/eval_single_posttrain.sh \
        || echo "WARNING: eval sbatch failed (training output is still saved at ${OUTPUT_DIR}/${EXP_NAME})"
fi
