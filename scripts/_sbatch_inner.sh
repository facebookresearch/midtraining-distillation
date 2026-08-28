#!/bin/bash
# scripts/_sbatch_inner.sh — runs INSIDE sbatch / srun. Sets up env, then
# torchruns apps.main.train with the chosen recipe YAML.
#
# Expects launch_midtrain.sh to have sourced scripts/env.sh first, so
# LINGUA_CONDA_ENV / etc. are already exported into this script's environment
# via sbatch --export=ALL,...
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/env.sh"

if [ -n "${CONDA_PROFILE_SH:-}" ] && [ -f "${CONDA_PROFILE_SH}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_PROFILE_SH}"
else
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "${LINGUA_CONDA_ENV}"

# Wandb / tuning env.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}
NNODES=${SLURM_NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-$((29501 + RANDOM % 5000))}
NODE_RANK=${SLURM_NODEID:-0}

if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
fi

echo "NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NODE_RANK=$NODE_RANK"
echo "RECIPE=$RECIPE CONFIG=$RECIPE_YAML STEPS=$STEPS_OVERRIDE FP8=$LINGUA_TEACHER_FP8 COMPILE=$LINGUA_COMPILE_TEACHER"

cd "${ROOT_DIR}"

TRAIN_CMD=(torchrun
    --nnodes="${NNODES}"
    --nproc-per-node="${NPROC_PER_NODE}"
    --node-rank="${NODE_RANK}"
    --master-addr="${MASTER_ADDR}"
    --master-port="${MASTER_PORT}"
    -m apps.main.train
    "config=${RECIPE_YAML}"
    "steps=${STEPS_OVERRIDE}")

if [ -n "${SLURM_JOB_NODELIST:-}" ] && [ "$NNODES" -gt 1 ]; then
    # Multi-node fan-out.
    srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
        --export=ALL,NPROC_PER_NODE,NNODES,MASTER_ADDR,MASTER_PORT,LINGUA_TEACHER_FP8,LINGUA_COMPILE_TEACHER,RECIPE,RECIPE_YAML,STEPS_OVERRIDE,ROOT_DIR,DATA_ROOT,TEACHER_1B_PATH,TEACHER_7B_PATH,STUDENT_INIT_PATH,STUDENT_HF_PATH,TOKENIZER_PATH,MIDTRAIN_ROOT,EVAL_ROOT,SLURM_LOG_DIR,LINGUA_CONDA_ENV,OLMES_CONDA_ENV,WANDB_API_KEY,WANDB_MODE,WANDB_ENTITY,CUDA_DEVICE_MAX_CONNECTIONS,TORCH_NCCL_AVOID_RECORD_STREAMS,PYTORCH_CUDA_ALLOC_CONF \
        bash -c "export NODE_RANK=\${SLURM_PROCID}; ${TRAIN_CMD[*]}"
else
    "${TRAIN_CMD[@]}"
fi
