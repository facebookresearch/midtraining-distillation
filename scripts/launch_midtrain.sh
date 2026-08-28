#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# scripts/launch_midtrain.sh — launch a mid-training run via sbatch.
#
# Usage:
#   RECIPE=<name> bash scripts/launch_midtrain.sh                # submit via sbatch
#   RECIPE=<name> DRY_RUN=1 bash scripts/launch_midtrain.sh      # print the sbatch
#                                                                # command without
#                                                                # submitting
#
# RECIPE values are the YAML basenames under apps/main/configs/midtrain_recipes/:
#   ntp_baseline               vanilla next-token prediction (no teacher)
#   fkd                        forward KL
#   rkd                        reverse KL (MiniLLM-style)
#   switch_distill             switch distillation (the headline method): partitions
#                              tokens between CE and reverse KL by teacher entropy
# See README.md for the loss forms. For from-scratch runs use
# scripts/launch_pretrain.sh, which reads apps/main/configs/pretrain_recipes/.
#
# The student is always the 1B model; the teacher is chosen separately:
#   TEACHER=1b|7b            default 7b. Selects OLMo-2-0425-1B-Instruct or
#                            OLMo-2-1124-7B-Instruct. Override TEACHER_PATH
#                            directly for any other teacher.
#
# Environment overrides:
#   RECIPE=<name>            (required) recipe basename
#   NNODES=<int>             default 4 (matches the FP8 teacher fast path)
#   STEPS_OVERRIDE=<int>     default 28800
#   LINGUA_TEACHER_FP8=0|1   default 1 for 7B teachers, 0 for 1B teacher / NTP
#   LINGUA_COMPILE_TEACHER=0|1   default 0
#   DRY_RUN=0|1              if 1, print the sbatch command and exit
#
set -euo pipefail

if [ -z "${RECIPE:-}" ]; then
    echo "ERROR: set RECIPE=<recipe_name>. See scripts/launch_midtrain.sh header." >&2
    exit 1
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Source the env script so DATA_ROOT, TEACHER_*_PATH, MIDTRAIN_ROOT, etc. are
# set before sbatch hands them to the inner script (and before downstream tools
# like OmegaConf's ${oc.env:...} resolve them).
source "${ROOT_DIR}/scripts/env.sh"

RECIPE_YAML="${ROOT_DIR}/apps/main/configs/midtrain_recipes/${RECIPE}.yaml"

if [ ! -f "${RECIPE_YAML}" ]; then
    echo "ERROR: recipe YAML not found: ${RECIPE_YAML}" >&2
    echo "Available recipes:" >&2
    ls "${ROOT_DIR}/apps/main/configs/midtrain_recipes/" | sed 's/\.yaml$//' >&2
    exit 1
fi

NNODES="${NNODES:-4}"
STEPS_OVERRIDE="${STEPS_OVERRIDE:-28800}"

# FP8 default: enabled for the 7B-Instruct teacher, disabled for the
# 1B-Instruct teacher. NTP / unspecified-teacher recipes inherit 0.
if [ -z "${LINGUA_TEACHER_FP8:-}" ]; then
    # FP8 is worth it for the 7B teacher and not for the 1B one; a recipe with
    # no teacher (NTP) inherits 0.
    if grep -q "teacher_model_path" "${RECIPE_YAML}" && [ "${TEACHER:-7b}" = "7b" ]; then
        LINGUA_TEACHER_FP8=1
    else
        LINGUA_TEACHER_FP8=0
    fi
fi
LINGUA_COMPILE_TEACHER="${LINGUA_COMPILE_TEACHER:-0}"

SLURM_LOG_DIR="${SLURM_LOG_DIR:-${HOME}/lingua-runs/slurm_logs}"
mkdir -p "${SLURM_LOG_DIR}"

JOBNAME="midtrain_${RECIPE}"

SBATCH_CMD=(
    sbatch
    --job-name="${JOBNAME}"
    --output="${SLURM_LOG_DIR}/${JOBNAME}-%j.out"
    --error="${SLURM_LOG_DIR}/${JOBNAME}-%j.err"
    --time=122:00:00
    --cpus-per-task=64
    --nodes="${NNODES}"
    --ntasks-per-node=1
    --gres=gpu:8
    --mem=456G
    --requeue
)

# Append cluster-specific flags only when the env vars are set, so collaborators
# on other clusters don't have to edit this file.
if [ -n "${SLURM_ACCOUNT:-}" ]; then
    SBATCH_CMD+=(--account="${SLURM_ACCOUNT}")
fi
if [ -n "${SLURM_QOS:-}" ]; then
    SBATCH_CMD+=(--qos="${SLURM_QOS}")
fi
if [ -n "${SLURM_PARTITION:-}" ]; then
    SBATCH_CMD+=(--partition="${SLURM_PARTITION}")
fi

SBATCH_CMD+=(
    --export=ALL,RECIPE,STEPS_OVERRIDE,LINGUA_TEACHER_FP8,LINGUA_COMPILE_TEACHER,ROOT_DIR,RECIPE_YAML,DATA_ROOT,TEACHER_1B_PATH,TEACHER_7B_PATH,TEACHER,TEACHER_PATH,STUDENT_INIT_PATH,STUDENT_HF_PATH,TOKENIZER_PATH,MIDTRAIN_ROOT,EVAL_ROOT,SLURM_LOG_DIR,LINGUA_CONDA_ENV,OLMES_CONDA_ENV,WANDB_ENTITY
    "${ROOT_DIR}/scripts/_sbatch_inner.sh"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1; would run:"
    printf '  %q' "${SBATCH_CMD[@]}"
    printf '\n'
    echo
    echo "Inner script: ${ROOT_DIR}/scripts/_sbatch_inner.sh"
    echo "Recipe YAML:  ${RECIPE_YAML}"
    exit 0
fi

"${SBATCH_CMD[@]}"
