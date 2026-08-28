#!/bin/bash
# scripts/launch_pretrain.sh — launch a from-scratch pretraining run via sbatch.
#
# This is the controlled from-scratch replication reported in the paper's
# pre-training table: same loss / teacher / data / token budget as the
# mid-training recipes, but starting from random init instead of the
# pretrained OLMo-2 1B student. It is what flips the KD-vs-NTP recall sign.
#
# Same wrapper as `launch_midtrain.sh` around `_sbatch_inner.sh`; the only
# difference is that the YAML lives in `apps/main/configs/pretrain_recipes/`.
#
# Usage:
#   RECIPE=pt_ntp bash scripts/launch_pretrain.sh                # submit
#   RECIPE=pt_fkd DRY_RUN=1 bash scripts/launch_pretrain.sh   # print sbatch
#
# Available RECIPE values (basenames under apps/main/configs/pretrain_recipes/):
#   pt_ntp, pt_fkd, pt_rkd
#
# Environment overrides:
#   RECIPE=<name>           default `pt_ntp`.
#   NNODES=<int>            default 4 (the teacher recipes need it; pt_ntp runs fine on 2).
#   STEPS_OVERRIDE=<int>    default 48000 (~100B Dolmino tokens at the configured batch/seq).
#   DRY_RUN=0|1             if 1, print the sbatch command and exit.

set -euo pipefail

RECIPE="${RECIPE:-pt_ntp}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/env.sh"

CONFIG_YAML="${ROOT_DIR}/apps/main/configs/pretrain_recipes/${RECIPE}.yaml"

if [ ! -f "${CONFIG_YAML}" ]; then
    echo "ERROR: pretrain recipe not found: ${CONFIG_YAML}" >&2
    echo "Available pretrain recipes:" >&2
    ls "${ROOT_DIR}/apps/main/configs/pretrain_recipes/" 2>/dev/null | sed 's/\.yaml$//' >&2
    exit 1
fi

NNODES="${NNODES:-4}"
STEPS_OVERRIDE="${STEPS_OVERRIDE:-48000}"

# FP8 default mirrors launch_midtrain.sh: on for the 7B-teacher recipes,
# off for the 1B teacher and for teacher-free pt_ntp.
if [ -z "${LINGUA_TEACHER_FP8:-}" ]; then
    if grep -q "teacher_model_path" "${CONFIG_YAML}" && [ "${TEACHER:-7b}" = "7b" ]; then
        LINGUA_TEACHER_FP8=1
    else
        LINGUA_TEACHER_FP8=0
    fi
fi
LINGUA_COMPILE_TEACHER="${LINGUA_COMPILE_TEACHER:-0}"

SLURM_LOG_DIR="${SLURM_LOG_DIR:-${HOME}/lingua-runs/slurm_logs}"
mkdir -p "${SLURM_LOG_DIR}"

JOBNAME="pretrain_${RECIPE}"

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

if [ -n "${SLURM_ACCOUNT:-}" ]; then
    SBATCH_CMD+=(--account="${SLURM_ACCOUNT}")
fi
if [ -n "${SLURM_QOS:-}" ]; then
    SBATCH_CMD+=(--qos="${SLURM_QOS}")
fi
if [ -n "${SLURM_PARTITION:-}" ]; then
    SBATCH_CMD+=(--partition="${SLURM_PARTITION}")
fi

# The inner script keys off RECIPE / RECIPE_YAML.
RECIPE_YAML="${CONFIG_YAML}"

SBATCH_CMD+=(
    --export=ALL,RECIPE,STEPS_OVERRIDE,LINGUA_TEACHER_FP8,LINGUA_COMPILE_TEACHER,ROOT_DIR,RECIPE_YAML,DATA_ROOT,TEACHER_1B_PATH,TEACHER_7B_PATH,TEACHER,TEACHER_PATH,STUDENT_INIT_PATH,STUDENT_HF_PATH,TOKENIZER_PATH,MIDTRAIN_ROOT,PRETRAIN_ROOT,EVAL_ROOT,SLURM_LOG_DIR,LINGUA_CONDA_ENV,OLMES_CONDA_ENV,WANDB_ENTITY
    "${ROOT_DIR}/scripts/_sbatch_inner.sh"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1; would run:"
    printf '  %q' "${SBATCH_CMD[@]}"
    printf '\n'
    echo
    echo "Inner script: ${ROOT_DIR}/scripts/_sbatch_inner.sh"
    echo "Recipe YAML:  ${CONFIG_YAML}"
    exit 0
fi

"${SBATCH_CMD[@]}"
