#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# Environment configuration. Source before training, eval, or analysis:
#   source scripts/env.sh
# Override any variable by exporting it before sourcing, or edit the defaults
# below. All generated artifacts root under ${CACHE_DIR}, so one `rm -rf` cleans
# everything up.
#
# ============================================================================
# YOU MUST SET THESE FOUR. There are no sensible defaults — they point at
# multi-hundred-GB artifacts that you have to download and prepare first.
#
#   DATA_ROOT          Dolmino splits dir  (~2 TB; do NOT put it under $HOME).
#                      Produced by: python setup/download_prepare_dolmino.py
#   TEACHER_1B_PATH    HF dir for OLMo-2-0425-1B-Instruct        (KD recipes only)
#   TEACHER_7B_PATH    HF dir for OLMo-2-1124-7B-Instruct        (KD recipes only)
#   STUDENT_INIT_PATH  Lingua DCP dir for OLMo-2-0425-1B @ stage1-step1907359-tokens4001B
#                      (mid-training only; the from-scratch recipes ignore it)
#                      Produced by: bash scripts/fetch_models.sh
#
# `bash scripts/fetch_models.sh` populates the three checkpoint paths for you
# once TEACHER_*/STUDENT_* point at writable locations. Run
# ============================================================================

export CACHE_DIR="${CACHE_DIR:-${HOME}/midtraining-distillation-sandbox}"

export DATA_ROOT="${DATA_ROOT:-${CACHE_DIR}/data/dolmino_splits}"
export TEACHER_1B_PATH="${TEACHER_1B_PATH:-${CACHE_DIR}/teachers/OLMo-2-0425-1B-Instruct}"
export TEACHER_7B_PATH="${TEACHER_7B_PATH:-${CACHE_DIR}/teachers/OLMo-2-1124-7B-Instruct}"
export STUDENT_INIT_PATH="${STUDENT_INIT_PATH:-${CACHE_DIR}/students/OLMo-2-0425-1B-stage1-4001B}"  # Lingua DCP root
export STUDENT_HF_PATH="${STUDENT_HF_PATH:-${STUDENT_INIT_PATH}/hf}"                                # HF mirror
export TOKENIZER_PATH="${TOKENIZER_PATH:-${STUDENT_HF_PATH}}"

# Which teacher the distillation recipes use. The student is always the 1B
# model; only the teacher changes. Set TEACHER=1b or TEACHER=7b (the launch
# scripts also accept it as an env var), or point TEACHER_PATH somewhere else
# entirely to distil from a teacher not listed here.
export TEACHER="${TEACHER:-7b}"
case "${TEACHER}" in
    1b) export TEACHER_PATH="${TEACHER_PATH:-${TEACHER_1B_PATH}}" ;;
    7b) export TEACHER_PATH="${TEACHER_PATH:-${TEACHER_7B_PATH}}" ;;
    *)  : "${TEACHER_PATH:?TEACHER must be 1b or 7b, or set TEACHER_PATH explicitly}" ;;
esac

# Output roots.
export MIDTRAIN_ROOT="${MIDTRAIN_ROOT:-${CACHE_DIR}/runs/midtrain}"
export PRETRAIN_ROOT="${PRETRAIN_ROOT:-${CACHE_DIR}/runs/pretrain}"
export EVAL_ROOT="${EVAL_ROOT:-${CACHE_DIR}/runs/evals}"
export SLURM_LOG_DIR="${SLURM_LOG_DIR:-${CACHE_DIR}/runs/slurm_logs}"

# Conda. The OLMES env is separate because it pins different vLLM /
# transformers versions than training does — see evaluation/README.md.
export LINGUA_CONDA_ENV="${LINGUA_CONDA_ENV:-midtraining-distillation}"
export OLMES_CONDA_ENV="${OLMES_CONDA_ENV:-}"

# SLURM. Leave a value empty to omit the flag and let the site default apply.
# These are cluster-specific: whatever your site uses, not ours.
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
export SLURM_QOS="${SLURM_QOS:-}"
export SLURM_PARTITION="${SLURM_PARTITION:-}"

# Wandb. Empty → your personal default entity.
export WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "${CACHE_DIR}" "${MIDTRAIN_ROOT}" "${PRETRAIN_ROOT}" "${EVAL_ROOT}" "${SLURM_LOG_DIR}"

echo "[env] CACHE_DIR=${CACHE_DIR}  (delete with: rm -rf \$CACHE_DIR)"
echo "[env] DATA_ROOT=${DATA_ROOT}"
echo "[env] SLURM_ACCOUNT=${SLURM_ACCOUNT:-<site default>}  SLURM_QOS=${SLURM_QOS:-<site default>}"

if [ ! -d "${DATA_ROOT}" ]; then
    echo "[env] WARNING: DATA_ROOT does not exist yet. Prepare it with:" >&2
    echo "[env]          python setup/download_prepare_dolmino.py --data-dir \${DATA_ROOT}" >&2
fi
