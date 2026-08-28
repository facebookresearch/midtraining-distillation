#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Per-stage async eval helper for pipeline_post_training.sh.
# Submitted with --dependency=afterok:<stage_train_jid>; when the stage's
# checkpoint exists it resolves the HF path (handles the RLVR
# __1__<ts>_checkpoints/step_N nesting) and submits eval_single_posttrain.sh.
# Tiny CPU slot; does NOT block the training chain (eval runs async).
#
# Required --export: MODEL_SPEC (exact HF dir, OR a parent to search for
#                    *_checkpoints/step_*), OUTPUT_DIR
# Optional: EVAL_QOS
#SBATCH --job-name=eval_stage
#SBATCH --output=${SLURM_LOG_DIR}/eval_stage-%j.out
#SBATCH --error=${SLURM_LOG_DIR}/eval_stage-%j.err
#SBATCH --time=00:30:00
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail
: "${MODEL_SPEC:?MODEL_SPEC must be set}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
# Empty => omit --qos and let the site default apply.
EVAL_QOS="${EVAL_QOS:-}"
QOS_ARG=(); [[ -n "${EVAL_QOS}" ]] && QOS_ARG=(--qos="${EVAL_QOS}")
EVAL_SCRIPT=${OLMES_ROOT}/eval_single_posttrain.sh

# Resolve the HF checkpoint. SFT/DPO -> MODEL_SPEC is the exact dir.
# RLVR* -> MODEL_SPEC is the parent; the real ckpt is the highest
# *_checkpoints/step_N snapshot inside it (give it a moment in case of lag).
MODEL_PATH=""
for i in $(seq 1 40); do
    if [[ -f "${MODEL_SPEC}/config.json" ]]; then
        MODEL_PATH="${MODEL_SPEC}"; break
    fi
    cand="$(ls -d ${MODEL_SPEC}/*_checkpoints/step_* 2>/dev/null | sort -V | tail -1 || true)"
    if [[ -n "${cand}" && -f "${cand}/config.json" ]]; then MODEL_PATH="${cand}"; break; fi
    sleep 15
done
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "ERROR: could not resolve a checkpoint from MODEL_SPEC=${MODEL_SPEC}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
echo "Submitting eval: ${MODEL_PATH} -> ${OUTPUT_DIR} (qos ${EVAL_QOS:-<site default>})"
# Do not pass `sbatch --export=`: some cgroup-v2 Slurm sites then hold the
# job at Priority=0 forever. Put the vars in sbatch's own environment instead.
env HOME="${HOME}" USER="${USER}" \
    MODEL_PATH="${MODEL_PATH}" OUTPUT_DIR="${OUTPUT_DIR}" \
    sbatch "${QOS_ARG[@]}" \
    "${EVAL_SCRIPT}"
