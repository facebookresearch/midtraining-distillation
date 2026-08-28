#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=train_watchdog
#SBATCH --output=${SLURM_LOG_DIR}/train_watchdog-%j.out
#SBATCH --error=${SLURM_LOG_DIR}/train_watchdog-%j.err
#SBATCH --time=10:00
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#
# Auto-resubmit watchdog for grpo_fast.py training jobs.
#
# Fires via --dependency=afternotok:<train_jobid> when the watched job fails
# (NODE_FAIL, OOM, transient network errors, etc.). Resubmits the same
# training script with the same env, and chains a fresh watchdog on the new
# job so the chain can survive multiple failures (up to MAX_RETRIES).
#
# The training script must support resume via --checkpoint_state_dir, and
# OUTPUT_DIR must be stable across retries (so the resume state is found).
#
# Required env (from --export):
#   TRAIN_SCRIPT      — path to the training sbatch script to re-submit
#   TRAIN_QOS         — sbatch --qos for the resubmit
#   TRAIN_EXPORT_VARS — comma-separated list of "KEY=VAL" to forward via --export
#                        e.g. "EXP_NAME=foo,OUTPUT_DIR=/path,DPO_CKPT=/path"
#   FAILED_JID        — the jobid that just failed (for logging)
#   ATTEMPT           — attempt counter (1 on first auto-restart)
#   MAX_RETRIES       — give up after this many tries
#
# Optional env:
#   POST_RETRY_JID    — jobid of a downstream job that was chained on the
#                        failed train job; we'll re-chain it on the resubmit
#                        (e.g., the rlvr2 launcher chained on rlvr1).
#                        If unset, no re-chain.

set -euo pipefail

ATTEMPT="${ATTEMPT:-1}"
MAX_RETRIES="${MAX_RETRIES:-3}"

echo "[watchdog] failed job: ${FAILED_JID:-?}"
echo "[watchdog] attempt: ${ATTEMPT}/${MAX_RETRIES}"
echo "[watchdog] script: ${TRAIN_SCRIPT}"
echo "[watchdog] qos: ${TRAIN_QOS}"

if [ "${ATTEMPT}" -gt "${MAX_RETRIES}" ]; then
    echo "[watchdog] MAX_RETRIES (${MAX_RETRIES}) reached; giving up."
    exit 0
fi

# Resubmit training. Forward the original export vars + bump ATTEMPT.
# Do not pass `sbatch --export=`: some cgroup-v2 Slurm sites then hold the
# job at Priority=0 forever. Put the vars in sbatch's own environment instead.
NEXT_ATTEMPT=$((ATTEMPT + 1))
IFS=',' read -ra TRAIN_KV <<< "${TRAIN_EXPORT_VARS}"

echo "[watchdog] resubmitting ${TRAIN_SCRIPT} with qos=${TRAIN_QOS:-<site default>}"
QOS_ARG=(); [[ -n "${TRAIN_QOS:-}" ]] && QOS_ARG=(--qos="${TRAIN_QOS}")
NEW_TRAIN_JID=$(env HOME="${HOME}" USER="${USER}" "${TRAIN_KV[@]}" \
    sbatch --parsable "${QOS_ARG[@]}" \
    "${TRAIN_SCRIPT}")
echo "[watchdog] new train jobid: ${NEW_TRAIN_JID}"

# Chain a fresh watchdog on the new training job
WATCHDOG_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
NEW_WATCHDOG_JID=$(env HOME="${HOME}" USER="${USER}" \
    TRAIN_SCRIPT="${TRAIN_SCRIPT}" TRAIN_QOS="${TRAIN_QOS}" \
    TRAIN_EXPORT_VARS="${TRAIN_EXPORT_VARS}" FAILED_JID="${NEW_TRAIN_JID}" \
    ATTEMPT="${NEXT_ATTEMPT}" MAX_RETRIES="${MAX_RETRIES}" \
    ${POST_RETRY_JID:+POST_RETRY_JID="${POST_RETRY_JID}"} \
    sbatch --parsable --dependency=afternotok:${NEW_TRAIN_JID} \
    "${WATCHDOG_SCRIPT}")
echo "[watchdog] new watchdog jobid: ${NEW_WATCHDOG_JID} (will fire if ${NEW_TRAIN_JID} fails)"

# If there's a downstream job that needs re-chaining (e.g. rlvr2 launcher
# was originally chained on the original train job), re-issue the dependency
# to point at the new train job. Slurm supports scontrol update.
if [ -n "${POST_RETRY_JID:-}" ]; then
    echo "[watchdog] re-chaining ${POST_RETRY_JID} to depend on ${NEW_TRAIN_JID}"
    scontrol update JobId=${POST_RETRY_JID} Dependency=afterok:${NEW_TRAIN_JID} || \
        echo "[watchdog] WARNING: scontrol update failed for ${POST_RETRY_JID} (may have run/cancelled already)"
fi

echo "[watchdog] done."
