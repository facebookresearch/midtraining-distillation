#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Deferred post-training launcher: waits for a midtrain HF checkpoint to be
# written, then runs pipeline_post_training.sh (SFT->DPO->RLVR1->RLVR2).
# Submitted with --dependency=afterok:<midtrain_job> so it only fires once
# midtrain completes. Tiny CPU slot (nested sbatch, like train_watchdog.sh).
#
# Required --export: BASE_CKPT, RUN_TAG   (optional: QOS, LEARNING_RATE)
#SBATCH --job-name=deferred_pipeline
#SBATCH --output=${SLURM_LOG_DIR}/deferred_pipeline-%j.out
#SBATCH --error=${SLURM_LOG_DIR}/deferred_pipeline-%j.err
#SBATCH --time=01:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Directory holding these release scripts (this file's own directory), so
# normalize_rope_config.py and siblings resolve without assuming a layout
# inside the open-instruct checkout.
POST_TRAINING_SCRIPTS="${POST_TRAINING_SCRIPTS:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

set -euo pipefail
: "${BASE_CKPT:?BASE_CKPT must be set}"
: "${RUN_TAG:?RUN_TAG must be set}"
# Empty => the flag is omitted downstream and the site default applies.
export QOS="${QOS:-}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
SCRIPTS_DIR=${POST_TRAINING_SCRIPTS}

# midtrain afterok fires when training completes; the final HF checkpoint /
# conversion may lag slightly. Poll up to ~45 min.
echo "Waiting for ${BASE_CKPT} ..."
for i in $(seq 1 90); do
    if [[ -f "${BASE_CKPT}/config.json" ]] && ls "${BASE_CKPT}"/*.safetensors >/dev/null 2>&1; then
        echo "Found checkpoint after $((i*30))s"; break
    fi
    sleep 30
done
if [[ ! -f "${BASE_CKPT}/config.json" ]]; then
    echo "ERROR: ${BASE_CKPT} never appeared (config.json missing) — aborting." >&2
    exit 1
fi

echo "Launching pipeline for ${RUN_TAG} from ${BASE_CKPT}"
export BASE_CKPT RUN_TAG
bash "${SCRIPTS_DIR}/pipeline_post_training.sh"
