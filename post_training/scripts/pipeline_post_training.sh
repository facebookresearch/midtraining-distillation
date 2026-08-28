#!/usr/bin/env bash
# Orchestrator: submits SFT -> DPO -> RLVR1 -> RLVR2 as four slurm jobs
# chained via --dependency=afterok. Each stage's HF checkpoint is wired
# into the next stage's inputs.
#
# Auto-retry: each stage also gets a sibling watchdog job
# (--dependency=afternotok) that resubmits the same training script with
# the same OUTPUT_DIR. The training scripts pass --checkpoint_state_dir
# to grpo_fast.py (and equivalent for SFT/DPO), so resubmits resume from
# the last saved step rather than restarting from scratch. Up to
# MAX_RETRIES retries per stage.
#
# Usage:
#   BASE_CKPT=/path/to/midtrain/checkpoints/0000028800/hf \
#   RUN_TAG=my_kd_variant \
#   bash post_training/scripts/pipeline_post_training.sh
#
# Output layout:
#   ${OUT_ROOT}/
#     ${RUN_TAG}_sft/${RUN_TAG}_sft/        # leaf ckpt
#     ${RUN_TAG}_dpo/${RUN_TAG}_dpo/        # leaf ckpt
#     ${RUN_TAG}_rlvr1/                      # parent (open-instruct nests __1__<ts>_checkpoints/step_N)
#     ${RUN_TAG}_rlvr2/                      # parent (same nesting)
#
# Defaults:
#   OUT_ROOT     = ${POST_TRAINING_ROOT}/${RUN_TAG}
#   RUN_TAG      = (required) — short label used to name every stage and dir
#   BASE_CKPT    = (required) — path to the midtrain HF checkpoint
#   QOS          = (optional) — sbatch --qos override for every stage
#   MAX_RETRIES  = 3   — watchdog gives up after this many resubmits per stage
#
# Skipping stages:
#   SKIP_DPO=1   → SFT only (DPO, RLVR1, RLVR2 not submitted)
#   SKIP_RLVR=1  → SFT + DPO only
#   SKIP_RLVR2=1 → SFT + DPO + RLVR1 only
#
# Starting mid-chain from an existing SFT (skip the head, not the tail):
#   SKIP_SFT=1   → do NOT submit SFT; start the chain at DPO from an
#                  already-saved SFT checkpoint. REQUIRES:
#                    SFT_CKPT=/path/to/sft/leaf   (HF dir containing config.json)
#                  BASE_CKPT is unused in this mode (no need to set it). DPO
#                  starts immediately unless EXTRA_DEPENDENCY gates it.
#   Example (chain DPO→RLVR1→RLVR2 off a saved SFT, on a chosen qos):
#     SKIP_SFT=1 SFT_CKPT=/path/to/sft_leaf RUN_TAG=my_variant \
#     OUT_ROOT=/path/to/post_training QOS=<your-qos> \
#     bash post_training/scripts/pipeline_post_training.sh
#
# Disable auto-retry:
#   NO_WATCHDOG=1 → no afternotok watchdog per stage
#
# Eval-by-default (async):
#   Each stage submits an eval (afterok:<stage>) via eval_stage.sh that runs
#   the Tulu-3 dev suite into ${RES_ROOT}/<stage_exp>. It does NOT block the
#   training chain (next stage depends on the prior STAGE, not its eval).
#   Disable with NO_EVAL=1. Override the eval qos with EVAL_QOS.
#   RES_ROOT default = ${OLMES_ROOT}/results/posttrain
#
# Dry run (prints sbatch commands but does not submit):
#   DRY_RUN=1 bash post_training/scripts/pipeline_post_training.sh
#
# Compute footprint when run end-to-end:
#   SFT  : 4 nodes × 8 GPUs × ~2.5h ≈ 80 GPU-h  (4-node parallel; 2.1× faster than 1-node)
#   DPO  : 1 node × 8 GPUs × ~2h    ≈ 16 GPU-h
#   RLVR1: 4 nodes × 8 GPUs × ~5h   ≈ 160 GPU-h
#   RLVR2: 4 nodes × 8 GPUs × ~5h   ≈ 160 GPU-h
#   ────────────────────────────────────────
#   Total: ~416 GPU-h per pipeline invocation. Wall-clock ~14.5h end-to-end
#   (was ~17h on 1-node SFT).

set -euo pipefail

# ---------- required inputs ----------
: "${RUN_TAG:?RUN_TAG must be set (short label for this pipeline, e.g. rkl_topkgap_125)}"
# BASE_CKPT feeds the SFT stage. When SKIP_SFT=1 the chain starts at DPO from
# an already-saved SFT_CKPT, so BASE_CKPT is not needed.
if [[ -z "${SKIP_SFT:-}" ]]; then
    : "${BASE_CKPT:?BASE_CKPT must be set (path to midtrain HF checkpoint, ending in /hf)}"
else
    : "${SFT_CKPT:?SKIP_SFT=1 requires SFT_CKPT (path to an existing SFT HF leaf dir with config.json)}"
    BASE_CKPT="${BASE_CKPT:-<skipped: SKIP_SFT=1>}"
fi

OUT_ROOT="${OUT_ROOT:-${POST_TRAINING_ROOT}/${RUN_TAG}}"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_RETRIES="${MAX_RETRIES:-3}"

if [[ -z "${SKIP_SFT:-}" ]]; then
    if [[ ! -d "${BASE_CKPT}" ]]; then
        echo "ERROR: BASE_CKPT does not exist: ${BASE_CKPT}" >&2
        echo "       (expected an HF checkpoint dir with config.json + model.safetensors)" >&2
        exit 1
    fi
    if [[ ! -f "${BASE_CKPT}/config.json" ]]; then
        echo "ERROR: ${BASE_CKPT}/config.json not found — is this really an HF checkpoint?" >&2
        exit 1
    fi
fi

mkdir -p "${OUT_ROOT}"

# QoS override — when set, applied to every stage via sbatch --qos=...
QOS_ARG=()
if [[ -n "${QOS:-}" ]]; then
    QOS_ARG=(--qos="${QOS}")
fi
# qos string for watchdog (watchdog itself runs on a tiny CPU slot; reuse same qos)
WATCHDOG_QOS="${QOS:-CHANGE_ME}"

SFT_EXP="${RUN_TAG}_sft"
DPO_EXP="${RUN_TAG}_dpo"
RLVR_EXP="${RUN_TAG}_rlvr1"
RLVR2_EXP="${RUN_TAG}_rlvr2"

# open_instruct's finetune.py writes to ${OUTPUT_DIR}/${EXP_NAME}/ (a nested
# subdir, exp_name repeated). RLVR1/RLVR2 use OUTPUT_DIR directly with the
# grpo_fast.py __1__<ts>_checkpoints layout inside it.
# Honor an externally provided SFT_CKPT (required under SKIP_SFT=1); otherwise
# default to the leaf that stage 1 (SFT) writes.
SFT_CKPT="${SFT_CKPT:-${OUT_ROOT}/${SFT_EXP}/${SFT_EXP}}"
DPO_CKPT="${OUT_ROOT}/${DPO_EXP}/${DPO_EXP}"
RLVR_PARENT="${OUT_ROOT}/${RLVR_EXP}"
RLVR2_PARENT="${OUT_ROOT}/${RLVR2_EXP}"

# Pre-check EXP_NAME wandb tag limits (wandb 0.23 rejects tags > 64 chars).
for name in "${SFT_EXP}" "${DPO_EXP}" "${RLVR_EXP}" "${RLVR2_EXP}"; do
    if (( ${#name} > 64 )); then
        echo "ERROR: EXP_NAME '${name}' is ${#name} chars (>64); shorten RUN_TAG." >&2
        echo "       wandb 0.23 rejects tags >64 chars at wandb.init() time." >&2
        exit 1
    fi
done

echo "==========================================================="
echo "Pipeline: ${RUN_TAG}"
echo "  BASE_CKPT   : ${BASE_CKPT}"
echo "  OUT_ROOT    : ${OUT_ROOT}"
echo "  Stages      : ${SKIP_SFT:+(start at DPO from existing SFT)}${SKIP_SFT:-SFT} ${SKIP_DPO:+(skip DPO)} ${SKIP_RLVR:+(skip RLVR1)} ${SKIP_RLVR2:+(skip RLVR2)}"
echo "  Auto-retry  : ${NO_WATCHDOG:+disabled}${NO_WATCHDOG:-enabled (MAX_RETRIES=${MAX_RETRIES})}"
echo "  QoS         : ${QOS:-<per-script default>}"
echo "==========================================================="

run_sbatch() {
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "DRY_RUN: sbatch $*" >&2
        echo "DRY_$(date +%s%N | tail -c 7)"
        return
    fi
    local out
    out="$(sbatch "$@")" || { echo "ERROR: sbatch failed: $*" >&2; exit 1; }
    echo "${out}" | awk '/Submitted batch job/ {print $4}'
}


run_sbatch_env() {
    local kv_csv="$1"; shift
    local -a kv=()
    [[ -n "${kv_csv}" ]] && IFS=',' read -ra kv <<< "${kv_csv}"
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "DRY_RUN: env ${kv[*]} sbatch $*" >&2
        echo "DRY_$(date +%s%N | tail -c 7)"
        return
    fi
    local out
    out="$(env HOME="${HOME}" USER="${USER}" "${kv[@]}" sbatch "$@")" \
        || { echo "ERROR: sbatch failed: $*" >&2; exit 1; }
    echo "${out}" | awk '/Submitted batch job/ {print $4}'
}

# Submit a watchdog for a just-submitted training job. The watchdog fires
# on FAILURE (afternotok) and resubmits the same training script with the
# same --export payload — relying on checkpoint_state_dir resume.
submit_watchdog() {
    local train_jid="$1"
    local train_script="$2"
    local train_qos="$3"
    local train_export_vars="$4"   # comma-separated KEY=VAL pairs (no leading NONE/HOME/USER)

    if [[ -n "${NO_WATCHDOG:-}" || -n "${DRY_RUN:-}" ]]; then
        return
    fi

    # The watchdog job itself runs on ${train_qos} (=WATCHDOG_QOS, honors QOS=
    # override); without this it would fall back to train_watchdog.sh's header.
    env HOME="${HOME}" USER="${USER}" \
        TRAIN_SCRIPT="${train_script}" TRAIN_QOS="${train_qos}" \
        TRAIN_EXPORT_VARS="${train_export_vars}" FAILED_JID="${train_jid}" \
        ATTEMPT=1 MAX_RETRIES="${MAX_RETRIES}" \
        sbatch --parsable \
        --qos="${train_qos}" \
        --dependency=afternotok:"${train_jid}" \
        "${SCRIPTS_DIR}/train_watchdog.sh" \
        | awk '{print "  watchdog jid:", $1}' >&2 || true
}

# Eval-by-default: after a stage's training job is submitted, submit an async
# eval that fires on that stage's success (afterok) and does NOT block the
# training chain (eval runs in parallel with later stages). Disable: NO_EVAL=1.
RES_ROOT="${RES_ROOT:-${OLMES_ROOT}/results/posttrain}"
EVAL_STAGE_SCRIPT="${SCRIPTS_DIR}/eval_stage.sh"
submit_stage_eval() {
    local train_jid="$1"
    local model_spec="$2"   # exact HF dir (SFT/DPO) or parent to glob (RLVR*)
    local out_dir="$3"
    if [[ -n "${NO_EVAL:-}" || -n "${DRY_RUN:-}" ]]; then
        return
    fi
    # The eval-stage launcher (tiny CPU slot) honors QOS= too; the GPU eval it
    # spawns uses EVAL_QOS.
    env HOME="${HOME}" USER="${USER}" \
        MODEL_SPEC="${model_spec}" OUTPUT_DIR="${out_dir}" \
        EVAL_QOS="${EVAL_QOS:-CHANGE_ME}" \
        sbatch --parsable \
        ${QOS:+--qos="${QOS}"} \
        --dependency=afterok:"${train_jid}" \
        --job-name="eval_$(basename "${out_dir}")" \
        "${EVAL_STAGE_SCRIPT}" \
        | awk '{print "  eval jid:", $1}' >&2 || true
}

# -------------------- Stage 1: SFT --------------------
# DPO's dependency: normally afterok on the SFT job submitted here. Under
# SKIP_SFT=1 there is no SFT job — DPO starts immediately (or gates on an
# external EXTRA_DEPENDENCY, if the caller provided one).
DPO_DEP_ARG=()
if [[ -z "${SKIP_SFT:-}" ]]; then
    echo
    echo "Submitting SFT..."
    SFT_SCRIPT="${SCRIPTS_DIR}/train_olmo2_1b_sft_4node.sh"
    SFT_EXPORT_VARS="BASE_CKPT=${BASE_CKPT},EXP_NAME=${SFT_EXP},OUT_ROOT=${OUT_ROOT},LEARNING_RATE=5e-6"

    SFT_DEP_ARG=()
    if [[ -n "${EXTRA_DEPENDENCY:-}" ]]; then
        SFT_DEP_ARG=(--dependency="${EXTRA_DEPENDENCY}")
        echo "  (SFT will wait on: ${EXTRA_DEPENDENCY})"
    fi
    SFT_JOBID=$(run_sbatch_env "${SFT_EXPORT_VARS}" \
        --job-name="${SFT_EXP}" \
        --output="${SLURM_LOG_DIR}/${SFT_EXP}-%j.out" \
        --error="${SLURM_LOG_DIR}/${SFT_EXP}-%j.err" \
        "${QOS_ARG[@]}" \
        "${SFT_DEP_ARG[@]}" \
            "${SFT_SCRIPT}")
    echo "  SFT job ID: ${SFT_JOBID}"
    submit_watchdog "${SFT_JOBID}" "${SFT_SCRIPT}" "${WATCHDOG_QOS}" "${SFT_EXPORT_VARS}"
    submit_stage_eval "${SFT_JOBID}" "${SFT_CKPT}" "${RES_ROOT}/${SFT_EXP}"
    DPO_DEP_ARG=(--dependency="afterok:${SFT_JOBID}")
else
    echo
    echo "SKIP_SFT=1 — starting chain at DPO from existing SFT checkpoint:"
    echo "  SFT_CKPT=${SFT_CKPT}"
    if [[ ! -d "${SFT_CKPT}" || ! -f "${SFT_CKPT}/config.json" ]]; then
        echo "ERROR: SKIP_SFT set but SFT_CKPT is not a valid HF checkpoint dir" >&2
        echo "       (need ${SFT_CKPT}/config.json). Point SFT_CKPT at the SFT leaf dir." >&2
        exit 1
    fi
    SFT_JOBID="(existing)"
    if [[ -n "${EXTRA_DEPENDENCY:-}" ]]; then
        DPO_DEP_ARG=(--dependency="${EXTRA_DEPENDENCY}")
        echo "  (DPO will wait on: ${EXTRA_DEPENDENCY})"
    fi
fi

if [[ -n "${SKIP_DPO:-}" ]]; then
    echo
    echo "SKIP_DPO=1 set — not submitting DPO/RLVR1/RLVR2."
    exit 0
fi

# -------------------- Stage 2: DPO (afterok SFT) --------------------
echo
echo "Submitting DPO (dep: ${DPO_DEP_ARG[*]:-none})..."
DPO_SCRIPT="${SCRIPTS_DIR}/train_olmo2_1b_dpo.sh"
DPO_EXPORT_VARS="SFT_CKPT=${SFT_CKPT},EXP_NAME=${DPO_EXP},OUT_ROOT=${OUT_ROOT}"
DPO_JOBID=$(run_sbatch_env "${DPO_EXPORT_VARS}" \
    "${DPO_DEP_ARG[@]}" \
    --job-name="${DPO_EXP}" \
    --output="${SLURM_LOG_DIR}/${DPO_EXP}-%j.out" \
    --error="${SLURM_LOG_DIR}/${DPO_EXP}-%j.err" \
    "${QOS_ARG[@]}" \
    "${DPO_SCRIPT}")
echo "  DPO job ID: ${DPO_JOBID}"
submit_watchdog "${DPO_JOBID}" "${DPO_SCRIPT}" "${WATCHDOG_QOS}" "${DPO_EXPORT_VARS}"
submit_stage_eval "${DPO_JOBID}" "${DPO_CKPT}" "${RES_ROOT}/${DPO_EXP}"

if [[ -n "${SKIP_RLVR:-}" ]]; then
    echo
    echo "SKIP_RLVR=1 set — not submitting RLVR1/RLVR2."
    exit 0
fi

# -------------------- Stage 3: RLVR1 (afterok DPO) --------------------
# RLVR LR is hardcoded to 5e-7 (Tulu-3 1B standard) and deliberately overrides
# any LEARNING_RATE in the calling env: --export=ALL would otherwise leak the
# SFT LR into the RL stages and overcook them. Do not make it overridable.
echo
echo "Submitting RLVR1 4-node scaled (afterok:${DPO_JOBID})..."
RLVR_SCRIPT="${SCRIPTS_DIR}/train_olmo2_1b_rlvr1_4node_scaled.sh"
RLVR_EXPORT_VARS="DPO_CKPT=${DPO_CKPT},EXP_NAME=${RLVR_EXP},OUT_ROOT=${OUT_ROOT},OUTPUT_DIR=${RLVR_PARENT},LEARNING_RATE=5e-7"
RLVR_JOBID=$(run_sbatch_env "${RLVR_EXPORT_VARS}" \
    --dependency="afterok:${DPO_JOBID}" \
    --job-name="${RLVR_EXP}" \
    --output="${SLURM_LOG_DIR}/${RLVR_EXP}-%j.out" \
    --error="${SLURM_LOG_DIR}/${RLVR_EXP}-%j.err" \
    "${QOS_ARG[@]}" \
    "${RLVR_SCRIPT}")
echo "  RLVR1 job ID: ${RLVR_JOBID}"
submit_watchdog "${RLVR_JOBID}" "${RLVR_SCRIPT}" "${WATCHDOG_QOS}" "${RLVR_EXPORT_VARS}"
submit_stage_eval "${RLVR_JOBID}" "${RLVR_PARENT}" "${RES_ROOT}/${RLVR_EXP}-2604steps"

if [[ -n "${SKIP_RLVR2:-}" ]]; then
    echo
    echo "SKIP_RLVR2=1 set — not submitting RLVR2."
    exit 0
fi

# -------------------- Stage 4: RLVR2 (afterok RLVR1) --------------------
# RLVR1 saves to ${RLVR_PARENT}/${RLVR_EXP}__1__<ts>_checkpoints/step_2600.
# We pass RLVR1_PARENT_DIR + RLVR1_EXP_NAME (known now) so the RLVR2
# script can lazy-resolve the snapshot path at runtime — no per-run
# launcher wrapper needed.
# LEARNING_RATE=5e-7 is HARDCODED (see RLVR1 comment above). DO NOT change.
echo
echo "Submitting RLVR2 4-node scaled (afterok:${RLVR_JOBID})..."
RLVR2_SCRIPT="${SCRIPTS_DIR}/train_olmo2_1b_rlvr2_4node_scaled.sh"
RLVR2_EXPORT_VARS="RLVR1_PARENT_DIR=${RLVR_PARENT},RLVR1_EXP_NAME=${RLVR_EXP},EXP_NAME=${RLVR2_EXP},OUT_ROOT=${OUT_ROOT},OUTPUT_DIR=${RLVR2_PARENT},LEARNING_RATE=5e-7"
RLVR2_JOBID=$(run_sbatch_env "${RLVR2_EXPORT_VARS}" \
    --dependency="afterok:${RLVR_JOBID}" \
    --job-name="${RLVR2_EXP}" \
    --output="${SLURM_LOG_DIR}/${RLVR2_EXP}-%j.out" \
    --error="${SLURM_LOG_DIR}/${RLVR2_EXP}-%j.err" \
    "${QOS_ARG[@]}" \
    "${RLVR2_SCRIPT}")
echo "  RLVR2 job ID: ${RLVR2_JOBID}"
submit_watchdog "${RLVR2_JOBID}" "${RLVR2_SCRIPT}" "${WATCHDOG_QOS}" "${RLVR2_EXPORT_VARS}"
submit_stage_eval "${RLVR2_JOBID}" "${RLVR2_PARENT}" "${RES_ROOT}/${RLVR2_EXP}-2604steps"

echo
echo "==========================================================="
echo "Pipeline submitted:"
echo "  SFT   ${SFT_JOBID}   -> ${SFT_CKPT}"
echo "  DPO   ${DPO_JOBID}   -> ${DPO_CKPT}                    (dep:${DPO_DEP_ARG[*]:-none})"
echo "  RLVR1 ${RLVR_JOBID}  -> ${RLVR_PARENT}/__1__<ts>/step_2600   (afterok:${DPO_JOBID})"
echo "  RLVR2 ${RLVR2_JOBID} -> ${RLVR2_PARENT}/__1__<ts>/step_2600  (afterok:${RLVR_JOBID})"
echo
echo "Inspect with: squeue -u \$USER | grep -E '${SFT_JOBID}|${DPO_JOBID}|${RLVR_JOBID}|${RLVR2_JOBID}'"
echo "Cancel chain with: scancel ${SFT_JOBID} ${DPO_JOBID} ${RLVR_JOBID} ${RLVR2_JOBID}"
echo "==========================================================="
