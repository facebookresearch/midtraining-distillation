#!/usr/bin/env bash
# OLMo-2 1B RLVR stage 1 (GSM/MATH/IF) — 4-NODE SCALED REPRODUCTION
#
# Preserves official local minibatch geometry but uses all 4 nodes for
# faster wall-clock by doubling the rollout batch. To match the published
# GRPO update count exactly we also double total_episodes so 2× larger
# rollouts × 2× episodes ⇒ the same 2604 updates as the 2-node recipe.
#
# Published 2-node recipe (docs/olmo2.md):
#   num_learners_per_node = "4 8"           → 12 learners
#   num_unique_prompts_rollout × samples    = 48 × 16 = 768 sequences/update
#   samples_per_learner_per_update          = 768 / 12 = 64
#   num_mini_batches = 2 → local minibatch  = 32
#   total_episodes = 2,000,000              → 2604 GRPO update steps
#   vllm_num_engines × TP = 1 × 4           = 4 GPUs
#   ⇒ 12 + 4 = 16 GPUs over 2 nodes, ~43h
#
# 4-node scaled (this script):
#   num_learners_per_node = "6 6 6 6"       → 24 learners
#   num_unique_prompts_rollout × samples    = 96 × 16 = 1536 sequences/update
#   samples_per_learner_per_update          = 1536 / 24 = 64
#   num_mini_batches = 2 → local minibatch  = 32   ✓ matches published
#   total_episodes = 4,000,000              → 2604 GRPO updates ✓ matches published
#   vllm_num_engines × TP = 4 × 2           = 8 GPUs (1 engine/node, TP=2)
#   ⇒ 24 + 8 = 32 GPUs across 4 nodes, ~24h (~1.8× faster than 43h published)
#
# Why TP=2 (not TP=4): with 6 learners/node, only 2 GPUs/node are free.
# A TP=4 vLLM engine requires 4 GPUs packed on a single node (STRICT_PACK),
# which is infeasible under this 6/6/6/6 learner layout — total GPU count
# matches (8 free, 8 needed) but per-node bin packing fails. Job 7936357
# (TP=4 × 2 engines) hung silently in vLLM placement-group wait for ~65 min
# before being killed. TP=2 × 4 engines fits exactly: 6 + 2 = 8 GPUs/node.
#
# What's preserved vs published:
#   ✓ local minibatch = 32 (the LR-calibration unit)
#   ✓ samples per learner = 64
#   ✓ num_mini_batches = 2
#   ✓ total GRPO updates = 2604
#
# What's different vs published:
#   ✗ 24 learners instead of 12 (doubled)
#   ✗ rollout batch 1536 instead of 768 (doubled)
#   ✗ total_episodes = 4M instead of 2M (doubled to preserve 2604 updates)
#   ✗ 8 vLLM GPUs instead of 4 (doubled with rollout)
#   ✗ vLLM sharding: 4 engines × TP=2 (vs published 1 × TP=4) —
#     forced by the 6-learner/node geometry; see TP rationale above.
#
# This is a "scaled" reproduction: faster wall-clock, identical update count
# and per-update geometry. The bit-exact 2-node version is
# train_olmo2_1b_rlvr1_4node.sh (2-node Ray on 4-node Slurm allocation,
# 16 GPUs idle, ~43h).
#
# If 6 6 6 6 placement fails in grpo_fast_resource_plan, fallback:
#   NUM_LEARNERS_PER_NODE="4 4 8 8"  (still 24 learners, asymmetric)
#
# Submit:
#   sbatch post_training/scripts/train_olmo2_1b_rlvr1_4node_scaled.sh
# Short test (cap episodes for fast smoke test):
#   TOTAL_EPISODES=100000 EXP_NAME=..._test sbatch ...
# Half-budget legacy run (1302 updates, matches earlier results):
#   TOTAL_EPISODES=2000000 sbatch ...
# Override DPO source:
#   DPO_CKPT=/path/to/dpo sbatch ...

#SBATCH --job-name=olmo2_1b_rlvr1_4node_scaled
#SBATCH --output=${SLURM_LOG_DIR}/olmo2_1b_rlvr1_4node_scaled-%j.out
#SBATCH --error=${SLURM_LOG_DIR}/olmo2_1b_rlvr1_4node_scaled-%j.err
#SBATCH --time=48:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --qos=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --requeue

set -euo pipefail

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

OPEN_INSTRUCT_ROOT="${OPEN_INSTRUCT_ROOT:?set OPEN_INSTRUCT_ROOT to your open-instruct checkout}"
OUT_ROOT="${OUT_ROOT:-${MIDTRAIN_CKPT}/post_training}"
# Uses the completed DPO checkpoint from the preceding stage
DPO_CKPT="${DPO_CKPT:-${OUT_ROOT}/olmo2_1b_baseline28800_dpo/olmo2_1b_baseline28800_dpo}"
EXP_NAME="${EXP_NAME:-olmo2_1b_baseline28800_rlvr1_4node_scaled}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT}/${EXP_NAME}}"

# 4-node scaled: 24 learners, 4 vLLM engines × TP=2, ⇒ 32 GPUs active.
# TP=2 (not TP=4) is required because each node has only 2 free GPUs
# after 6 learners — a TP=4 engine cannot fit on a single node here.
NUM_LEARNERS_PER_NODE="${NUM_LEARNERS_PER_NODE:-6 6 6 6}"
VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-4}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"

# Scaled hyperparameters: rollout batch doubled to 1536 to keep
# samples-per-learner=64 (so local minibatch stays at 32 with num_mini_batches=2).
# total_episodes doubled to 4M so that GRPO updates = 4M / 1536 ≈ 2604 — the
# same update count as the published 2-node recipe.
# LEARNING_RATE: HARDCODED to 5e-7 (AI2 Tulu-3 1B standard). DO NOT change
# and DO NOT make this overridable via env. On 2026-06-24 we caught the
# baselines (ntp/kd_rl1b/kd_rl7b) trained at 5e-6 because the pipeline
# orchestrator's --export=ALL leaked SFT's LR into RLVR via the
# ${LEARNING_RATE:-5e-7} default that used to live here. Hardcoding
# prevents recurrence regardless of caller env hygiene.
LEARNING_RATE="5e-7"
BETA="${BETA:-0.01}"
KL_ESTIMATOR="${KL_ESTIMATOR:-3}"
TOTAL_EPISODES="${TOTAL_EPISODES:-4000000}"
NUM_SAMPLES_PER_PROMPT_ROLLOUT="${NUM_SAMPLES_PER_PROMPT_ROLLOUT:-16}"
NUM_UNIQUE_PROMPTS_ROLLOUT="${NUM_UNIQUE_PROMPTS_ROLLOUT:-96}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_PROMPT_TOKEN_LENGTH="${MAX_PROMPT_TOKEN_LENGTH:-2048}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-2048}"
PACK_LENGTH="${PACK_LENGTH:-4096}"
CHAT_TEMPLATE_NAME="${CHAT_TEMPLATE_NAME:-tulu}"
LOCAL_EVAL_EVERY="${LOCAL_EVAL_EVERY:-5}"
SAVE_FREQ="${SAVE_FREQ:-200}"
SEED="${SEED:-1}"

TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HOME}/.triton/autotune}"
WANDB_PROJECT="${WANDB_PROJECT:-olmo2_1b_post_training}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# Point WANDB_BASE_URL at a self-hosted W&B if you use one.
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
# Default to offline: compute nodes often cannot reach the W&B API, and a
# blocking wandb.init() will stall (or kill) a multi-hour job. Sync afterwards
# with `wandb sync <run-dir>`, or set WANDB_MODE=online if your nodes have
# outbound network access.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
mkdir -p "${WANDB_DIR}"


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

# Use uv-based virtual environment instead of conda
VENV_PATH="${OPEN_INSTRUCT_ROOT:?set OPEN_INSTRUCT_ROOT}/.venv"
source "$VENV_PATH/bin/activate"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

if [[ ! -d "${OPEN_INSTRUCT_ROOT}" ]]; then
  echo "ERROR: OPEN_INSTRUCT_ROOT missing: ${OPEN_INSTRUCT_ROOT}"; exit 1
fi
if [[ ! -d "${DPO_CKPT}" ]]; then
  echo "ERROR: DPO_CKPT missing: ${DPO_CKPT}"; exit 1
fi
mkdir -p "${OUTPUT_DIR}" "${TRITON_CACHE_DIR}"
cd "${OPEN_INSTRUCT_ROOT}"

export PYTHONPATH="${OPEN_INSTRUCT_ROOT}:${PYTHONPATH:-}"
export TRITON_CACHE_DIR
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_CUMEM_ENABLE=0

# Ensure Ray can see all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Prefer system CA bundle; openinstruct env's certifi has been observed to
# point at a missing cacert.pem, which silently breaks wandb artifact upload.
if [[ -z "${CA_BUNDLE:-}" ]]; then
  for c in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
    [[ -f "$c" ]] && { CA_BUNDLE="$c"; break; }
  done
fi
if [[ -n "${CA_BUNDLE:-}" ]]; then
  export REQUESTS_CA_BUNDLE="${CA_BUNDLE}" SSL_CERT_FILE="${CA_BUNDLE}" CURL_CA_BUNDLE="${CA_BUNDLE}"
fi

# Never attach to a stale Ray cluster carried over from a previous job.
unset RAY_ADDRESS RAY_NAMESPACE

RAY_BIN="$(command -v ray)"
PYTHON_BIN="$(command -v python)"
if [[ -z "${RAY_BIN}" || -z "${PYTHON_BIN}" ]]; then
  echo "ERROR: ray or python not on PATH inside openinstruct env"; exit 1
fi

echo "Preflight imports..."
"${PYTHON_BIN}" - <<'PY'
import importlib.util
mods = ["torch", "ray", "deepspeed", "vllm", "open_instruct"]
miss = [m for m in mods if importlib.util.find_spec(m) is None]
if miss:
    raise SystemExit("Missing modules: " + ", ".join(miss))
import ray
print("Ray version:", ray.__version__)
PY

# -----------------------------------------------------------------------------
# Slurm / Ray cluster topology
# -----------------------------------------------------------------------------

# Slurm allocates 4 nodes; this scaled variant uses ALL 4 (RAY_NNODES=4)
# because the 16-learner geometry needs all of them.
SLURM_ALLOC_NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-4}}"
RAY_NNODES="${RAY_NNODES:-4}"
NNODES="${RAY_NNODES}"
GPUS_PER_NODE=8
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-64}"

if (( RAY_NNODES > SLURM_ALLOC_NNODES )); then
  echo "ERROR: RAY_NNODES=${RAY_NNODES} > allocated ${SLURM_ALLOC_NNODES}"; exit 1
fi

# Job-unique GCS port above Ray's worker port range (10002-19999) to avoid
# the worker_ports conflict that Ray 2.53 raises during cluster startup.
RAY_PORT="${RAY_PORT:-$(( 20000 + (SLURM_JOB_ID % 20000) ))}"

mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
# Restrict Ray cluster to the first RAY_NNODES allocated nodes.
RAY_NODES=("${ALL_NODES[@]:0:${RAY_NNODES}}")
HEAD_NODE="${RAY_NODES[0]}"
HEAD_IP="$(getent hosts "${HEAD_NODE}" | awk '{print $1; exit}')"
if [[ -z "${HEAD_IP}" ]]; then
  echo "ERROR: could not resolve HEAD_IP for ${HEAD_NODE}"; exit 1
fi
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

read -r -a NUM_LEARNERS_PER_NODE_ARR <<<"${NUM_LEARNERS_PER_NODE}"
if (( ${#NUM_LEARNERS_PER_NODE_ARR[@]} != RAY_NNODES )); then
  echo "ERROR: NUM_LEARNERS_PER_NODE must list ${RAY_NNODES} Ray-training-node values, got ${#NUM_LEARNERS_PER_NODE_ARR[@]}"
  exit 1
fi

# Bin-pack preflight: each vLLM engine reserves TP GPUs on a single node
# (Ray PACK strategy, see open_instruct/vllm_utils.py:1228). Verify that at
# least VLLM_NUM_ENGINES nodes have >= TP free GPUs, otherwise vLLM placement
# will deadlock silently after weights load.
_VLLM_FEASIBLE_NODES=0
for n in "${NUM_LEARNERS_PER_NODE_ARR[@]}"; do
  free=$(( GPUS_PER_NODE - n ))
  if (( free >= VLLM_TENSOR_PARALLEL_SIZE )); then
    _VLLM_FEASIBLE_NODES=$(( _VLLM_FEASIBLE_NODES + (free / VLLM_TENSOR_PARALLEL_SIZE) ))
  fi
done
if (( _VLLM_FEASIBLE_NODES < VLLM_NUM_ENGINES )); then
  echo "ERROR: vLLM placement infeasible:"
  echo "  num_learners_per_node=${NUM_LEARNERS_PER_NODE_ARR[*]} (GPUS_PER_NODE=${GPUS_PER_NODE})"
  echo "  need ${VLLM_NUM_ENGINES} engines × TP=${VLLM_TENSOR_PARALLEL_SIZE} GPUs (packed per engine)"
  echo "  but only ${_VLLM_FEASIBLE_NODES} TP=${VLLM_TENSOR_PARALLEL_SIZE} slots available across nodes"
  echo "  reduce VLLM_TENSOR_PARALLEL_SIZE or rebalance NUM_LEARNERS_PER_NODE."
  exit 1
fi

echo "Slurm allocated ${SLURM_ALLOC_NNODES} nodes; Ray will use ${RAY_NNODES}:"
printf '  ALL_NODES[%d]=%s\n' $(for i in "${!ALL_NODES[@]}"; do echo "$i"; echo "${ALL_NODES[$i]}"; done)
echo "  RAY_NODES (used): ${RAY_NODES[*]}"
if (( SLURM_ALLOC_NNODES > RAY_NNODES )); then
  IDLE=("${ALL_NODES[@]:${RAY_NNODES}}")
  echo "  IDLE nodes (allocated but unused): ${IDLE[*]}"
fi
echo
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "  HEAD_NODE=${HEAD_NODE}"
echo "  HEAD_IP=${HEAD_IP}"
echo "  RAY_ADDRESS=${RAY_ADDRESS}"
echo
echo "  EXP_NAME=${EXP_NAME}"
echo "  DPO_CKPT=${DPO_CKPT}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo
echo "  num_learners_per_node=${NUM_LEARNERS_PER_NODE_ARR[*]}"
echo "  vllm_num_engines=${VLLM_NUM_ENGINES}  TP=${VLLM_TENSOR_PARALLEL_SIZE}  mem_util=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "  lr=${LEARNING_RATE}  beta=${BETA}  kl=${KL_ESTIMATOR}  episodes=${TOTAL_EPISODES}"
echo "  chat_template=${CHAT_TEMPLATE_NAME}  response_len=${RESPONSE_LENGTH}  pack=${PACK_LENGTH}"

# -----------------------------------------------------------------------------
# Ray cleanup + bootstrap (simplified approach)
# -----------------------------------------------------------------------------

cleanup_ray_all_nodes() {
  echo "Cleaning Ray on the ${RAY_NNODES} Ray nodes..."
  srun --overlap --nodes="${RAY_NNODES}" --ntasks="${RAY_NNODES}" --ntasks-per-node=1 \
       -w "$(IFS=,; echo "${RAY_NODES[*]}")" \
       bash -lc "\"${RAY_BIN}\" stop --force >/dev/null 2>&1 || true" || true
}

trap 'echo "trap: cleaning Ray..."; kill $(jobs -p) 2>/dev/null; cleanup_ray_all_nodes' EXIT

cleanup_ray_all_nodes
sleep 5

# Create Ray temp directory with shorter path to avoid Unix socket path limit (107 bytes)
RAY_TMP_DIR=/tmp/ray_${SLURM_JOB_ID}
mkdir -p "${RAY_TMP_DIR}"

echo "Starting Ray head with direct ray start command..."
"${RAY_BIN}" start \
  --head \
  --node-ip-address="${HEAD_IP}" \
  --port="${RAY_PORT}" \
  --disable-usage-stats \
  --temp-dir="${RAY_TMP_DIR}" \
  --verbose

# Wait for Ray head to become available
echo "Waiting for Ray head to initialize..."
sleep 30

# Check if Ray is running and show status
echo "Checking Ray status..."
"${RAY_BIN}" status --address="${RAY_ADDRESS}" || true

echo "Ray head startup complete. Continuing with worker nodes..."

WORKER_NODES=$((RAY_NNODES - 1))
echo "Starting Ray workers on ${WORKER_NODES} nodes..."
for node in "${RAY_NODES[@]:1}"; do
  echo "  -> worker: ${node}"
  srun --overlap \
       --nodes=1 --ntasks=1 --ntasks-per-node=1 \
       -w "${node}" \
       "${RAY_BIN}" start \
         --address="${RAY_ADDRESS}" \
         --num-cpus="${CPUS_PER_NODE}" \
         --num-gpus="${GPUS_PER_NODE}" \
         --disable-usage-stats \
         --block &
done
sleep 30

echo "Ray status:"
"${RAY_BIN}" status --address="${RAY_ADDRESS}" || echo "WARNING: ray status failed — cluster may still be converging"

# -----------------------------------------------------------------------------
# Driver: open_instruct/grpo_fast.py on the submission host
# -----------------------------------------------------------------------------

TRACKING_ARGS=(--with_tracking --wandb_project "${WANDB_PROJECT}")
[[ -n "${WANDB_ENTITY:-}" ]] && TRACKING_ARGS+=(--wandb_entity "${WANDB_ENTITY}")

echo "Launching grpo_fast.py with RAY_ADDRESS=${RAY_ADDRESS}..."

"${PYTHON_BIN}" -u open_instruct/grpo_fast.py \
    --exp_name "${EXP_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${DPO_CKPT}" \
    --tokenizer_name_or_path "${DPO_CKPT}" \
    --add_bos \
    --chat_template_name "${CHAT_TEMPLATE_NAME}" \
    --apply_verifiable_reward true \
    --beta "${BETA}" \
    --kl_estimator "${KL_ESTIMATOR}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler_type constant \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length "${MAX_PROMPT_TOKEN_LENGTH}" \
    --response_length "${RESPONSE_LENGTH}" \
    --pack_length "${PACK_LENGTH}" \
    --num_unique_prompts_rollout "${NUM_UNIQUE_PROMPTS_ROLLOUT}" \
    --num_samples_per_prompt_rollout "${NUM_SAMPLES_PER_PROMPT_ROLLOUT}" \
    --num_mini_batches "${NUM_MINI_BATCHES}" \
    --num_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size 1 \
    --num_learners_per_node "${NUM_LEARNERS_PER_NODE_ARR[@]}" \
    --vllm_num_engines "${VLLM_NUM_ENGINES}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_enable_prefix_caching \
    --deepspeed_stage 2 \
    --deepspeed_zpg 12 \
    --gradient_checkpointing \
    --non_stop_penalty \
    --non_stop_penalty_value 0.0 \
    --stop_strings "</answer>" \
    --remap_verifier ifeval=ifeval_old \
    --temperature 1.0 \
    --total_episodes "${TOTAL_EPISODES}" \
    --local_eval_every "${LOCAL_EVAL_EVERY}" \
    --save_freq "${SAVE_FREQ}" \
    --checkpoint_state_dir "${OUTPUT_DIR}/checkpoint_state" \
    --checkpoint_state_freq "${SAVE_FREQ}" \
    --seed "${SEED}" \
    --push_to_hub false \
    --hf_entity "${HF_ENTITY:-}" \
    "${TRACKING_ARGS[@]}"

# Normalize RoPE config schema for older inference stacks (safety net).
# RLVR1 historically saved the flat schema (so eval works), but if any future
# open-instruct change routes through transformers.save_pretrained, the new
# nested `rope_parameters` schema would silently break vllm==0.11.0 /
# transformers==4.57.3 evals (rope_theta falls back to 10000, attention
# corrupts, output collapses to "4. 4. 4. ..."). See
# post_training/scripts/normalize_rope_config.py for context. Idempotent.
echo "=== Normalizing RoPE config schema for older inference stacks ==="
"${PYTHON_BIN}" "$(dirname "$0")/normalize_rope_config.py" "${OUTPUT_DIR}" \
    || echo "WARNING: normalize_rope_config.py failed; eval may produce degenerate output"

echo "=== RLVR-1 done -> ${OUTPUT_DIR} ==="

# Auto-launch the 13-task OLMES eval on the RLVR1 step_2600 checkpoint.
# eval_single_rlvr1.sh expects RLVR1_PARENT_DIR + EXP_NAME and resolves the
# most recent __1__<ts>_checkpoints/step_2600/ snapshot. EVAL_OUTPUT_DIR can
# be overridden; default is olmes/results/posttrain/${EXP_NAME}-2604steps.
# Set AUTO_EVAL=0 to opt out.
if [ "${AUTO_EVAL:-1}" = "1" ]; then
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OLMES_ROOT}/results/posttrain/${EXP_NAME}-2604steps}"
    EVAL_QOS="${EVAL_QOS:-CHANGE_ME}"
    # Partition is inferred from QoS prefix (h100_* -> h100, h200_* -> h200).
    EVAL_PARTITION="${EVAL_QOS%%_*}"
    echo "=== Auto-submitting OLMES eval -> ${EVAL_OUTPUT_DIR} (qos=${EVAL_QOS}) ==="
    # NO `sbatch --export=` (cgroup v2 -> Priority=0 held). Vars ride in
    # sbatch's own env and reach the job via the default --export=ALL.
    env RLVR1_PARENT_DIR="${OUTPUT_DIR}" EXP_NAME="${EXP_NAME}" \
        OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
        sbatch --qos="${EVAL_QOS}" --partition="${EVAL_PARTITION}" \
        ${OLMES_ROOT}/eval_single_rlvr1.sh \
        || echo "WARNING: eval sbatch failed (training output is still saved at ${OUTPUT_DIR})"
fi
