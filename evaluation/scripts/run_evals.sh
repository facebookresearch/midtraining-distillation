#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Run the paper's canonical evaluation suite, grouped by reported category.
#
# Usage:
#   MODEL=/path/to/hf/checkpoint OUTPUT_DIR=/path/to/results \
#   STAGE=mid bash evaluation/scripts/run_evals.sh
#
# Required:
#   MODEL         HF checkpoint directory (contains config.json)
#   OUTPUT_DIR    where olmes writes per-task results
#
# Optional:
#   STAGE=mid|post   default mid. Selects the alias set (see "Suites" below).
#   TASK_GROUPS="..."  subset of REASONING KNOWLEDGE FACTUAL_RECALL
#                    INSTRUCTION_FOLLOWING. Default: all applicable to STAGE.
#   TOKENIZER        default ${MODEL}
#   LOCAL=1          run inline instead of submitting one sbatch per group
#   DRY_RUN=1        print what would run, submit nothing
#   OLMES_CONDA_ENV  conda env with olmes installed (see evaluation/README.md)

set -euo pipefail

: "${MODEL:?set MODEL to an HF checkpoint directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a results directory}"
STAGE="${STAGE:-mid}"
TOKENIZER="${TOKENIZER:-${MODEL}}"

case "${STAGE}" in
    mid|post) ;;
    *) echo "ERROR: STAGE must be 'mid' or 'post', got '${STAGE}'" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Task groups. Aliases that differ by stage are selected below; everything else
# is identical at both stages.
# ---------------------------------------------------------------------------
if [[ "${STAGE}" == "mid" ]]; then
    GSM8K="gsm8k::olmes"; BBH="bbh:cot::olmes"
    DROP="drop::olmes";   MATH="minerva_math_500::olmes"
    MMLU="mmlu:mc::olmes"
else
    GSM8K="gsm8k::tulu";  BBH="bbh:cot::tulu"
    DROP="drop::llama3";  MATH="minerva_math_500::tulu"
    MMLU="mmlu:mc::tulu"
fi

# REASONING (6). GSM-Symbolic is sampled at T=0.2; GSM-Plus is greedy over the
# full 10552-example split. Both are memory-hungry -- see MEM_REASONING below.
REASONING=("${GSM8K}" "gsm_symbolic::olmo3" "gsm_plus::none" "${BBH}" "${DROP}" "${MATH}")

# KNOWLEDGE / multiple-choice (6). ARC-C, OpenBookQA and WinoGrande are scored
# as answer-selection (`:mc`), not ranked-completion (`:rc`).
KNOWLEDGE=("${MMLU}" "mmlu_pro:mc::none" "arc_challenge:mc::olmes"
           "openbookqa:mc::olmes" "winogrande:mc::olmes"
           "agi_eval_english:1shot::olmes")

# FACTUAL_RECALL (3), full splits: TriviaQA 7993, NaturalQs 3610, SimpleQA 4321.
FACTUAL_RECALL=("triviaqa::olmes" "naturalqs::olmes" "simpleqa::no-judge-short-form")

# INSTRUCTION_FOLLOWING (1). Needs a chat template, so post-training only.
# Report `inst_level_loose_acc`, not olmes' default `prompt_level_loose_acc`.
INSTRUCTION_FOLLOWING=("ifeval::tulu")

if [[ "${STAGE}" == "mid" ]]; then
    DEFAULT_TASK_GROUPS="REASONING KNOWLEDGE FACTUAL_RECALL"
else
    DEFAULT_TASK_GROUPS="REASONING KNOWLEDGE FACTUAL_RECALL INSTRUCTION_FOLLOWING"
fi
# NB: GROUPS (unsuffixed) is a bash builtin holding the caller's group ids.
TASK_GROUPS="${TASK_GROUPS:-${DEFAULT_TASK_GROUPS}}"

# ---------------------------------------------------------------------------
# Per-group resources. GSM-Symbolic / GSM-Plus exhaust host RAM at the default
# datasets cache setting, so REASONING gets a larger allocation and
# DATASETS_IN_MEMORY_MAX_SIZE=0.
# ---------------------------------------------------------------------------
MEM_REASONING="${MEM_REASONING:-1000G}"
MEM_DEFAULT="${MEM_DEFAULT:-200G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
GPUS="${GPUS:-1}"

MODEL_ARGS=$(printf '{"trust_remote_code": true, "gpu_memory_utilization": 0.8, "max_length": 4096, "tokenizer": "%s", "tokenizer_revision": "main"}' "${TOKENIZER}")

run_group() {
    local group="$1"; shift
    local tasks=("$@")
    local out="${OUTPUT_DIR}/${STAGE}/${group,,}"
    local mem="${MEM_DEFAULT}"
    local extra_env=""
    if [[ "${group}" == "REASONING" ]]; then
        mem="${MEM_REASONING}"
        extra_env="export DATASETS_IN_MEMORY_MAX_SIZE=0"
    fi

    # Listing the plan needs nothing from the environment, so check this first.
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "=== ${group} (${#tasks[@]} tasks, mem=${mem}) -> ${out}"
        printf '  %s\n' "${tasks[@]}"
        return
    fi
    : "${OLMES_CONDA_ENV:?set OLMES_CONDA_ENV (see evaluation/README.md)}"

    local body
    body=$(cat <<EOF
set -euo pipefail
eval "\$(conda shell.bash hook)"
conda activate "${OLMES_CONDA_ENV}"

# olmes must not inherit the training job's distributed environment.
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
unset SLURM_PROCID SLURM_LOCALID SLURM_NTASKS SLURM_NODEID
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
${extra_env}

mkdir -p "${out}"
olmes \\
    --model "${MODEL}" \\
    --model-type vllm \\
    --model-args '${MODEL_ARGS}' \\
    --task ${tasks[*]} \\
    --output-dir "${out}"
EOF
)

    if [[ -n "${LOCAL:-}" ]]; then
        echo "=== ${group}: running inline -> ${out}"
        bash -c "${body}"
        return
    fi
    echo "=== ${group}: submitting (mem=${mem}) -> ${out}"
    sbatch --job-name="eval_${STAGE}_${group,,}" \
        --output="${OUTPUT_DIR}/${STAGE}/${group,,}-%j.out" \
        --error="${OUTPUT_DIR}/${STAGE}/${group,,}-%j.err" \
        --time="${TIME_LIMIT}" --gres="gpu:${GPUS}" --cpus-per-task=8 --mem="${mem}" \
        ${SLURM_ACCOUNT:+--account="${SLURM_ACCOUNT}"} \
        ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
        ${SLURM_PARTITION:+--partition="${SLURM_PARTITION}"} \
        --wrap "${body}"
}

mkdir -p "${OUTPUT_DIR}/${STAGE}"
echo "model=${MODEL}"
echo "stage=${STAGE}  groups=${TASK_GROUPS}"

for g in ${TASK_GROUPS}; do
    case "${g}" in
        REASONING)             run_group REASONING "${REASONING[@]}" ;;
        KNOWLEDGE)             run_group KNOWLEDGE "${KNOWLEDGE[@]}" ;;
        FACTUAL_RECALL)        run_group FACTUAL_RECALL "${FACTUAL_RECALL[@]}" ;;
        INSTRUCTION_FOLLOWING)
            if [[ "${STAGE}" == "mid" ]]; then
                echo "skip INSTRUCTION_FOLLOWING: needs a chat template, post-training only" >&2
            else
                run_group INSTRUCTION_FOLLOWING "${INSTRUCTION_FOLLOWING[@]}"
            fi ;;
        *) echo "ERROR: unknown group '${g}'" >&2; exit 1 ;;
    esac
done
