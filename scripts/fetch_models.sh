#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# scripts/fetch_models.sh — download the three HF checkpoints used by the
# supported recipes and convert the student init into a Lingua DCP checkpoint.
#
# Usage:
#   source scripts/env.sh
#   bash scripts/fetch_models.sh                  # default: all of {student, 1b_teacher, 7b_teacher}
#   ONLY=student bash scripts/fetch_models.sh     # download only the student init
#   ONLY=1b_teacher,7b_teacher bash scripts/fetch_models.sh
#
# What this does:
#   - student     → downloads allenai/OLMo-2-0425-1B at revision
#                   `${STUDENT_REVISION}` (default stage1-step928646-tokens3897B).
#                   HF weights land at ${STUDENT_HF_PATH}; Lingua DCP weights
#                   (`.metadata`, `__0_0.distcp`, `consolidated/`) land at
#                   ${STUDENT_INIT_PATH}.
#   - 1b_teacher  → snapshot-downloads allenai/OLMo-2-0425-1B-Instruct to ${TEACHER_1B_PATH}
#                   (HF format; teachers are loaded via AutoModelForCausalLM, no DCP needed).
#   - 7b_teacher  → snapshot-downloads allenai/OLMo-2-1124-7B-Instruct to ${TEACHER_7B_PATH}.
#                   (not needed by any released recipe; kept for convenience).

set -euo pipefail

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy X2P_PROXY_URL

: "${STUDENT_INIT_PATH:?source scripts/env.sh first}"
: "${STUDENT_HF_PATH:?source scripts/env.sh first}"
: "${TEACHER_1B_PATH:?source scripts/env.sh first}"
: "${TEACHER_7B_PATH:?source scripts/env.sh first}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

ONLY="${ONLY:-student,1b_teacher,7b_teacher}"

STUDENT_REVISION="${STUDENT_REVISION:-stage1-step1907359-tokens4001B}"
DTYPE="${DTYPE:-float32}"


want() { [[ ",${ONLY}," == *",$1,"* ]]; }

snapshot_download() {
    local repo_id="$1" local_dir="$2"
    mkdir -p "${local_dir}"
    echo "[fetch] huggingface-cli download ${repo_id} → ${local_dir}"
    huggingface-cli download "${repo_id}" \
        --local-dir "${local_dir}" \
        --local-dir-use-symlinks False
}

if want student; then
    echo "[fetch] Student init → HF weights at ${STUDENT_HF_PATH}"
    snapshot_download "allenai/OLMo-2-0425-1B" "${STUDENT_HF_PATH}"

    echo "[fetch] Student init → Lingua DCP at ${STUDENT_INIT_PATH}"
    pushd "${ROOT_DIR}" >/dev/null
    python setup/hf_to_lingua_dcp.py \
        --model allenai/OLMo-2-0425-1B \
        --revision "${STUDENT_REVISION}" \
        --output "${STUDENT_INIT_PATH}" \
        --dtype "${DTYPE}"
    popd >/dev/null
fi

if want 1b_teacher; then
    snapshot_download "allenai/OLMo-2-0425-1B-Instruct" "${TEACHER_1B_PATH}"
fi

if want 7b_teacher; then
    snapshot_download "allenai/OLMo-2-1124-7B-Instruct" "${TEACHER_7B_PATH}"
fi


echo "[fetch] Done. Re-check the env-var summary:"
echo "  STUDENT_INIT_PATH (Lingua DCP root): ${STUDENT_INIT_PATH}"
echo "  STUDENT_HF_PATH   (HF mirror):       ${STUDENT_HF_PATH}"
echo "  TEACHER_1B_PATH:                     ${TEACHER_1B_PATH}"
echo "  TEACHER_7B_PATH:                     ${TEACHER_7B_PATH}"
