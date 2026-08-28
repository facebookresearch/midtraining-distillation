#!/usr/bin/env bash
# scripts/lingua_to_hf.sh — consolidate a Lingua DCP checkpoint and convert it
# back to HuggingFace format.
#
# Usage:
#   source scripts/env.sh
#   bash scripts/lingua_to_hf.sh <ckpt_step_dir> [output_dir]
#
# Args:
#   <ckpt_step_dir>   Lingua training checkpoint dir, i.e. one of the per-step
#                     dirs under ${MIDTRAIN_ROOT}/<recipe>/checkpoints/, e.g.
#                       ${MIDTRAIN_ROOT}/switch_distill/checkpoints/0000028800
#                     (the dir containing `.metadata` and `*.distcp`).
#   [output_dir]      Where to write the HF model. Defaults to
#                       <ckpt_step_dir>/hf
#
# Optional env:
#   TOKENIZER_PATH    HF tokenizer dir used to validate vocab + set BOS/EOS.
#                     Defaults to ${STUDENT_HF_PATH} (set by env.sh).
#
# What this does (two stages, matching the upstream lingua workflow):
#   1. consolidate: DCP shards → consolidated/consolidated.pth + params.json
#      via setup/consolidate.py (idempotent; skips if consolidated dir exists).
#   2. convert:     consolidated/ → HuggingFace LlamaForCausalLM / Olmo2ForCausalLM
#      via setup/convert_consolidated_lingua_ckpt_to_hf.py.

set -euo pipefail

: "${STUDENT_HF_PATH:?source scripts/env.sh first}"

CKPT_DIR="${1:?usage: bash scripts/lingua_to_hf.sh <ckpt_step_dir> [output_dir]}"
OUT_DIR="${2:-${CKPT_DIR}/hf}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${STUDENT_HF_PATH}}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${CKPT_DIR}/.metadata" ]]; then
    echo "[lingua→hf] ERROR: ${CKPT_DIR} does not contain a DCP .metadata file." >&2
    echo "             Pass a per-step checkpoint dir (the one with __0_0.distcp)." >&2
    exit 1
fi

CONSOLIDATED_DIR="${CKPT_DIR}/consolidated"

pushd "${ROOT_DIR}" >/dev/null

echo "[lingua→hf] Stage 1: consolidate DCP → ${CONSOLIDATED_DIR}"
python setup/consolidate.py --ckpt_dir "${CKPT_DIR}"

mkdir -p "${OUT_DIR}"
echo "[lingua→hf] Stage 2: consolidated → HF at ${OUT_DIR}"
python setup/convert_consolidated_lingua_ckpt_to_hf.py \
    --input_dir "${CONSOLIDATED_DIR}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --output_dir "${OUT_DIR}"

popd >/dev/null

echo "[lingua→hf] Done. HF model at: ${OUT_DIR}"
