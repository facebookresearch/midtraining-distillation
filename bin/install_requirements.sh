#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# bin/install_requirements.sh
#
# Set up the conda env for midtraining-distillation training, eval, and analysis.
# Creates the env if it doesn't exist, then installs deps into it. Safe to
# re-run -- pip skips already-satisfied packages.
#
# Pre-reqs:
#   * conda is installed and `conda` is on PATH.
#   * CUDA 12.1-compatible drivers on the host (matches the torch wheels below).
#
# Usage:
#   bash bin/install_requirements.sh                  # uses env name from $LINGUA_CONDA_ENV (default: midtraining-distillation)
#   LINGUA_CONDA_ENV=myenv bash bin/install_requirements.sh
#
# Afterwards, activate the env yourself for interactive work:
#   conda activate midtraining-distillation
set -euo pipefail

ENV_NAME="${LINGUA_CONDA_ENV:-midtraining-distillation}"
PY_VERSION="${PY_VERSION:-3.11}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Make `conda activate` work inside a non-interactive script. Try the env-var
# override first (set by scripts/env.sh on this cluster), then fall back to
# `conda info --base`.
if [ -n "${CONDA_PROFILE_SH:-}" ] && [ -f "${CONDA_PROFILE_SH}" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_PROFILE_SH}"
elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not on PATH. Install miniconda first." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[install] conda env '${ENV_NAME}' already exists -- reusing"
else
    echo "[install] creating conda env '${ENV_NAME}' (python=${PY_VERSION})"
    conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
fi

conda activate "${ENV_NAME}"
echo "[install] CONDA_PREFIX=${CONDA_PREFIX}"
echo "[install] ROOT_DIR=${ROOT_DIR}"

# --- 1. Lingua deps from requirements.txt (everything except the CUDA wheels) ---
echo "[install] pip install -r requirements.txt"
pip install -r "${ROOT_DIR}/requirements.txt"

# --- 2. Torch + xformers (CUDA 12.1 wheels) ---
echo "[install] torch 2.5.0 + xformers 0.0.28.post2 (cu121)"
pip install torch==2.5.0 xformers==0.0.28.post2 \
    --index-url https://download.pytorch.org/whl/cu121

# --- 3. Flash-attention (needs torch pre-installed) ---
echo "[install] flash-attn 2.7.4.post1"
pip install flash-attn==2.7.4.post1 --no-build-isolation

echo
echo "[install] Done. Sanity-check imports:"
python - <<'PY'
import importlib, sys
mods = [
    "torch", "xformers", "flash_attn",
    "transformers", "tokenizers", "huggingface_hub", "datasets",
    "omegaconf", "wandb", "tiktoken",
    "numpy", "pandas", "matplotlib",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, repr(e)))
if missing:
    print("MISSING / broken:")
    for m, e in missing:
        print(f"  - {m}: {e}")
    sys.exit(1)
print("All checked imports OK.")
PY

echo "[install] Finished installing requirements into env '${ENV_NAME}'."
echo "[install] Activate it for interactive work:  conda activate ${ENV_NAME}"
