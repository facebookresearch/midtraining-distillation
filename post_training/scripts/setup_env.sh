#!/usr/bin/env bash
# Fresh uv-based environment setup for OLMo-2 1B SFT/DPO/RLVR.
# Creates a clean virtual environment from scratch, bypassing conda dependency hell.
#
# Run interactively:
#   bash post_training/scripts/setup_env.sh

set -euo pipefail

export TMPDIR="${TMPDIR:-/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME}/.cache/pip}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

VENV_PATH="${OPEN_INSTRUCT_ROOT:?set OPEN_INSTRUCT_ROOT}/.venv"

echo "=== Step 1: Installing uv (ultra-fast package manager) ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "=== Step 2: Creating fresh virtual environment with uv ==="
cd ${OPEN_INSTRUCT_ROOT}
rm -rf "$VENV_PATH"  # Remove any existing venv
uv venv "$VENV_PATH" --python 3.12

echo "=== Step 3: Installing all dependencies via uv sync (uses uv.lock) ==="
# This installs ALL dependencies including torch, flash-attn pre-built wheels, etc.
# Much faster than conda/pip and guarantees compatible versions
uv sync --frozen || {
    echo "ERROR: uv sync failed"
    exit 1
}

echo "=== Verifying imports ==="
cd ${OPEN_INSTRUCT_ROOT}
python - <<'PY'
import importlib
for name in ("torch","torchvision","transformers","vllm","liger_kernel","flash_attn"):
    try:
        m = importlib.import_module(name)
        print(f"  OK  {name:18s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  --  {name:18s} not importable ({e.__class__.__name__})")

# The real test: can open_instruct entrypoints import?
for ep in ("open_instruct.finetune", "open_instruct.dpo_tune_cache", "open_instruct.grpo_fast"):
    try:
        importlib.import_module(ep)
        print(f"  OK  {ep}")
    except Exception as e:
        print(f"  XX  {ep}: {e}")
PY

echo "=== setup_env.sh done ==="
