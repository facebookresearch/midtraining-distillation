"""scripts/smoke_test.py — end-to-end smoke test for the training setup.

Confirms:
  1. `source scripts/env.sh` set all required vars.
  2. `scripts/fetch_models.sh` populated ${STUDENT_HF_PATH} (HF) and
     ${STUDENT_INIT_PATH} (Lingua DCP root with `consolidated/`).
  3. The HF→Lingua-DCP conversion is numerically faithful: a forward pass through
     the HF student and the Lingua-loaded student produce matching top-5
     predictions on a fixed prompt.
  4. (optional) ${TEACHER_7B_PATH} loads as a HF causal LM.

Run:
    source scripts/env.sh
    python scripts/smoke_test.py
    python scripts/smoke_test.py --check-teacher    # also tries the 7B teacher

Requires one GPU (~5 GB for the 1B student forward, ~16 GB if --check-teacher).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apps.main.generate import load_consolidated_model_and_tokenizer  # noqa: E402

PROMPT = "The capital of France is"
TOP_K = 5
MAX_TOP1_LOGIT_DIFF = 0.5  # bf16, single fwd; loose but catches a real bug.


def check_env():
    required = [
        "STUDENT_HF_PATH",
        "STUDENT_INIT_PATH",
        "TEACHER_1B_PATH",
        "TEACHER_7B_PATH",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"[FAIL] Missing env vars: {missing}. Did you `source scripts/env.sh`?")
        sys.exit(1)
    print("[ok] env vars present")
    for k in required:
        print(f"      {k}={os.environ[k]}")


def check_paths():
    hf = Path(os.environ["STUDENT_HF_PATH"])
    dcp_root = Path(os.environ["STUDENT_INIT_PATH"])
    consolidated = dcp_root / "consolidated" / "consolidated.pth"
    params = dcp_root / "consolidated" / "params.json"
    metadata = dcp_root / ".metadata"

    for label, p in [
        ("HF config", hf / "config.json"),
        ("DCP metadata", metadata),
        ("Lingua consolidated.pth", consolidated),
        ("Lingua params.json", params),
    ]:
        if not p.exists():
            print(f"[FAIL] missing {label}: {p}")
            print("       Run `bash scripts/fetch_models.sh ONLY=student` first.")
            sys.exit(1)
    print(f"[ok] HF student at {hf}")
    print(f"[ok] Lingua DCP root at {dcp_root}")


def check_student_parity():
    hf_path = os.environ["STUDENT_HF_PATH"]
    dcp_root = Path(os.environ["STUDENT_INIT_PATH"])
    consolidated_dir = dcp_root / "consolidated"

    print("[..] loading HF student")
    tok = AutoTokenizer.from_pretrained(hf_path)
    hf_model = (
        AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )

    print("[..] loading Lingua-consolidated student")
    lingua_model, _, _ = load_consolidated_model_and_tokenizer(str(consolidated_dir))

    input_ids = tok(PROMPT, return_tensors="pt")["input_ids"].cuda()
    print(f"[..] forward on prompt: {PROMPT!r} ({input_ids.shape[1]} tokens)")
    with torch.inference_mode():
        hf_logits = hf_model(input_ids).logits[0, -1].float()
        lingua_logits = lingua_model(input_ids)[0, -1].float()

    hf_top = torch.topk(hf_logits, TOP_K)
    li_top = torch.topk(lingua_logits, TOP_K)
    hf_top1 = hf_top.indices[0].item()
    li_top1 = li_top.indices[0].item()
    top1_diff = (hf_logits[hf_top1] - lingua_logits[hf_top1]).abs().item()

    print(
        f"     HF top-{TOP_K}:    {[tok.decode([t]) for t in hf_top.indices.tolist()]}"
    )
    print(
        f"     Lingua top-{TOP_K}: {[tok.decode([t]) for t in li_top.indices.tolist()]}"
    )
    print(f"     |HF - Lingua| at HF-top1: {top1_diff:.4f}")
    print(
        f"     max |HF - Lingua|:       {(hf_logits - lingua_logits).abs().max().item():.4f}"
    )

    if hf_top1 != li_top1:
        print(
            f"[FAIL] top-1 mismatch: HF={hf_top1} ({tok.decode([hf_top1])!r}) "
            f"vs Lingua={li_top1} ({tok.decode([li_top1])!r})"
        )
        sys.exit(2)
    if top1_diff > MAX_TOP1_LOGIT_DIFF:
        print(f"[FAIL] top-1 logit diff {top1_diff:.3f} > tol {MAX_TOP1_LOGIT_DIFF}")
        sys.exit(2)
    print("[ok] HF / Lingua student agree on top-1 prediction")


def check_teacher():
    p = os.environ["TEACHER_7B_PATH"]
    print(f"[..] loading 7B teacher from {p}")
    AutoTokenizer.from_pretrained(p)
    model = (
        AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(p)
    input_ids = tok(PROMPT, return_tensors="pt")["input_ids"].cuda()
    with torch.inference_mode():
        _ = model(input_ids).logits
    print("[ok] 7B teacher forwards without error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check-teacher",
        action="store_true",
        help="Also load the 7B teacher and run one forward (needs ~16 GB VRAM).",
    )
    args = ap.parse_args()

    check_env()
    check_paths()
    check_student_parity()
    if args.check_teacher:
        check_teacher()
    print("\n[all green]")


if __name__ == "__main__":
    main()
