"""scripts/hf_generate.py — load a converted HF checkpoint and print completions
for a few fixed prompts.

Usage:
    source scripts/env.sh
    python scripts/hf_generate.py <hf_model_dir>

`<hf_model_dir>` is the output of `scripts/lingua_to_hf.sh`, i.e. the dir
containing `config.json` and `*.safetensors`. The tokenizer is taken from
`${TOKENIZER_PATH}` (= `${STUDENT_HF_PATH}` by default) unless `--tokenizer`
is passed.

Single GPU, bf16, greedy by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time, in a small village by the sea,",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "model_dir", type=Path, help="HF model dir (output of lingua_to_hf.sh)."
    )
    p.add_argument(
        "--tokenizer",
        default=os.environ.get("TOKENIZER_PATH"),
        help="HF tokenizer dir. Defaults to $TOKENIZER_PATH.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy. >0 enables sampling.",
    )
    p.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Override default prompts (repeatable).",
    )
    args = p.parse_args()

    if not (args.model_dir / "config.json").exists():
        raise SystemExit(
            f"[hf_generate] no config.json in {args.model_dir}; "
            "pass the HF dir produced by scripts/lingua_to_hf.sh."
        )
    if not args.tokenizer:
        raise SystemExit(
            "[hf_generate] no tokenizer; pass --tokenizer or "
            "source scripts/env.sh so TOKENIZER_PATH is set."
        )

    prompts = args.prompt or DEFAULT_PROMPTS

    print(f"[hf_generate] model: {args.model_dir}")
    print(f"[hf_generate] tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )

    do_sample = args.temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tok.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature

    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(**ids, **gen_kwargs)
        text = tok.decode(out[0], skip_special_tokens=True)
        print(f"\n--- [{i}] prompt ---\n{prompt}\n--- completion ---\n{text}")


if __name__ == "__main__":
    main()
