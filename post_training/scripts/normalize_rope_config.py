#!/usr/bin/env python3
"""Normalize HF config.json to the flat RoPE schema for older inference stacks.

Background
----------
open-instruct's venv pins `transformers==5.4.0`, which saves
`config.json` with the new nested schema:

    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}

Older inference stacks we care about (e.g. `olmes` env with
`transformers==4.57.3` + `vllm==0.11.0`) only know the flat schema
(`rope_theta` at top level) and silently fall back to `rope_theta=10000`
when they see only the new key. For OLMo-2 (which trained at
`rope_theta=500000`) this catastrophically corrupts attention at every
non-trivial position and produces degenerate generations like
`" 4. 4. 4. 4. ..."` (observed on local SFT/DPO checkpoints, gsm8k::tulu
~0.02 vs RLVR1 ~0.48).

Fix
---
After every training run, rewrite the saved config.json so it matches the
shape RLVR1 already saves: `rope_theta` flat at the top level,
`rope_scaling` (null unless we're actually scaling), and no
`rope_parameters` key at all. We drop the nested key because mixed
schemas cause confusion and have already bitten us once; the flat schema
is what every consumer in our stack (vllm, olmes, transformers 4.x) reads.

If a future loader needs the nested form back, it can be re-derived from
the flat fields trivially. Idempotent: re-running it on an
already-normalized config is a no-op.

Usage
-----
    python normalize_rope_config.py PATH [PATH ...]

Each PATH may be either an HF model directory (containing config.json)
or a parent that contains one. We walk up to depth 2 so callers can pass
the SFT/DPO `OUTPUT_DIR` and we'll find `OUTPUT_DIR/<EXP_NAME>/config.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _find_config_jsons(root: Path, max_depth: int = 2) -> list[Path]:
    """Find config.json under root, up to max_depth levels deep."""
    if root.is_file() and root.name == "config.json":
        return [root]
    if not root.is_dir():
        return []

    found: list[Path] = []
    direct = root / "config.json"
    if direct.is_file():
        found.append(direct)

    if max_depth > 0:
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                found.extend(_find_config_jsons(child, max_depth - 1))
    return found


def normalize(cfg_path: Path) -> bool:
    """Rewrite cfg_path so RoPE matches the flat-only schema RLVR1 saves.

    The post-condition is: `rope_theta` is a top-level float, `rope_scaling`
    is set (null in the default case), and `rope_parameters` is absent.

    Returns True if the file was modified, False otherwise. Idempotent.
    """
    with cfg_path.open() as f:
        cfg = json.load(f)

    rope_params = cfg.get("rope_parameters")
    has_nested = isinstance(rope_params, dict)

    if has_nested:
        nested_theta = rope_params.get("rope_theta")
        nested_type = rope_params.get("rope_type", "default")
        if nested_theta is None:
            return False

        desired_theta = float(nested_theta)
        if nested_type and nested_type != "default":
            desired_scaling = {
                k: v for k, v in rope_params.items() if k != "rope_theta"
            }
        else:
            desired_scaling = None
    else:
        flat_theta = cfg.get("rope_theta")
        if flat_theta is None:
            return False
        desired_theta = float(flat_theta)
        desired_scaling = cfg.get("rope_scaling")

    already_flat_only = (
        not has_nested
        and cfg.get("rope_theta") == desired_theta
        and cfg.get("rope_scaling", None) == desired_scaling
        and "rope_scaling" in cfg
    )
    if already_flat_only:
        return False

    cfg.pop("rope_parameters", None)
    cfg["rope_theta"] = desired_theta
    cfg["rope_scaling"] = desired_scaling

    backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(cfg_path.read_text())

    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", help="HF model dir(s) or config.json path(s)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="How deep to look for config.json under each input dir (default: 2)",
    )
    args = parser.parse_args(argv)

    any_changed = False
    any_seen = False
    for raw in args.paths:
        root = Path(raw)
        cfgs = _find_config_jsons(root, args.max_depth)
        if not cfgs:
            print(
                f"[normalize_rope_config] no config.json under {root}", file=sys.stderr
            )
            continue
        for cfg_path in cfgs:
            any_seen = True
            try:
                changed = normalize(cfg_path)
            except Exception as e:
                print(
                    f"[normalize_rope_config] FAILED {cfg_path}: {e}", file=sys.stderr
                )
                return 2
            if changed:
                any_changed = True
                print(f"[normalize_rope_config] flattened RoPE schema in {cfg_path}")
            else:
                print(f"[normalize_rope_config] ok (no change): {cfg_path}")

    if not any_seen:
        return 1
    return 0 if any_seen else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
