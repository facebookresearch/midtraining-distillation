# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Trimmed for release: this module previously also hosted the lm-eval-harness
# evaluation backend (EvalHarnessLM / launch_eval / LMHarnessArgs). Every
# released recipe evaluates with OLMES via apps/main/eval_olmes.py, so only the
# validation-perplexity path -- which eval_olmes.py imports -- is kept here.

from collections import defaultdict
from dataclasses import dataclass, field
import logging
import os
from typing import List, Optional

from lingua.data import init_choice_state, setup_sources
from lingua.distributed import dist_mean_dict, get_global_rank, get_world_size

logger = logging.getLogger()


@dataclass
class ValidationArgs:
    max_steps: Optional[int] = (
        None  # If None the whole validation file is used -> /!\ This number of steps is gpu dependent (100 max steps on 8 gpus = 800 steps on 1 gpu)
    )
    use_val_from_train_src: bool = True  # Use the validation set from training sources
    root_dir: str = ""
    sources: List[str] = field(default_factory=list)  # Other sources to eval on


def eval_on_val(generator, val_args: ValidationArgs, train_cfg):
    srcs = {}

    if val_args.use_val_from_train_src:
        for src in val_args.sources:
            path = os.path.join(val_args.root_dir, src)
            srcs[path] = 1.0
        for src in train_cfg.data.sources:
            path = os.path.join(train_cfg.data.root_dir, src)
            srcs[path] = 1.0
    else:
        for src in val_args.sources:
            path = os.path.join(val_args.root_dir, src)
            srcs[path] = 1.0
    multi_state = init_choice_state(
        "", srcs, 0, get_global_rank(), get_world_size(), "*.val.jsonl"
    )
    path_to_iter = setup_sources(multi_state)

    max_gen_len = generator.max_gen_len
    # We temporarily lower max gen len
    generator.max_gen_len = 1

    all_val_metrics = {}
    for src in path_to_iter:
        jsonl_iterator = path_to_iter[src]
        texts = []
        logger.info(f"Running validation on {src}...")
        for step, (content, state) in enumerate(jsonl_iterator):
            if state["current_iter"] > 0 or (
                val_args.max_steps is not None and step >= val_args.max_steps
            ):
                break
            if "text" in content:
                text = content["text"]
            elif "content" in content:
                text = content["content"]
            else:
                continue
            if text.strip() != "":
                texts.append(text)

        _, loglikelihood, _ = generator.generate(texts)

        metrics = defaultdict(list)
        for i, ll in enumerate(loglikelihood):
            try:
                tmp = ll.sum().item()
                metrics["nll"].append(tmp)
                metrics["nll_per_token"].append(tmp / len(ll))
                metrics["nll_per_char"].append(tmp / len(texts[i]))

                metrics["avg_seqlen"].append(len(ll))
            except:
                print(f"[Warning] Empty input at step {i}")

        for m in metrics:
            metrics[m] = sum(metrics[m]) / len(metrics[m])
        metrics.update(dist_mean_dict(metrics))
        logger.info(f"Validation on {src} done. Metrics: {metrics}")

        name = os.path.basename(src)
        if name in all_val_metrics:
            logger.warning(
                f"Duplicate source name {name}, path {src} in validation sources, renaming to {name}_1"
            )
            name = f"{name}_1"
        all_val_metrics[name] = metrics

    generator.max_gen_len = max_gen_len

    return all_val_metrics
