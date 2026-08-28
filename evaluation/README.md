# Evaluation

Everything needed to reproduce the paper's evaluation *protocol*: which OLMES
task alias, how many shots, which metric, and how tasks aggregate into the
reported REASON / MC / RECALL macros.

> **This directory ships code only.** No evaluation outputs, no prediction
> files, and no model weights are released. The scripts here read result trees
> produced by your own runs; they raise a clear `SystemExit` telling you which
> environment variable to set rather than silently producing an empty table.

## 1. Set up olmes

```bash
git clone https://github.com/allenai/olmes.git
cd olmes
git checkout c8194ffe7d6b2f7c694a3d5073d382dd01447f30    # the SHA the paper used
conda create -n olmes python=3.11 -y && conda activate olmes && pip install -e .
export OLMES_CONDA_ENV=$(conda info --base)/envs/olmes    # read by scripts/env.sh
```

Use a **separate** conda env: olmes pins different vLLM/transformers versions
than training does. `apps/main/eval_olmes.py` shells out to it as a subprocess
(so the eval gets a clean GPU); that is the only seam between training and
evaluation.

> **Local modifications are not distributed.** Our runs used a lightly patched
> olmes. Those patches are not part of this release, so reproducing the exact
> reported numbers from stock olmes requires re-creating them. The ones that
> materially affect scores are described below; §3 specifies the intended
> protocol in full.
>
> | Area | What our fork did |
> |---|---|
> | `configs/tasks.py` | Added `gsm_plus::none` and `gsm_symbolic::olmo3` task configs. **Neither task exists in stock olmes**, so those two REASON members cannot be run without re-adding them. |
> | `tasks/oe_eval_tasks/gsm8k.py` | The local answer extractor pre-truncated on `\n\n`, which destroyed verbose chat outputs that paragraph-break before "The answer is X". We removed `\n\n` from that list; without the fix, chat-formatted models are under-scored. |
> | `configs/tasks.py` | Set `minerva_math_500::tulu` `primary_metric` to `exact_match_flex`. The default `exact_match` is far too strict on chat-formatted LaTeX (collapsed to ~0.6-7.8%). |
> | `utilities/datasets_wrapper.py` | Offline fallback that loads `mandarjoshi/trivia_qa` from the local HF parquet cache when the Hub is unreachable. Convenience only; no score impact. |
> | `models/eleuther_vllm_causallms.py` | Coerce JSON string keys in `logit_bias` to ints (vllm >= 0.11 validates them as ints); stop resetting `CUDA_VISIBLE_DEVICES` to `0` when a data-parallel worker already pinned its GPU. Both are bug fixes, no score impact. |

## 2. Suite per stage

Two suites, selected by training stage — this matters, and mixing them makes
numbers non-comparable:

| Stage | Suite | Aliases |
|---|---|---|
| Base (stage-1), mid-training, from-scratch | base-style OLMES, no chat format, greedy | `::olmes` |
| SFT / DPO / RLVR1 / RLVR2 | Tulu-3 dev suite, chat format | `::tulu`, `::llama3` |

Six tasks have no `::tulu` variant (ARC-C, NaturalQs, TriviaQA, HellaSwag,
WinoGrande, MBPP) and use `::olmes` at every stage.

## 3. Task → alias → metric

| Task | Alias (mid / post) | Shots | Metric |
|---|---|---|---|
| GSM8K | `gsm8k::olmes` / `gsm8k::tulu` | 8 CoT | exact_match |
| GSM-Symbolic | `gsm_symbolic::olmo3` (both) | 8 CoT | exact_match, T=0.2 sampled, `max_gen_toks=1024` |
| GSM-Plus | `gsm_plus::none` (both) | 8 CoT | exact_match, greedy, n=10552 |
| BBH | `bbh:cot::olmes` / `bbh:cot::tulu` macro | 3 | **`bbh_flex`** — see the warning below |
| MATH | `minerva_math_500::olmes` / `::tulu` | 4 | `exact_match_flex` |
| DROP | `drop::olmes` / `drop::llama3` | 5 / 3 | F1 |
| MMLU | `mmlu:mc::olmes` / `mmlu:mc::tulu` | 0 | 57-subject macro |
| MMLU-Pro | `mmlu_pro:mc::none` | 0 | ~14-category macro, acc_raw |
| ARC-C / OpenBookQA / WinoGrande | `:mc::olmes` | 0 | acc (answer-selection, **not** `:rc`) |
| AGI-Eval | `agi_eval_english:1shot::olmes` | 1 | macro of 8 English subtasks |
| TriviaQA / NaturalQs / SimpleQA | `triviaqa::olmes`, `naturalqs::olmes`, `simpleqa::no-judge-short-form` | 5 | format-robust F1 (`robust_recall.py`) |
| IFEval | `ifeval::tulu` | 0 | **`inst_level_loose_acc`**, not olmes' default `prompt_level_loose_acc` |

### Reported macros

| Macro | Members |
|---|---|
| **REASON** (6) | GSM8K, GSM-Symbolic, GSM-Plus, BBH(flex), DROP, MATH |
| **MC** (6) | MMLU, MMLU-Pro, ARC-C, OpenBookQA, WinoGrande, AGI-Eval |
| **RECALL** (3) | TriviaQA, NaturalQs, SimpleQA — full splits (7993 / 3610 / 4321) |

A macro is reported **only when every member is present**. Partial macros are
coverage-flagged (`cov_*` in the CSV, trailing `*` in markdown) and are not
comparable to complete rows.

Dropped from the canonical macros, retained as display-only columns: DM-Math
(post-train-only — every registered variant requires a chat template and errors
on base checkpoints — and near-floor at 1B), HellaSwag (saturated), TruthfulQA
(mc2 tracks the *teacher's* truthfulness rather than student capability),
Jeopardy (our config is 5-shot where the standard task is 0-shot, so it isn't
comparable), PopQA.

### ⚠️ BBH is scored with a non-standard extractor

We report **`bbh_flex`**, our own format-robust re-extraction, **not** the
official OLMES scorer. This is a deliberate, disclosed deviation.

The shipped `bbh:cot::tulu` extractor only parses `"So the answer is X"`. Models
whose RL training pushed them toward `\boxed{}` or "the final answer is X" — i.e.
every RLVR-MATH and distilled run — get correct answers scored 0. That
under-counts BBH by ~8–11 pp for those runs but only ~2.5 pp for NTP (which
keeps the expected format), so it is a **model-dependent artifact that flips
the ranking**: uncorrected, it reads as "distillation hurts BBH", when
correcting it shows the opposite.

`bbh_flex` (in `scripts/bbh_robust_rescore.py`) parses `\boxed{}`, answer cues,
and last-option-token, then matches gold within the final-answer span. It is a
**homegrown heuristic we judge more appropriate for these models, not a
published or standard metric.** It was validated against NTP, where robust ≈
official (+2.5 pp), indicating it recovers genuine answers rather than
over-crediting. The uncorrected `bbh:cot::tulu` number is retained as a
provenance column. This is the same class of fix as MATH's `exact_match_flex`.

## 4. Scorers

This release ships the two **metric definitions**, not the table- and
figure-generation code. Both implement scoring that departs from the OLMES
defaults, so the paper's numbers cannot be interpreted without them:

| Script | Purpose |
|---|---|
| `robust_recall.py` | Format-robust free-form recall scoring (TriviaQA / NaturalQs / SimpleQA). Defines the F1 used for the RECALL macro. |
| `bbh_robust_rescore.py` | `bbh_flex` re-extraction for BBH — see the warning in §3. Writes `<run>_bbhflex/bbh_flex_summary.json` sidecars and prints official-vs-robust per subtask. |

Run `bbh_robust_rescore.py` over a results tree to reproduce the BBH column;
`robust_recall.py` is a library the recall scoring imports.

The table, significance-testing, and figure scripts we used internally are not
part of this release. §2 and §3 above specify the protocol completely enough to
recompute every reported number from OLMES output.

### A note on directory names

`bbh_robust_rescore.py` discovers runs by globbing `${RESULTS_ROOT}/posttrain`,
so it adapts to whatever your run directories are called. No evaluation outputs
ship with this repo.

## 5. Known gaps

- `bbh_robust_rescore.py` has a `SL3` mode targeting SmolLM runs that are
  outside the scope of this release.
- Reproducing AI2's published OLMo-2 1B numbers exactly was not always possible;
  the two largest residuals we could not close were GSM8K at mid-training
  (37.6 ours vs 43.8 official) and MMLU at SFT.
