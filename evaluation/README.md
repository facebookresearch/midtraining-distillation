# Evaluation

This section contains information on how to reproduce the paper's evaluation protocol.

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
(so the eval gets a clean GPU) for async evaluation during training.

## 2. Run the canonical suite

`scripts/run_evals.sh` submits the paper's evaluation suite, one Slurm job per
reported category:

```bash
MODEL=${MIDTRAIN_ROOT}/switch_distill/checkpoints/0000028800/hf \
OUTPUT_DIR=${EVAL_ROOT}/switch_distill \
STAGE=mid bash evaluation/scripts/run_evals.sh

# inspect without submitting, pick a subset, or run inline:
DRY_RUN=1 ...            # print the plan only
TASK_GROUPS="REASONING FACTUAL_RECALL" ...
LOCAL=1 ...              # run in the current allocation instead of sbatch
```

| Group | Tasks | Mid-training alias | Post-training alias |
|---|---|---|---|
| `REASONING` (6) | GSM8K | `gsm8k::olmes` | `gsm8k::tulu` |
| | GSM-Symbolic | `gsm_symbolic::olmo3` | same |
| | GSM-Plus | `gsm_plus::none` | same |
| | BBH | `bbh:cot::olmes` | `bbh:cot::tulu` |
| | DROP | `drop::olmes` | `drop::llama3` |
| | MATH | `minerva_math_500::olmes` | `minerva_math_500::tulu` |
| `KNOWLEDGE` (6) | MMLU | `mmlu:mc::olmes` | `mmlu:mc::tulu` |
| | MMLU-Pro | `mmlu_pro:mc::none` | same |
| | ARC-C | `arc_challenge:mc::olmes` | same |
| | OpenBookQA | `openbookqa:mc::olmes` | same |
| | WinoGrande | `winogrande:mc::olmes` | same |
| | AGI-Eval | `agi_eval_english:1shot::olmes` | same |
| `FACTUAL_RECALL` (3) | TriviaQA / NaturalQs / SimpleQA | `triviaqa::olmes`, `naturalqs::olmes`, `simpleqa::no-judge-short-form` | same |
| `INSTRUCTION_FOLLOWING` (1) | IFEval | — (needs a chat template) | `ifeval::tulu` |


GSM-Symbolic and GSM-Plus exhaust host RAM at the default datasets cache setting, so `REASONING` is
submitted with a larger `--mem` and `DATASETS_IN_MEMORY_MAX_SIZE=0`; and olmes
must not inherit a training job's distributed environment, so the generated
command unsets the `MASTER_ADDR`/`RANK`/`SLURM_PROCID` family before running.
