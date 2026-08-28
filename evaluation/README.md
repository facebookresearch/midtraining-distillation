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
