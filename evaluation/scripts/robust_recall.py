#!/usr/bin/env python3
"""Canonical format-robust scorer for free-form RECALL tasks
(triviaqa / naturalqs_open / simpleqa / jeopardy).

ONE home for the robust-recall logic: the format-robust F1 behind the
RECALL macro (TriviaQA / NaturalQs / SimpleQA).
Analogous to bbh_flex: a conservative, homegrown robustness heuristic (NOT a
published metric) applied identically to every run, to stop strict OLMES f1 from
under-crediting correct-but-differently-formatted free-form answers.

Credit rule (robust_ok): a prediction scores 1.0 if strict f1 >= 0.5, OR any gold
answer appears as a *contiguous token subsequence* of the prediction (model gave the
answer + context), OR the prediction is a contiguous span of a gold answer (shorter
exact form), after normalizing lowercase / hyphen->space / drop articles+punct /
singular-plural stemming. Guards:
  - token-contiguous (not char-substring, not bag-of-words): 'pole' != 'polarity'
  - anti-hedge: a >=3-item list (commas/"and") cannot earn the gold-in-pred credit,
    so scattershot "A, B, C, D, E" guessing is not rewarded.
Validated on jeopardy (40/42 recoveries clean) and nq (hedge guard removes list gaming).
"""

import json, glob, re, string, os

RECALL_TOKENS = {
    "triviaqa": "triviaqa",
    "nq": "naturalqs_open",
    "simpleqa": "simpleqa",
    "jeopardy": "jeopardy",
}


def _norm(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def _toks(s):
    return _norm(s).split()


def _stem(t):
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in t]


def _contig(sub, seq):
    return (
        bool(sub)
        and len(sub) <= len(seq)
        and any(seq[i : i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))
    )


def _is_hedge_list(out):
    items = [i for i in re.split(r",|\band\b", out.lower()) if i.strip()]
    return len(items) >= 3


def robust_ok(out, labels):
    """True if the (strict-wrong) prediction should be credited under robust scoring."""
    p = _toks(out)
    ps = _stem(p)
    hedge = _is_hedge_list(out)
    for l in labels:
        g = _toks(l)
        if not g:
            continue
        gs = _stem(g)
        if _contig(p, g) or _contig(ps, gs):  # pred is a short exact span of gold
            return True
        if not hedge and (
            _contig(g, p) or _contig(gs, ps)
        ):  # gold inside pred (not a hedge list)
            return True
    return False


# ---- prediction-file resolvers (path logic centralised here) ----
def post_pred_files(B, run, token):
    """Post-train: flat sibling dirs under B.

    COMPLETE-SPLIT PREFERENCE (2026-08-17). NQ/SimpleQA were originally evaluated at
    limit=1000 and later re-run at full split into `<run>_recallfull` siblings, so the
    same task can appear twice. per_example() keys on native_id and lets later files
    overwrite, which USED to resolve correctly only because `_recallfull` happens to
    sort after the main dir -- an accident that a differently-named sibling would break.
    Resolve it explicitly instead: per task name, keep only the file with the most
    examples (the complete split). Never pools two protocols for one task.
    """
    cands = sorted(set(glob.glob(f"{B}/{run}*/task-*{token}*-predictions.jsonl")))
    best = {}
    for f in cands:
        name = re.sub(r"^task-\d+-", "", os.path.basename(f)).replace(
            "-predictions.jsonl", ""
        )
        try:
            n = sum(1 for line in open(f) if line.strip())
        except Exception:
            continue
        if name not in best or n > best[name][0]:
            best[name] = (n, f)
    return sorted(f for _n, f in best.values())


def mid_pred_files(run, token):
    """Midtrain/from-scratch: pick the largest step dir that has this task.
    Covers evals/<step>/{olmes_results,*_results} and checkpoints/evals/<step>/*_results.
    """
    cands = glob.glob(
        f"{run}/evals/*/*_results/task-*{token}*-predictions.jsonl"
    ) + glob.glob(
        f"{run}/checkpoints/evals/*/*_results/task-*{token}*-predictions.jsonl"
    )
    if not cands:
        return []

    def step(p):
        m = re.search(r"/evals/0*(\d+)/", p)
        return int(m.group(1)) if m else -1

    best = max(step(p) for p in cands)
    return sorted(p for p in cands if step(p) == best)


# ---- scoring over a set of prediction files ----
def per_example(pred_files):
    """Return list of per-example robust scores in [0,1] (1.0 if credited, else strict f1)."""
    scores = {}
    for fp in pred_files:
        for line in open(fp):
            d = json.loads(line)
            lab = d["label"]
            lab = lab if isinstance(lab, list) else [lab]
            out = d["model_output"][0]["continuation"].strip()
            f1 = d["metrics"].get("f1", 0)
            scores[d["native_id"]] = 1.0 if (f1 >= 0.5 or robust_ok(out, lab)) else f1
    return list(scores.values())


def strict_per_example(pred_files):
    s = {}
    for fp in pred_files:
        for line in open(fp):
            d = json.loads(line)
            s[d["native_id"]] = d["metrics"].get("f1", 0)
    return list(s.values())


def robust_mean_f1(pred_files):
    v = per_example(pred_files)
    return 100.0 * sum(v) / len(v) if v else None


def strict_mean_f1(pred_files):
    v = strict_per_example(pred_files)
    return 100.0 * sum(v) / len(v) if v else None
