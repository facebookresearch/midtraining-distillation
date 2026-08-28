"""Format-robust BBH re-scoring for bbh:cot::tulu predictions.

The official bbh:cot::tulu extractor only parses "So the answer is X"; models that
answer in \\boxed{}/"the final answer is X" format (RLVR-MATH/distilled runs) get
correct answers scored 0, understating BBH by ~8-11pp (NTP only ~2.5pp — the artifact
is model-dependent and creates false rankings). This re-extracts from the raw model
output with a format-tolerant parser and reports per-subtask + macro official vs robust.

Non-destructive: writes <run>_bbhflex/bbh_flex_summary.json + a combined CSV; does NOT
touch the canonical *_bbhcottulu metrics files.

  python bbh_robust_rescore.py            # re-scores every run found under
                                          # ${RESULTS_ROOT}/posttrain
Validate: NTP robust should be ~+2.5pp over official (extractor not over-crediting).
"""

import glob, json, re, csv, statistics as st, os, sys

# --- Release note -----------------------------------------------------------
# This repository ships code only: no evaluation outputs and no model weights.
# The roots below therefore have no defaults -- point them at your own runs.
# ----------------------------------------------------------------------------
import os as _os


def _env_root(var, hint):
    v = _os.environ.get(var)
    if not v:
        raise SystemExit(
            "%s is not set. This script reads %s.\n"
            "  export %s=/path/to/your/results\n"
            "See evaluation/README.md." % (var, hint, var)
        )
    return v


class _LazyRoot(str):
    """A path root that only errors if something actually interpolates it.

    ``robust_correct`` and friends are usable as plain functions by any
    caller; resolving roots at import time would make such imports fail for
    variables the caller never uses.
    """

    def __new__(cls, var, hint):
        obj = super().__new__(cls, "")
        obj._var, obj._hint = var, hint
        return obj

    def _resolve(self):
        return _env_root(self._var, self._hint)

    def __str__(self):
        return self._resolve()

    def __fspath__(self):
        return self._resolve()

    def __repr__(self):
        return self._resolve()


# Resolved lazily -- see _LazyRoot above.
RESULTS_ROOT = _LazyRoot("RESULTS_ROOT", "OLMES eval output directories")
WORK_ROOT = _LazyRoot("WORK_ROOT", "the root holding your runs")


# Anchored on RESULTS_ROOT rather than the cwd: this used to be the
# relative path "results/posttrain", which silently found nothing
# unless you happened to run from the olmes repo root.
P = _os.path.join(RESULTS_ROOT, "posttrain")
# (label, run_prefix)  -- resolved at RLVR2; edit/extend for project-wide rollout
# AUTO mode (default): re-score every RLVR2 run that has a *_bbhcottulu results dir.
# Pass explicit (label, run_prefix) pairs in RUNS to restrict to a subset instead.
RUNS = []  # empty => auto-discover
_CSV = "bbh_flex_rescore_all.csv"
# STAGE selects which post-train stage to auto-discover. Default rlvr2 (the reported
# endpoint); set STAGE=sft|dpo|rlvr1|rlvr2 to build bbh_flex sidecars for earlier stages
# so stage traces use the same format-robust extractor as the headline table.
_STAGE = os.environ.get("STAGE", "rlvr2")
_STAGE_GLOB = {"sft": "*_sft", "dpo": "*_dpo", "rlvr1": "*rlvr1*", "rlvr2": "*rlvr2*"}[
    _STAGE
]
if _STAGE != "rlvr2":
    _CSV = f"bbh_flex_rescore_all_{_STAGE}.csv"


def _is_cottulu(pred_file):
    """True if the sibling metrics file for this predictions file says cot::tulu."""
    mf = pred_file.replace("-predictions.jsonl", "-metrics.json")
    try:
        import json as _j

        alias = str(
            (_j.load(open(mf)).get("task_config") or {})
            .get("metadata", {})
            .get("alias")
            or ""
        )
    except Exception:
        return False
    return "cot::tulu" in alias


def discover():
    seen = {}
    for d in sorted(glob.glob(f"{P}/{_STAGE_GLOB}*bbhcottulu*")):
        pref = d.split("/")[-1].split("_bbhcottulu")[0]
        if pref in seen:
            continue
        if not glob.glob(f"{P}/{pref}*bbhcottulu*/*bbh_*predictions.jsonl"):
            continue
        seen[pref] = 1
        yield (pref, pref)  # label=prefix (concise labeling can be added later)
    # Runs evaluated by eval_single_posttrain.sh have NO _bbhcottulu sibling -- their
    # bbh:cot::tulu predictions sit in the MAIN results dir, so the glob above skipped
    # them and bbh_flex silently read as absent. Pick those up too, gated on the
    # task_config alias so cot::olmes / cot-v1::tulu predictions are never re-scored here.
    for d in sorted(glob.glob(f"{P}/{_STAGE_GLOB}")):
        pref = d.split("/")[-1]
        if pref in seen or "_bbhcottulu" in pref or pref.endswith("_bbhflex"):
            continue
        fs = glob.glob(f"{d}/*bbh_*predictions.jsonl")
        if not fs or not _is_cottulu(fs[0]):
            continue
        seen[pref] = 1
        yield (pref, pref)


def norm(s):
    s = str(s).lower()
    s = re.sub(r"\\boxed|\\text|\\mathrm|\\textbf|\$|\*", "", s)
    s = s.replace("{", "").replace("}", "").replace("(", "").replace(")", "")
    s = re.sub(r"[^a-z0-9\- ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract(out):
    o = str(out)
    b = re.findall(r"\\boxed\{([^}]*)\}", o)
    if b:
        return b[-1]
    cues = list(
        re.finditer(r"(?:final answer is|the answer is|answer is|answer:)\s*", o, re.I)
    )
    if cues:
        return re.split(r"[\.\n]", o[cues[-1].end() :])[0]
    lines = [l for l in o.strip().splitlines() if l.strip()]
    return lines[-1] if lines else o


def robust_correct(gold, out):
    g, c = norm(gold), norm(extract(out))
    if not g:
        return 0
    if g == c:
        return 1
    return 1 if re.search(r"(^| )" + re.escape(g) + r"( |$)", c) else 0


def score_run(prefix):
    res = {}
    _fs = glob.glob(f"{P}/{prefix}*bbhcottulu*/*bbh_*predictions.jsonl")
    if not _fs:  # main-dir case (no _bbhcottulu sibling); alias-gated
        _fs = [
            f
            for f in glob.glob(f"{P}/{prefix}/*bbh_*predictions.jsonl")
            if _is_cottulu(f)
        ]
    for f in _fs:
        sub = (
            f.split("/")[-1]
            .split("-", 2)[-1]
            .replace("-predictions.jsonl", "")
            .split(":")[0]
            .replace("bbh_", "")
        )
        off, rob = [], []
        for ln in open(f):
            r = json.loads(ln)
            gold = r.get("label")
            mo = r.get("model_output")
            o = mo[0] if isinstance(mo, list) else mo
            out = (o.get("continuation") if isinstance(o, dict) else str(o)) or ""
            sc = (r.get("metrics") or {}).get(
                "exact_match", (r.get("metrics") or {}).get("primary_score", 0)
            )
            off.append(sc)
            rob.append(robust_correct(gold, out))
        if off:
            res[sub] = (sum(off) / len(off) * 100, sum(rob) / len(rob) * 100)
    return res


# ---------------------------------------------------------------------------
# TRAJECTORY MODE (TRAJ=1): re-score BBH for the MID-TRAINING checkpoint sweep.
#
# The default mode walks results/posttrain and gates on a cot::tulu alias. Mid-training
# checkpoints are scored with bbh:cot::olmes instead, and their predictions live under
# <run>/evals/<step>/<results-dir>/, so neither the STAGE glob nor _is_cottulu applies.
# The extractor itself is alias-agnostic -- it reads raw model output -- so the same
# robust_correct() is the right scorer for both; only the discovery differs.
#
# Writes <run>/evals/<step>/bbh_flex_summary.json (same schema as the post-train sidecar)
# so plot_trajectory.py can read bbh_flex without re-parsing predictions on every render.
#   TRAJ=1 python bbh_robust_rescore.py
def _traj_runs():
    """TRAJ-mode run dirs. Built on demand so importing this module as a
    library does not require WORK_ROOT to be set."""
    b = str(WORK_ROOT)
    return {
        "NTP-only": f"{b}/dolmino_expert_runs/baseline-28800",
        "FKD (7B teacher)": f"{b}/dolmino2_runs_v3/dolmino_midtrain_olmo2_kd_rl7b_28800steps",
        "SwitchDist (q=0.20)": f"{b}/dolmino2_runs_v3/dolmino_midtrain_olmo2_rkl_entswitch_q20_lam1p0_28800steps",
    }


def score_pred_files(fs):
    """{subtask: (official_pct, robust_pct)} from a list of BBH predictions files."""
    res = {}
    for f in fs:
        sub = (
            f.split("/")[-1]
            .split("-", 2)[-1]
            .replace("-predictions.jsonl", "")
            .split(":")[0]
            .replace("bbh_", "")
        )
        off, rob = [], []
        for ln in open(f):
            r = json.loads(ln)
            gold = r.get("label")
            mo = r.get("model_output")
            o = mo[0] if isinstance(mo, list) else mo
            out = (o.get("continuation") if isinstance(o, dict) else str(o)) or ""
            sc = (r.get("metrics") or {}).get(
                "exact_match", (r.get("metrics") or {}).get("primary_score", 0)
            )
            off.append(sc)
            rob.append(robust_correct(gold, out))
        if off:
            res[sub] = (sum(off) / len(off) * 100, sum(rob) / len(rob) * 100, len(rob))
    return res


# The mid-training INITIALISATION: the pretrained 1B base every arm starts from. It is a flat
# results dir, not a <run>/evals/<step>/ tree, so it needs its own discovery -- but its BBH is
# cot::olmes like the checkpoints, so the same scorer applies and the point is protocol-matched.
def _traj_student_init():
    return f"{RESULTS_ROOT}/" "allenai_OLMo-2-0425-1B_stage1-step1907359-tokens4001B"


def _write_bbh_sidecar(fs, outf):
    res = score_pred_files(fs)
    if not res:
        return None
    off = sum(v[0] for v in res.values()) / len(res)
    rob = sum(v[1] for v in res.values()) / len(res)
    _se = []
    for _o, _r, _n in res.values():
        _pp = _r / 100.0
        _se.append((_pp * (1 - _pp) / _n) ** 0.5 * 100 if _n else 0.0)
    json.dump(
        {
            "n_subtasks": len(res),
            "macro_official": off,
            "macro_robust": rob,
            "macro_robust_se": (sum(e * e for e in _se) ** 0.5) / len(_se),
            "per_subtask": {
                k: {"official": v[0], "robust": v[1], "n": v[2]} for k, v in res.items()
            },
        },
        open(outf, "w"),
        indent=1,
    )
    return off, rob, len(res)


def traj_main():
    for lab, run in _traj_runs().items():
        steps = sorted(
            {
                p.split("/evals/")[1].split("/")[0]
                for p in glob.glob(f"{run}/evals/*/*/task-*-bbh_*-predictions.jsonl")
            }
        )
        print(f"\n{lab}: {len(steps)} steps")
        for st_ in steps:
            outf = f"{run}/evals/{st_}/bbh_flex_summary.json"
            if os.path.exists(outf) and not os.environ.get("FORCE"):
                continue
            # exclude quarantined dirs -- they hold known-truncated generations
            fs = [
                f
                for f in glob.glob(
                    f"{run}/evals/{st_}/*/task-*-bbh_*-predictions.jsonl"
                )
                if "_quarantine" not in f
            ]
            if not fs:
                continue
            res = score_pred_files(fs)
            if not res:
                continue
            off = sum(v[0] for v in res.values()) / len(res)
            rob = sum(v[1] for v in res.values()) / len(res)
            # SE of the 27-subtask macro. Each subtask is a mean of Bernoulli draws, so its
            # SE is sqrt(p(1-p)/n); the macro is their unweighted mean, hence sqrt(sum)/k.
            # This is EVAL-SAMPLING noise only -- it says nothing about checkpoint-to-
            # checkpoint variation, which is the larger term (see plot_trajectory.py).
            _se = []
            for _o, _r, _n in res.values():
                _pp = _r / 100.0
                _se.append((_pp * (1 - _pp) / _n) ** 0.5 * 100 if _n else 0.0)
            macro_se = (sum(e * e for e in _se) ** 0.5) / len(_se)
            json.dump(
                {
                    "n_subtasks": len(res),
                    "macro_official": off,
                    "macro_robust": rob,
                    "macro_robust_se": macro_se,
                    "per_subtask": {
                        k: {"official": v[0], "robust": v[1], "n": v[2]}
                        for k, v in res.items()
                    },
                },
                open(outf, "w"),
                indent=1,
            )
            print(
                f"   {st_}  n={len(res):2d}  official {off:5.2f} -> robust {rob:5.2f}"
            )
    # student init (x=0 on the trajectory)
    _init = _traj_student_init()
    _o = f"{_init}/bbh_flex_summary.json"
    if not os.path.exists(_o) or os.environ.get("FORCE"):
        _fs = [
            f
            for f in glob.glob(
                f"{_init}/**/task-*-bbh_*-predictions.jsonl", recursive=True
            )
            if "_quarantine" not in f
        ]
        r = _write_bbh_sidecar(_fs, _o) if _fs else None
        if r:
            print(
                f"\nstudent init: n={r[2]}  official {r[0]:5.2f} -> robust {r[1]:5.2f}"
            )


def main():
    runs = RUNS if RUNS else list(discover())
    data = {lab: score_run(pref) for lab, pref in runs}
    rows = []
    for lab, pref in runs:
        d = data[lab]
        macro_off = st.mean(v[0] for v in d.values())
        macro_rob = st.mean(v[1] for v in d.values())
        outdir = f"{P}/{pref}_bbhflex"
        os.makedirs(outdir, exist_ok=True)
        json.dump(
            {
                "macro_official": macro_off,
                "macro_robust": macro_rob,
                "n_subtasks": len(d),
                "per_subtask": d,
            },
            open(f"{outdir}/bbh_flex_summary.json", "w"),
            indent=2,
        )
        rows.append((lab, macro_off, macro_rob, macro_rob - macro_off, len(d)))
    with open(globals().get("_CSV", "bbh_flex_rescore.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "bbh_official", "bbh_robust", "artifact", "n_subtasks"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.1f}", f"{r[2]:.1f}", f"{r[3]:+.1f}", r[4]])
    rows.sort(key=lambda r: -r[2])
    print(f"{'run':52}{'official':>9}{'robust':>8}{'artifact':>10}")
    for lab, off, rob, art, n in rows:
        print(f"{lab[:52]:52}{off:9.1f}{rob:8.1f}{art:+10.1f}")
    ntp = [r for r in rows if r[0].startswith("ntp")]
    print(
        f"\nwrote {globals().get('_CSV')} + per-run <run>_bbhflex/bbh_flex_summary.json  ({len(rows)} runs)"
    )
    if ntp:
        print(f"validation: NTP artifact = {ntp[0][3]:+.1f}pp (should be small, ~+2.5)")


if __name__ == "__main__":
    if os.environ.get("TRAJ"):
        traj_main()
    else:
        main()
