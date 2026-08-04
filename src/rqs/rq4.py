"""RQ4 evidence: what are the individual and combined effects of imbalance-aware
loss functions, validation-based threshold tuning, and multi-seed ensembling on
minority-class detection and the precision-recall trade-off?

Menu-driven (reveal | devign | both). Per dataset, four bundles, each a
controlled comparison read from the cached per-seed runs -- NO retraining:

  A) IMBALANCE-AWARE LOSS -- the loss each run used (focal + capped class weight
     vs plain cross-entropy), read straight from config.use_focal / alpha_pos /
     focal_gamma, with minority PR-AUC / Recall / Precision / F1 at the honest
     tuned operating point. If a counterfactual loss cache is supplied
     (--loss-baseline-prefix), the per-rung delta (loss on - loss off) is
     computed; otherwise the single arm is reported with a note on how to
     populate the other (train.py focal on/off toggle).

  B) VALIDATION-BASED THRESHOLD TUNING -- the precision-recall trade-off, tabular.
     Minority Recall / Precision / F1 at four operating points (argmax@0.5,
     calibrated@0.5, tuned-on-tune@t, tuned-on-val@t), mean+/-std over seeds, with
     the delta vs the naive @0.5 point. The tuned threshold is chosen on the
     stratified TUNE slice carved from TRAIN (non-circular) -- this is the
     "validation-based threshold selection procedure" RQ4 outputs.

  C) MULTI-SEED ENSEMBLING -- single-seed mean vs the probability-averaged
     ensemble over the seeds present, at @0.5 and at the tune-selected threshold
     (threshold chosen on the AVERAGED tune probs -> non-circular). Delta =
     ensemble - single. Requires >=2 seeds to be meaningful.

  D) COMBINED STRATEGY -- minority Recall / Precision / F1 for the strategy stack
     single@0.5 (naive) -> single+tuned -> ensemble@0.5 -> ensemble+tuned. The
     recommended imbalance-handling strategy = ensemble + validation-tuned
     threshold; its lift over the naive baseline is the RQ4 headline.

Reads the per-seed cache JSONs + *_probs.npz under
experiments/runs/<prefix>_<ds>_l{1,2,3}_cache/ (--cache-prefix, default 'final').
Read-only; no GPU; safe on a partial ladder (missing rungs/seeds are reported,
not fatal). Minority class = the positive (vulnerable) label throughout.

Usage:
    python src/rqs/rq4.py                       # interactive menu
    python src/rqs/rq4.py reveal
    python src/rqs/rq4.py both --rung L2
    python src/rqs/rq4.py reveal --loss-baseline-prefix noloss
"""
from __future__ import annotations

import os
import sys
import glob
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

from src.config import RUNS_DIR

DATASETS = ("reveal", "devign")
RUNGS = ("L1", "L2", "L3")

# Accept both the current "semanticvul_" tag and the historical
# "fusevul_ladder_" tag (pre-rename runs on disk), mirroring aggregate_seeds.py.
_CACHE_TAGS = ("semanticvul", "fusevul_ladder")


def _glob_tagged(cache_dir, ds, rung, suffix=".json"):
    files = set()
    for tag in _CACHE_TAGS:
        files.update(glob.glob(os.path.join(cache_dir, "s*", f"{tag}_{ds}_{rung}{suffix}")))
    return sorted(files)


# FuSEVul's stated numbers (the anchor). Reveal is ~91% negative, so minority
# (positive-class) Recall/Precision/F1 -- not accuracy -- are the meaningful axes.
FUSEVUL = {"devign": {"acc": 60.39, "f1": 55.91},
           "reveal": {"acc": 91.68, "f1": 46.76, "prec": 57.24, "rec": 39.52}}

# Operating points for the threshold-tuning bundle (@0.5 is the naive anchor).
THRESH_POINTS = [
    ("argmax@0.5",  "argmax"),
    ("calib@0.5",   "calibrated_at_05"),
    ("tuned/tune",  "tuned_on_tune"),
    ("tuned/val",   "tuned_on_val"),
]


# --------------------------------------------------------------------------- IO
def _dig(payload, dotted):
    cur = payload
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float, str, bool)) else None


def _cache(ds, rung, cache_prefix):
    return f"{cache_prefix}_{ds}_{rung.lower()}_cache"


def _rung_payloads(ds, rung, cache_prefix):
    """(cache_name, [payloads]) for <prefix>_<ds>_<rung>_cache across seeds."""
    cache = _cache(ds, rung, cache_prefix)
    payloads = []
    for f in _glob_tagged(os.path.join(RUNS_DIR, cache), ds, rung):
        try:
            with open(f, encoding="utf-8") as fh:
                payloads.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            pass
    return cache, payloads


def _rung_probs(ds, rung, cache_prefix):
    """[npz, ...] across seeds for one rung, or []. Each carries val_prob/val_y
    and (aligned) tune_prob/tune_y/tune_idx."""
    cache = _cache(ds, rung, cache_prefix)
    out = []
    for f in _glob_tagged(os.path.join(RUNS_DIR, cache), ds, rung, "_probs.npz"):
        try:
            d = np.load(f, allow_pickle=True)
        except (OSError, ValueError):
            continue
        if "val_prob" in d.files and "val_y" in d.files:
            out.append(d)
    return out


def _present_rungs(ds, cache_prefix):
    return [r for r in RUNGS if _rung_payloads(ds, r, cache_prefix)[1]]


# ---------------------------------------------------------------- small helpers
def _mean_std(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else float("nan"))


def _cell(mu, sd=None, width=13):
    if mu is None:
        return "--".center(width)
    if sd is None or sd != sd:         # None or NaN -> single value, no std
        return f"{mu:.2f}".center(width)
    return f"{mu:.2f}+-{sd:.2f}".center(width)


def _delta(width=9):
    return lambda d: (f"{d:+.2f}".center(width) if d is not None else "--".center(width))


def _pos_metrics(prob, y, thr):
    """Positive- (minority-) class acc/f1/prec/rec at a threshold, as percents."""
    yh = (np.asarray(prob) >= thr).astype(int)
    y = np.asarray(y).astype(int)
    return dict(acc=100 * accuracy_score(y, yh),
                f1=100 * f1_score(y, yh, zero_division=0),
                prec=100 * precision_score(y, yh, zero_division=0),
                rec=100 * recall_score(y, yh, zero_division=0))


def _best_f1_thr(prob, y):
    best, bs = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        s = f1_score(y, (np.asarray(prob) >= t).astype(int), zero_division=0)
        if s > bs:
            bs, best = s, float(t)
    return best


def _rung_tag(rung, top):
    if rung == top:
        return f"{rung} (SemanticVul)"
    return f"{rung} code-only" if rung == "L1" else rung


# ------------------------------------------------------------ A) imbalance loss
def _loss_desc(payload):
    """Human-readable loss config from a run's config block."""
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    if cfg.get("use_focal"):
        a = cfg.get("alpha_pos"); g = cfg.get("focal_gamma")
        a = f"{a:.2f}" if isinstance(a, (int, float)) else "?"
        g = f"{g:.1f}" if isinstance(g, (int, float)) else "?"
        return f"focal(alpha_pos={a} capped, gamma={g})"
    return "plain CE (unweighted)"


def imbalance_loss(ds, present, top, cache_prefix, loss_baseline_prefix):
    print(f"\n  -- A) Imbalance-aware loss: config + minority metrics at the tuned "
          f"point (mean over seeds) --")
    base_note = ""
    if loss_baseline_prefix:
        base_note = f"  [delta vs {loss_baseline_prefix}_* = loss-off arm]"
    print("    " + "rung".ljust(16) + "loss".ljust(34)
          + "PR-AUC".center(13) + "Rec".center(13) + "Prec".center(13)
          + "F1".center(13) + base_note)
    dl = _delta()
    for rung in present:
        _, payloads = _rung_payloads(ds, rung, cache_prefix)
        loss = _loss_desc(payloads[0]) if payloads else "?"
        pr, _ = _mean_std([_dig(p, "val_pr_auc") for p in payloads])
        rec, _ = _mean_std([_dig(p, "tuned_on_tune.rec") for p in payloads])
        prec, _ = _mean_std([_dig(p, "tuned_on_tune.prec") for p in payloads])
        f1, _ = _mean_std([_dig(p, "tuned_on_tune.f1") for p in payloads])
        row = ("    " + _rung_tag(rung, top).ljust(16) + loss.ljust(34)
               + _cell(pr, None) + _cell(rec, None) + _cell(prec, None)
               + _cell(f1, None))
        if loss_baseline_prefix:
            _, base_pl = _rung_payloads(ds, rung, loss_baseline_prefix)
            b_pr, _ = _mean_std([_dig(p, "val_pr_auc") for p in base_pl])
            b_f1, _ = _mean_std([_dig(p, "tuned_on_tune.f1") for p in base_pl])
            d_pr = (pr - b_pr) if (pr is not None and b_pr is not None) else None
            d_f1 = (f1 - b_f1) if (f1 is not None and b_f1 is not None) else None
            row += "  " + dl(d_pr).strip().center(10) + dl(d_f1).strip().center(10)
        print(row)
    if not loss_baseline_prefix:
        print("    (single loss arm cached. To ablate the loss lever, train the "
              "counterfactual\n     with train.py's focal on/off toggle into a "
              "separate cache and pass --loss-baseline-prefix.)")


# --------------------------------------------- B) validation-based threshold tuning
def threshold_tuning(ds, top, cache_prefix):
    _, payloads = _rung_payloads(ds, top, cache_prefix)
    if not payloads:
        return
    print(f"\n  -- B) Validation-based threshold tuning: precision-recall "
          f"trade-off ({top}, mean over {len(payloads)} seeds) --")
    print("    " + "operating point".ljust(15) + "thr".center(7)
          + "Recall".center(13) + "Precision".center(13) + "F1".center(13)
          + "dF1 vs @0.5".center(13))
    # @0.5 F1 baseline for the delta column.
    f1_base, _ = _mean_std([_dig(p, "argmax.f1") for p in payloads])
    for label, key in THRESH_POINTS:
        thr, _ = _mean_std([_dig(p, f"{key}.threshold") for p in payloads])
        rec, rsd = _mean_std([_dig(p, f"{key}.rec") for p in payloads])
        prec, psd = _mean_std([_dig(p, f"{key}.prec") for p in payloads])
        f1, fsd = _mean_std([_dig(p, f"{key}.f1") for p in payloads])
        d = (f"{f1 - f1_base:+.2f}" if (f1 is not None and f1_base is not None)
             else "--")
        thr_s = f"{thr:.2f}" if thr is not None else "--"
        print("    " + label.ljust(15) + thr_s.center(7)
              + _cell(rec, rsd) + _cell(prec, psd) + _cell(f1, fsd)
              + d.center(13))
    print("    (tuned/tune = threshold chosen on the TRAIN-carved tune slice, "
          "non-circular;\n     tuned/val = optimistic upper bound, threshold "
          "peeked on val.)")


# --------------------------------------------------- ensemble maths (bundles C/D)
def _ensemble(members):
    """(val_y, per_seed_val_probs, ens_val_prob, tune_y, ens_tune_prob).
    Members whose val split mismatches are dropped; tune probs are averaged only
    across seeds whose tune_idx aligns (so the tuned threshold stays honest)."""
    val_y = None
    vps, tps = [], []
    tune_y, tune_idx = None, None
    for d in members:
        vy = d["val_y"]
        if val_y is None:
            val_y = vy
        elif len(vy) != len(val_y) or not np.array_equal(vy, val_y):
            continue
        vps.append(d["val_prob"])
        if "tune_idx" in d.files:
            if tune_idx is None:
                tune_idx, tune_y = d["tune_idx"], d["tune_y"]
                tps.append(d["tune_prob"])
            elif np.array_equal(d["tune_idx"], tune_idx):
                tps.append(d["tune_prob"])
    if not vps:
        return None, None, None, None, None
    ens_val = np.mean(vps, axis=0)
    ens_tune = np.mean(tps, axis=0) if tps else None
    return val_y, vps, ens_val, tune_y, ens_tune


def _single_mean(vps, val_y, thr_fn):
    """Mean over seeds of per-seed minority metrics; thr_fn(vp) -> threshold
    (0.5 for the naive point, or each seed tuned on nothing here -> caller passes
    a per-seed tuned threshold via tune probs)."""
    rows = [_pos_metrics(vp, val_y, thr_fn(i)) for i, vp in enumerate(vps)]
    out = {}
    for m in ("acc", "f1", "prec", "rec"):
        out[m] = _mean_std([r[m] for r in rows])
    return out


def ensembling(ds, top, cache_prefix):
    members = _rung_probs(ds, top, cache_prefix)
    n = len(members)
    print(f"\n  -- C) Multi-seed ensembling ({top}, {n} seed member(s)) --")
    if n == 0:
        print("    [skip] no prob files.")
        return
    if n == 1:
        print("    [note] only 1 seed cached -> ensembling is a no-op; "
              "train more seeds to populate this lever.")
    val_y, vps, ens_val, tune_y, ens_tune = _ensemble(members)
    if val_y is None:
        print("    [skip] val splits did not align across members.")
        return
    # per-seed tuned thresholds (each seed tuned on its own tune probs).
    seed_thr = []
    for d in members:
        if "tune_prob" in d.files and "tune_y" in d.files:
            seed_thr.append(_best_f1_thr(d["tune_prob"], d["tune_y"]))
        else:
            seed_thr.append(0.5)
    ens_thr = _best_f1_thr(ens_tune, tune_y) if ens_tune is not None else 0.5

    single_05 = _single_mean(vps, val_y, lambda i: 0.5)
    single_tn = _single_mean(vps, val_y, lambda i: seed_thr[i])
    ens_05 = _pos_metrics(ens_val, val_y, 0.5)
    ens_tn = _pos_metrics(ens_val, val_y, ens_thr)

    roc_single = _mean_std([100 * roc_auc_score(val_y, vp) for vp in vps])
    pr_single = _mean_std([100 * average_precision_score(val_y, vp) for vp in vps])
    roc_ens = 100 * roc_auc_score(val_y, ens_val)
    pr_ens = 100 * average_precision_score(val_y, ens_val)
    print(f"    threshold-free:  ROC single {_cell(*roc_single, 0).strip()} "
          f"-> ensemble {roc_ens:.2f}  ({roc_ens - roc_single[0]:+.2f})  |  "
          f"PR single {_cell(*pr_single, 0).strip()} -> ensemble {pr_ens:.2f}  "
          f"({pr_ens - pr_single[0]:+.2f})")
    print("    " + "config".ljust(18) + "thr".center(7)
          + "Recall".center(13) + "Precision".center(13) + "F1".center(13))
    print("    " + f"single @0.5".ljust(18) + "0.50".center(7)
          + _cell(*single_05["rec"]) + _cell(*single_05["prec"])
          + _cell(*single_05["f1"]))
    print("    " + f"single tuned".ljust(18)
          + f"{np.mean(seed_thr):.2f}".center(7)
          + _cell(*single_tn["rec"]) + _cell(*single_tn["prec"])
          + _cell(*single_tn["f1"]))
    print("    " + f"ensemble @0.5".ljust(18) + "0.50".center(7)
          + _cell(ens_05["rec"], None) + _cell(ens_05["prec"], None)
          + _cell(ens_05["f1"], None))
    print("    " + f"ensemble tuned".ljust(18) + f"{ens_thr:.2f}".center(7)
          + _cell(ens_tn["rec"], None) + _cell(ens_tn["prec"], None)
          + _cell(ens_tn["f1"], None))


# ------------------------------------------------------- D) combined strategy
def combined_strategy(ds, top, cache_prefix):
    """The RQ4 headline: minority F1/Recall lift of the full imbalance strategy
    (ensemble + validation-tuned threshold) over the naive single-seed @0.5."""
    members = _rung_probs(ds, top, cache_prefix)
    if not members:
        return
    val_y, vps, ens_val, tune_y, ens_tune = _ensemble(members)
    if val_y is None:
        return
    ens_thr = _best_f1_thr(ens_tune, tune_y) if ens_tune is not None else 0.5
    naive = _single_mean(vps, val_y, lambda i: 0.5)          # single-seed @0.5
    full = _pos_metrics(ens_val, val_y, ens_thr)             # ensemble + tuned
    print(f"\n  -- D) Combined imbalance strategy ({top}): "
          f"naive single@0.5 -> ensemble+val-tuned (thr={ens_thr:.2f}) --")
    print("    " + "metric".ljust(12) + "naive @0.5".center(16)
          + "full strategy".center(16) + "lift".center(10))
    for m, name in (("rec", "Recall"), ("prec", "Precision"), ("f1", "F1")):
        nv = naive[m][0]
        fv = full[m]
        lift = (f"{fv - nv:+.2f}" if nv is not None else "--")
        print("    " + name.ljust(12) + _cell(*naive[m], 16)
              + _cell(fv, None, 16) + lift.center(10))
    fu = FUSEVUL[ds]
    if "f1" in fu:
        print(f"    (FuSEVul stated F1={fu['f1']}"
              + (f", Recall={fu['rec']}, Precision={fu['prec']}"
                 if "rec" in fu else "") + ")")


# ------------------------------------------------------------------------ driver
def run(ds, cache_prefix, pin_rung, loss_baseline_prefix):
    print("=" * 74)
    print(f"RQ4  |  {ds.upper()}   (FuSEVul stated: {FUSEVUL[ds]})   "
          f"[cache: {cache_prefix}_*]   minority = positive (vulnerable) class")
    print("=" * 74)
    present = _present_rungs(ds, cache_prefix)
    if not present:
        print(f"  [skip] no {cache_prefix}_* cache runs found for this dataset.\n")
        return
    if pin_rung and pin_rung in present:
        top = pin_rung
    elif pin_rung:
        print(f"  [warn] --rung {pin_rung} absent; using top present {present[-1]}")
        top = present[-1]
    else:
        top = present[-1]
    imbalance_loss(ds, present, top, cache_prefix, loss_baseline_prefix)
    threshold_tuning(ds, top, cache_prefix)
    ensembling(ds, top, cache_prefix)
    combined_strategy(ds, top, cache_prefix)
    print()


def _choose_datasets(args):
    if args.dataset:
        arg = args.dataset.strip().lower()
        if arg in DATASETS:
            return [arg]
        if arg in ("both", "all"):
            return list(DATASETS)
        print(f"unknown dataset '{arg}' (use: reveal | devign | both)")
        sys.exit(2)
    print("RQ4 evidence -- select dataset:")
    print("  1) reveal")
    print("  2) devign")
    print("  3) both")
    try:
        c = input("choice [1/2/3]: ").strip()
    except EOFError:
        print("(no input; defaulting to both)")
        return list(DATASETS)
    return {"1": ["reveal"], "2": ["devign"], "3": list(DATASETS)}.get(
        c, list(DATASETS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=None,
                    help="reveal | devign | both (omit for an interactive menu)")
    ap.add_argument("--rung", choices=list(RUNGS), default=None,
                    help="pin which rung is 'SemanticVul' (default: top present)")
    ap.add_argument("--cache-prefix", default="final",
                    help="cache family -> <prefix>_<ds>_<rung>_cache (default 'final')")
    ap.add_argument("--loss-baseline-prefix", default=None,
                    help="cache prefix of the loss-off counterfactual arm; when "
                         "given, bundle A prints the loss-lever delta")
    args = ap.parse_args()
    for ds in _choose_datasets(args):
        run(ds, args.cache_prefix, args.rung, args.loss_baseline_prefix)


if __name__ == "__main__":
    main()
