"""Base-paper comparison table for the final_* ladder (the thesis headline).

For each dataset it reports, under TWO scoring protocols shown side by side, how
much the LLM-explanation modality contributes -- framed like FuSEVul's own RQ3
ablation so the two are directly comparable:

  PROTOCOLS
  * circular  -- best val-ACC epoch, argmax@0.5, scored on the SAME val set used
                 to pick the epoch. Matches FuSEVul's *selection* rule but is
                 mildly optimistic (selection set == report set). == the stored
                 base_paper_protocol.by_val_acc.
  * held-out  -- val is split into a SELECTION half and a disjoint REPORT half
                 (stratified by label, fixed split across seeds). The epoch is
                 picked on the SELECTION half; every metric is scored on the
                 held-out REPORT half. Non-circular (selection != report), the
                 structure FuSEVul uses. Recomputed from the saved per-epoch val
                 probabilities -- NO retraining, NO LLM calls. Still a slice of
                 VAL, not the paper's (unavailable) official TEST split, so the
                 absolute level is de-optimised but not row-comparable to theirs.

  BLOCKS (per dataset)
  * EXPLANATION MODALITY -- our L1->L2 gain under both protocols beside FuSEVul's
    own code-only->full ablation delta (paper Table 3), plus a paired two-sided
    t-test p-value (L2 vs L1, matched by seed id) per protocol. The gain is
    robust to the circularity (both rungs carry it), so circular ~ held-out here.
  * ABSOLUTE (x2) -- FuSEVul-full beside our per-rung mean+-std (n seeds shown in
    the column header), once per protocol, so the optimism gap is visible. Delta
    L2-FuSE flagged (*): ours=VAL, theirs=TEST.
  * PER-SEED SPREAD -- n/mean+-SD/median/min/max/95%CI across seeds, per protocol,
    for Acc/F1/Prec/Rec + ROC-AUC/PR-AUC/MCC/balanced-acc. --full-spread adds the
    var/range/IQR/CV% columns back for an exhaustive appendix version.

PROTOCOL NOTE. FuSEVul (Information Fusion 125 (2026) 103450, S4.2/4.5) selects
the best-val-accuracy epoch and reports on a held-out TEST split; only Acc/Prec/
Rec/F1 (no ROC/PR). This pipeline carves train'/tune/val only (NO test set exists
on disk -- the raw FuSEVul release ships train/val only), so "held-out" here is a
split of val, the closest non-circular analog attainable without their test rows.

Reads experiments/runs/<cache>/s<seed>/fusevul_ladder_<ds>_<rung>.json (+ _probs.npz),
<cache> = <prefix>_<ds>_<rung>_cache (default prefix: final). Read-only; no GPU.

Usage:
    python src/rqs/aggregate_seeds.py                 # interactive menu
    python src/rqs/aggregate_seeds.py reveal
    python src/rqs/aggregate_seeds.py both --detailed
    python src/rqs/aggregate_seeds.py devign --holdout-frac 0.5 --json out.json
"""
from __future__ import annotations
import os
import sys
import glob
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "experiments", "runs")

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score)
from scipy.stats import ttest_rel, wilcoxon

# Windows consoles default to cp1252; prefer UTF-8 but keep all output ASCII so
# it renders identically on any terminal (matches the '+-' / '====' house style).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

DATASETS = ("reveal", "devign")
HOLDOUT_SEED = 1337   # fixes the val select/report split -> identical rows across seeds/rungs

# FuSEVul reported numbers -- Information Fusion 125 (2026) 103450.
#   full      = Table 2 (published FuSEVul row).
#   code_only = Table 3 "Without Explanation" == the CodeT5+ fine-tuned baseline
#               (our L1 analog). full - code_only = the paper's own measured
#               explanation-modality contribution (our L1->L2 analog).
# All on the held-out TEST split, best-val-accuracy epoch. No ROC/PR reported.
PAPER = {
    "reveal": {"code_only": {"acc": 90.05, "f1": 38.58, "prec": 44.93, "rec": 33.80},
               "full":      {"acc": 91.68, "f1": 46.76, "prec": 57.24, "rec": 39.52}},
    "devign": {"code_only": {"acc": 58.96, "f1": 54.30, "prec": 52.60, "rec": 56.10},
               "full":      {"acc": 60.39, "f1": 55.91, "prec": 54.14, "rec": 57.79}},
}

PAPER_METRICS = ["acc", "f1", "prec", "rec"]              # have a FuSEVul comparison cell
EXTRA_METRICS = ["spec", "roc", "pr", "mcc", "balacc"]    # beyond-paper (blank cell)
ALL_METRICS = PAPER_METRICS + EXTRA_METRICS
LABEL = {"acc": "Accuracy", "f1": "F1", "prec": "Precision", "rec": "Sensitivity",
         "spec": "Specificity", "roc": "ROC-AUC", "pr": "PR-AUC", "mcc": "MCC",
         "balacc": "Bal.Acc"}
PROTOCOLS = ["circular", "heldout"]

# t(0.975, df) for a small-sample 95% CI half-width (df = n-1); normal past df=10.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


# ----- metric primitives ----------------------------------------------------
def _clf_metrics(y, p):
    """All 9 metrics at threshold 0.5 for probability vector p vs labels y.

    'rec' (sensitivity/TPR) and 'spec' (specificity/TNR) are the standard
    paired pair from diagnostic-classifier reporting: rec = recall_score on
    the positive class (label 1), spec = recall_score on the negative class
    (label 0) -- i.e. TN / (TN + FP)."""
    y = np.asarray(y, dtype=int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    pred = (p >= 0.5).astype(int)
    return {"acc": 100 * accuracy_score(y, pred),
            "f1": 100 * f1_score(y, pred, zero_division=0),
            "prec": 100 * precision_score(y, pred, zero_division=0),
            "rec": 100 * recall_score(y, pred, zero_division=0),
            "spec": 100 * recall_score(y, pred, pos_label=0, zero_division=0),
            "roc": 100 * roc_auc_score(y, p),
            "pr": 100 * average_precision_score(y, p),
            "mcc": 100 * matthews_corrcoef(y, pred),
            "balacc": 100 * balanced_accuracy_score(y, pred)}


def _holdout_masks(y, frac):
    """Stratified select/report split of val indices. `frac` -> report share.
    Deterministic (HOLDOUT_SEED) so every seed/rung reports on the same rows."""
    y = np.asarray(y, dtype=int).ravel()
    rng = np.random.default_rng(HOLDOUT_SEED)
    report = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        ci = np.where(y == cls)[0]
        rng.shuffle(ci)
        report[ci[:int(round(len(ci) * frac))]] = True
    return ~report, report   # select_mask, report_mask


def _dig(payload, dotted):
    cur = payload
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


def _seed_of(path):
    parent = os.path.basename(os.path.dirname(path))
    return int(parent[1:]) if parent.startswith("s") and parent[1:].isdigit() else None


def _find(ds, rung, cache_prefix):
    # Accept both the current "semanticvul_" tag and the historical
    # "fusevul_ladder_" tag (pre-rename L1/L2 runs on disk) -- per-seed dir, so a
    # dict keyed by seed dir dedupes if (implausibly) both exist for one seed.
    cache = f"{cache_prefix}_{ds}_{rung.lower()}_cache"
    found = {}
    for tag in ("semanticvul", "fusevul_ladder"):
        pat = os.path.join(RUNS, cache, "s*", f"{tag}_{ds}_{rung}.json")
        for f in glob.glob(pat):
            found[os.path.dirname(f)] = f
    return cache, sorted(found.values())


def _seed_values(json_path, frac):
    """{'circular': {metric: v}, 'heldout': {metric: v}} for one seed.

    circular: all 8 from val_prob_ba (best-val-acc epoch, full val) -- verified to
              reproduce the stored base_paper_protocol.by_val_acc argmax exactly;
              falls back to the json's argmax cells if the npz is absent.
    heldout:  pick the best-ACC epoch on the SELECTION half of val, score all 8 on
              the disjoint REPORT half. Needs val_probs_per_epoch + val_y.
    """
    with open(json_path, encoding="utf-8") as fh:
        p = json.load(fh)
    out = {"circular": {}, "heldout": {}}

    npz_path = json_path[:-5] + "_probs.npz"
    z = None
    if os.path.exists(npz_path):
        try:
            z = np.load(npz_path)
        except (OSError, ValueError):
            z = None

    # ---- circular ----
    if z is not None and "val_prob_ba" in z and "val_y" in z:
        y = z["val_y"]
        if len(np.unique(y)) >= 2:
            out["circular"] = _clf_metrics(y, z["val_prob_ba"])
    if not out["circular"]:                      # npz-less fallback (acc/f1/prec/rec only)
        for m in PAPER_METRICS:
            v = _dig(p, f"base_paper_protocol.by_val_acc.argmax.{m}")
            if v is not None:
                out["circular"][m] = v

    # ---- held-out ----
    if z is not None and "val_probs_per_epoch" in z and "val_y" in z:
        P, y = z["val_probs_per_epoch"], np.asarray(z["val_y"], dtype=int).ravel()
        if P.ndim == 2 and P.shape[1] == len(y) and len(np.unique(y)) >= 2:
            sel, rep = _holdout_masks(y, frac)
            if y[sel].sum() and y[rep].sum() and (~y[rep].astype(bool)).sum():
                accs = [accuracy_score(y[sel], (P[i][sel] >= 0.5))
                        for i in range(P.shape[0])]
                best = int(np.argmax(accs))
                out["heldout"] = _clf_metrics(y[rep], P[best][rep])
    return out


# ----- statistics -----------------------------------------------------------
def _stats(vals):
    a = np.asarray(vals, dtype=float)
    n = a.size
    if n == 0:
        return None
    mean = float(a.mean())
    if n == 1:
        z = float("nan")
        return {"n": 1, "mean": mean, "std": z, "var": z, "min": mean, "med": mean,
                "max": mean, "range": 0.0, "iqr": 0.0, "cv": 0.0, "ci": z}
    std = float(a.std(ddof=1))
    q1, q3 = float(np.percentile(a, 25)), float(np.percentile(a, 75))
    t = _T95.get(n - 1, 1.96)
    return {"n": n, "mean": mean, "std": std, "var": float(a.var(ddof=1)),
            "min": float(a.min()), "med": float(np.median(a)), "max": float(a.max()),
            "range": float(a.max() - a.min()), "iqr": q3 - q1,
            "cv": (std / mean * 100) if mean else float("nan"),
            "ci": float(t * std / np.sqrt(n))}


def aggregate(ds, rung, cache_prefix, expect, frac):
    cache, files = _find(ds, rung, cache_prefix)
    seeds = []
    # metric -> {seed: value}; keyed by seed id so L1/L2 values can be paired for
    # the significance test below. Falls back to the file path as key on the
    # (rare) unparseable seed-dir name, so _stats still sees every value.
    per = {pr: {m: {} for m in ALL_METRICS} for pr in PROTOCOLS}
    for f in files:
        s = _seed_of(f)
        try:
            vals = _seed_values(f, frac)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [warn] unreadable {f}: {e}")
            continue
        if s is not None:
            seeds.append(s)
        key = s if s is not None else f
        for pr in PROTOCOLS:
            for m, v in vals[pr].items():
                per[pr][m][key] = v
    seeds = sorted(seeds)
    stats = {pr: {m: _stats(list(per[pr][m].values())) for m in ALL_METRICS} for pr in PROTOCOLS}
    return {"cache": cache, "seeds": seeds, "n": len(seeds),
            "missing": [s for s in expect if s not in seeds], "stats": stats,
            "per_seed": per}


# ----- formatting helpers ---------------------------------------------------
def _num(x, w=7, d=2):
    if x is None or x != x:
        return "--".rjust(w)
    return f"{x:.{d}f}".rjust(w)


def _signed(x, w=8, d=2):
    if x is None or x != x:
        return "--".rjust(w)
    return f"{x:+.{d}f}".rjust(w)


def _mean_sd(st, w=13):
    if st is None:
        return "--".center(w)
    if st["n"] == 1 or st["std"] != st["std"]:
        return f"{st['mean']:.2f}".center(w)
    return f"{st['mean']:.2f}+-{st['std']:.2f}".center(w)


def _delta(results, proto, m):
    s1, s2 = results["L1"]["stats"][proto][m], results["L2"]["stats"][proto][m]
    return (s2["mean"] - s1["mean"]) if (s1 and s2) else None


def _paired_test(results, proto, m):
    """Paired L1->L2 significance for one metric/protocol, matched by seed id
    (not by list position -- a seed missing from one rung must not silently
    pair with the wrong seed's value). None if fewer than 2 seeds are common
    to both rungs. Reports both a paired t-test and Wilcoxon signed-rank
    (the latter needs >=1 non-zero difference; falls back to NaN otherwise)."""
    d1 = results.get("L1", {}).get("per_seed", {}).get(proto, {}).get(m, {})
    d2 = results.get("L2", {}).get("per_seed", {}).get(proto, {}).get(m, {})
    common = sorted((s for s in d1 if s in d2 and isinstance(s, int)))
    if len(common) < 2:
        return None
    a = np.array([d1[s] for s in common], dtype=float)
    b = np.array([d2[s] for s in common], dtype=float)
    diff = b - a
    t_p = w_p = float("nan")
    try:
        t_p = float(ttest_rel(b, a).pvalue)
    except (ValueError, ZeroDivisionError):
        pass
    if np.any(diff != 0):
        try:
            w_p = float(wilcoxon(b, a).pvalue)
        except ValueError:
            pass
    return {"n": len(common), "t_p": t_p, "w_p": w_p}


def _pval(x, w=8):
    if x is None or x != x:
        return "--".rjust(w)
    if x < 0.001:
        s = "<.001"
    elif x < 1:
        s = f"{x:.3f}"[1:]           # APA style: drop the leading 0
    else:
        s = "1.000"
    return s.rjust(w)


# ----- rendering ------------------------------------------------------------
def _absolute_block(title, results, present, proto, paper, has_l2):
    print(f"\nABSOLUTE -- {title}")
    cols = "".join(f"{r + '(n=' + str(results[r]['n']) + ')':>13}" for r in present)
    dcol = f"{'L2-FuSE*':>11}" if has_l2 and paper else ""
    print(f"  {'metric':<12}{'FuSEVul':>9}{cols}{dcol}")
    print("  " + "-" * (12 + 9 + 13 * len(present) + (11 if dcol else 0)))
    for m in ALL_METRICS:
        fuse = paper["full"][m] if (m in PAPER_METRICS and paper) else None
        fcell = _num(fuse, 9) if fuse is not None else "n/a".rjust(9)
        row = f"  {LABEL[m]:<12}{fcell}"
        for r in present:
            row += _mean_sd(results[r]["stats"][proto][m], 13)
        if dcol:
            s2 = results["L2"]["stats"][proto][m]
            d = (s2["mean"] - fuse) if (fuse is not None and s2) else None
            row += _signed(d, 11)
        print(row)


def render_dataset(ds, results, rungs, frac, full_spread=False):
    paper = PAPER.get(ds, {})
    present = [r for r in rungs if results[r]["n"] > 0]
    rep = int(round(frac * 100))
    line = "=" * 78
    print(f"\n{line}\n{ds.upper()}   |   best val-acc epoch, argmax@0.5   |   "
          f"held-out = {100 - rep}/{rep} val split\n{line}")
    print("rungs present:  " + "  |  ".join(
        f"{r} {results[r]['n']}/{results[r]['n'] + len(results[r]['missing'])}"
        for r in rungs))
    print("NOTE: 'x+-y' = mean +- sample standard deviation (ddof=1) across the n"
          " seeds named above (n=<seeds> in column headers below); per-metric n can"
          " be lower -- see PER-SEED SPREAD for the exact count behind each cell.")

    L1, L2 = ("L1" in present), ("L2" in present)

    # ---- Block A: explanation-modality gain, both protocols vs paper ----
    print("\nEXPLANATION MODALITY  --  gain L1->L2, ours (both protocols) vs"
          " FuSEVul's own ablation")
    if L1 and L2:
        print(f"  {'metric':<12}{'circular':>11}{'held-out':>11}{'FuSEVul':>11}"
              f"{'p(circ)':>9}{'p(held)':>9}")
        print("  " + "-" * 63)
        for m in ALL_METRICS:
            pap = (paper["full"][m] - paper["code_only"][m]) if (m in PAPER_METRICS and paper) else None
            pt_c, pt_h = _paired_test(results, "circular", m), _paired_test(results, "heldout", m)
            print(f"  {LABEL[m]:<12}{_signed(_delta(results,'circular',m),11)}"
                  f"{_signed(_delta(results,'heldout',m),11)}"
                  f"{(_signed(pap,11) if pap is not None else 'n/a'.rjust(11))}"
                  f"{_pval(pt_c['t_p'] if pt_c else None, 9)}"
                  f"{_pval(pt_h['t_p'] if pt_h else None, 9)}")
        print("  (p = paired two-sided t-test, L2 vs L1, matched by seed id;"
              " Wilcoxon signed-rank also computed -- see --json dump)")
    else:
        need = "L2" if L1 else "L1 and L2"
        print(f"  gain unavailable ({need} not in cache) -- no delta fabricated")

    # ---- Block B: two absolute blocks (circular, then held-out) ----
    _absolute_block("base-paper (circular: val-select + val-report)",
                    results, present, "circular", paper, L2)
    _absolute_block(f"held-out (non-circular: select {100-rep}%, report {rep}% of val)",
                    results, present, "heldout", paper, L2)

    # ---- Block C: per-seed spread, per protocol ----
    # Default is the compact reporting table (mean+-SD, median, min, max, 95% CI);
    # --full-spread additionally prints var/range/IQR/CV%, kept for appendix use.
    print("\nPER-SEED SPREAD  (across seeds | VAL)")
    if full_spread:
        hdr = (f"  {'rung':<5}{'metric':<12}{'n':>3}{'mean':>8}{'std':>7}{'var':>8}{'min':>8}"
               f"{'med':>8}{'max':>8}{'range':>8}{'IQR':>7}{'CV%':>7}{'95%CI':>8}")
    else:
        hdr = (f"  {'rung':<5}{'metric':<12}{'n':>3}{'mean+-SD':>14}{'median':>8}"
               f"{'min':>8}{'max':>8}{'95%CI':>8}")
    for proto in PROTOCOLS:
        rows = [(r, m) for r in present for m in ALL_METRICS
                if results[r]["stats"][proto][m] is not None]
        if not rows:
            continue
        print(f"  [{proto}]")
        print(hdr + "\n  " + "-" * (len(hdr) - 2))
        last = None
        for r, m in rows:
            st = results[r]["stats"][proto][m]
            tag = r if r != last else ""
            last = r
            if full_spread:
                print(f"  {tag:<5}{LABEL[m]:<12}{st['n']:>3}{_num(st['mean'],8)}"
                      f"{_num(st['std'],7)}{_num(st['var'],8)}{_num(st['min'],8)}"
                      f"{_num(st['med'],8)}{_num(st['max'],8)}{_num(st['range'],8)}"
                      f"{_num(st['iqr'],7)}{_num(st['cv'],7,1)}{_num(st['ci'],8)}")
            else:
                print(f"  {tag:<5}{LABEL[m]:<12}{st['n']:>3}{_mean_sd(st,14)}"
                      f"{_num(st['med'],8)}{_num(st['min'],8)}{_num(st['max'],8)}"
                      f"{_num(st['ci'],8)}")
    if not full_spread:
        print("  (var/range/IQR/CV% omitted here for readability -- pass"
              " --full-spread for the exhaustive appendix table)")

# ----- optional old multi-protocol table (--detailed) -----------------------
_DETAIL = [("ROC-AUC", "val_roc_auc"), ("PR-AUC", "val_pr_auc"),
           ("argmax@0.5 acc", "argmax.acc"), ("argmax@0.5 f1", "argmax.f1"),
           ("calib@0.5 acc", "calibrated_at_05.acc"), ("calib@0.5 f1", "calibrated_at_05.f1"),
           ("tuned acc", "tuned_on_tune.acc"), ("tuned f1", "tuned_on_tune.f1"),
           ("basepaper acc", "base_paper_protocol.by_val_acc.argmax.acc"),
           ("basepaper f1", "base_paper_protocol.by_val_acc.argmax.f1"),
           ("best_epoch", "best_epoch")]


def render_detailed(ds, rungs, cache_prefix):
    print(f"\n--- detailed multi-protocol table ({ds.upper()}) ---")
    per = {}
    for r in rungs:
        _cache, files = _find(ds, r, cache_prefix)
        payloads = []
        for f in files:
            try:
                payloads.append(json.load(open(f, encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        per[r] = payloads
    header = "metric".ljust(16) + "".join(rr.center(15) for rr in rungs)
    print(header + "\n" + "-" * len(header))
    for label, path in _DETAIL:
        row = label.ljust(16)
        for r in rungs:
            vals = [v for v in (_dig(p, path) for p in per[r]) if v is not None]
            if not vals:
                row += "--".center(15)
            else:
                mu = np.mean(vals)
                sd = np.std(vals, ddof=1) if len(vals) > 1 else float("nan")
                row += (f"{mu:.2f}" if sd != sd else f"{mu:.2f}+-{sd:.2f}").center(15)
        print(row)


# ----- CLI ------------------------------------------------------------------
def _choose_datasets(args):
    if args.dataset:
        arg = args.dataset.strip().lower()
        if arg in DATASETS:
            return [arg]
        if arg in ("both", "all"):
            return list(DATASETS)
        print(f"unknown dataset '{arg}' (use: reveal | devign | both)")
        sys.exit(2)
    if args.datasets:
        return args.datasets
    print("ladder aggregate -- select dataset:\n  1) reveal\n  2) devign\n  3) both")
    try:
        c = input("choice [1/2/3]: ").strip()
    except EOFError:
        return list(DATASETS)
    return {"1": ["reveal"], "2": ["devign"], "3": list(DATASETS)}.get(c, list(DATASETS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=None,
                    help="reveal | devign | both (omit for an interactive menu)")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--rungs", nargs="*", default=["L1", "L2", "L3"])
    ap.add_argument("--cache-prefix", default="final")
    ap.add_argument("--expect-seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--holdout-frac", type=float, default=0.5,
                    help="fraction of val used as the held-out REPORT half (rest"
                         " = selection half); default 0.5")
    ap.add_argument("--detailed", action="store_true",
                    help="also print the old multi-protocol table")
    ap.add_argument("--full-spread", action="store_true",
                    help="print the exhaustive PER-SEED SPREAD columns"
                         " (var/range/IQR/CV%%) instead of the compact default")
    ap.add_argument("--json", default=None, help="dump the aggregate stats to this path")
    args = ap.parse_args()

    dump = {}
    for ds in _choose_datasets(args):
        results = {r: aggregate(ds, r, args.cache_prefix, args.expect_seeds,
                                args.holdout_frac) for r in args.rungs}
        render_dataset(ds, results, args.rungs, args.holdout_frac, args.full_spread)
        if args.detailed:
            render_detailed(ds, args.rungs, args.cache_prefix)
        dump[ds] = {r: {k: results[r][k] for k in ("cache", "seeds", "n", "missing", "stats", "per_seed")}
                    for r in args.rungs}
        if "L1" in results and "L2" in results:
            dump[ds]["sig_L1_vs_L2"] = {proto: {m: _paired_test(results, proto, m) for m in ALL_METRICS}
                                        for proto in PROTOCOLS}

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=2)
        print(f"\n[wrote] {args.json}")


if __name__ == "__main__":
    main()
