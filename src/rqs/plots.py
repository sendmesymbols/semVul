"""RQ figures for SemanticVul -- publication-quality static charts from the
cached per-seed probabilities (no re-training).

Menu-driven (reveal | devign | both). Reads the per-seed *_probs.npz under
experiments/runs/<prefix>_<ds>_l{1,2,3}_cache/ and writes PNGs (300 dpi,
OVERWRITTEN each run) to reports/plots/, each filename ending _RQ<n>.png.

Figures (filename suffix = the RQ it serves):
  <ds>_roc_RQ3.png              ROC per rung, mean curve + std band across seeds
  <ds>_pr_RQ3.png               PR  per rung, mean curve + std band + no-skill floor
  <ds>_seedspread_RQ3.png       box+points of per-seed Accuracy & F1 per rung
  <ds>_train_process_RQ3.png    val ROC-AUC vs epoch (the training process)
  <ds>_expl_contribution_RQ1.png  L1 vs L2 ROC+PR, explanation lift annotated
  <ds>_threshold_sweep_RQ4.png    precision/recall/F1 vs threshold (top rung)
  <ds>_imbalance_strategy_RQ4.png minority Rec/Prec/F1 across the strategy stack
                                  (single@0.5 -> single tuned -> ensemble@0.5
                                  -> ensemble tuned): loss-agnostic view of the
                                  threshold-tuning and ensembling levers

Style: SciencePlots ('science','no-latex','grid') -> journal look without a
LaTeX install. Uses sklearn for the curves. Read-only on the caches; no GPU.

Usage:
    python src/rqs/plots.py                 # interactive menu
    python src/rqs/plots.py reveal
    python src/rqs/plots.py both --cache-prefix final
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
import matplotlib
matplotlib.use("Agg")                       # headless: write files, no display
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa: F401  (registers the styles)
    plt.style.use(["science", "no-latex", "grid"])
except Exception:                            # SciencePlots missing -> plain mpl
    pass
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, roc_auc_score,
                             f1_score, precision_score, recall_score)

THRS = np.linspace(0.05, 0.95, 91)          # threshold-sweep grid (RQ4)

from src.config import RUNS_DIR

DATASETS = ("reveal", "devign")
RUNGS = ("L1", "L2", "L3")
OUTDIR = os.path.join(ROOT, "reports", "plots")

# Accept both the current "semanticvul_" tag and the historical
# "fusevul_ladder_" tag (pre-rename runs on disk), mirroring aggregate_seeds.py.
_CACHE_TAGS = ("semanticvul", "fusevul_ladder")


def _glob_tagged(cache_dir, ds, rung, suffix=".json"):
    files = set()
    for tag in _CACHE_TAGS:
        files.update(glob.glob(os.path.join(cache_dir, "s*", f"{tag}_{ds}_{rung}{suffix}")))
    return sorted(files)
# Colour-blind-safe, stable per rung.
COLORS = {"L1": "#4C72B0", "L2": "#DD8452", "L3": "#55A868"}
GRID = np.linspace(0.0, 1.0, 200)


def _tag(rung, top):
    if rung == top:
        return f"{rung} (SemanticVul)"
    return f"{rung} code-only" if rung == "L1" else rung


def _dig(payload, dotted):
    cur = payload
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


# ------------------------------------------------------------------- cache IO
def _seed_probs(ds, rung, cache_prefix):
    """[(val_prob, val_y, npz), ...] across seeds for one rung, or []."""
    cache = f"{cache_prefix}_{ds}_{rung.lower()}_cache"
    out = []
    for f in _glob_tagged(os.path.join(RUNS_DIR, cache), ds, rung, "_probs.npz"):
        try:
            d = np.load(f, allow_pickle=True)
        except (OSError, ValueError):
            continue
        if "val_prob" in d.files and "val_y" in d.files:
            out.append((d["val_prob"], d["val_y"], d))
    return out


def _seed_json_vals(ds, rung, cache_prefix, path):
    cache = f"{cache_prefix}_{ds}_{rung.lower()}_cache"
    vals = []
    for f in _glob_tagged(os.path.join(RUNS_DIR, cache), ds, rung):
        try:
            v = _dig(json.load(open(f, encoding="utf-8")), path)
        except (OSError, json.JSONDecodeError):
            continue
        if v is not None:
            vals.append(v)
    return vals


def _present(ds, cache_prefix):
    return [r for r in RUNGS if _seed_probs(ds, r, cache_prefix)]


# --------------------------------------------------------------- curve maths
def _roc_band(seed_probs):
    tprs, aucs = [], []
    for p, y, _ in seed_probs:
        fpr, tpr, _ = roc_curve(y, p)
        t = np.interp(GRID, fpr, tpr); t[0] = 0.0
        tprs.append(t); aucs.append(auc(fpr, tpr))
    tprs = np.asarray(tprs)
    return tprs.mean(0), tprs.std(0), float(np.mean(aucs)), float(np.std(aucs))


def _pr_band(seed_probs):
    precs, aps, prev = [], [], []
    for p, y, _ in seed_probs:
        pr, rc, _ = precision_recall_curve(y, p)
        order = np.argsort(rc)
        precs.append(np.interp(GRID, rc[order], pr[order]))
        aps.append(average_precision_score(y, p))
        prev.append(float(np.mean(y)))
    precs = np.asarray(precs)
    return (precs.mean(0), precs.std(0), float(np.mean(aps)),
            float(np.std(aps)), float(np.mean(prev)))


# ------------------------------------------------------------------ figures
def fig_roc(ds, cache_prefix, top):
    present = _present(ds, cache_prefix)
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    n = 0
    for rung in present:
        sp = _seed_probs(ds, rung, cache_prefix); n = len(sp)
        mu, sd, a_mu, a_sd = _roc_band(sp)
        c = COLORS[rung]
        ax.plot(GRID, mu, color=c, lw=1.4,
                label=f"{_tag(rung, top)}  AUC={a_mu:.3f}$\\pm${a_sd:.3f}")
        ax.fill_between(GRID, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                        color=c, alpha=0.18)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"ROC -- {ds} (mean$\\pm$std over {n} seeds)")
    ax.legend(loc="lower right", fontsize=6, frameon=True)
    return _save(fig, f"{ds}_roc_RQ3.png")


def fig_pr(ds, cache_prefix, top):
    present = _present(ds, cache_prefix)
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    n, prev = 0, None
    for rung in present:
        sp = _seed_probs(ds, rung, cache_prefix); n = len(sp)
        mu, sd, ap_mu, ap_sd, prev = _pr_band(sp)
        c = COLORS[rung]
        ax.plot(GRID, mu, color=c, lw=1.4,
                label=f"{_tag(rung, top)}  AP={ap_mu:.3f}$\\pm${ap_sd:.3f}")
        ax.fill_between(GRID, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                        color=c, alpha=0.18)
    if prev is not None:
        ax.axhline(prev, ls="--", color="grey", lw=0.8,
                   label=f"no-skill ({prev:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Precision-Recall -- {ds} (mean$\\pm$std over {n} seeds)")
    ax.legend(loc="upper right", fontsize=6, frameon=True)
    return _save(fig, f"{ds}_pr_RQ3.png")


def fig_seedspread(ds, cache_prefix, top):
    present = _present(ds, cache_prefix)
    if not present:
        return None
    rows = []
    for rung in present:
        for m_label, path in (("Accuracy", "base_paper_protocol.by_val_acc.argmax.acc"),
                              ("F1", "base_paper_protocol.by_val_acc.argmax.f1")):
            for v in _seed_json_vals(ds, rung, cache_prefix, path):
                rows.append({"rung": _tag(rung, top), "metric": m_label, "value": v})
    if not rows:
        return None
    import pandas as pd
    df = pd.DataFrame(rows)
    order = [_tag(r, top) for r in present]
    pal = {_tag(r, top): COLORS[r] for r in present}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    for ax, m in zip(axes, ("Accuracy", "F1")):
        sub = df[df.metric == m]
        sns.boxplot(data=sub, x="rung", y="value", order=order, ax=ax,
                    hue="rung", hue_order=order, palette=pal, legend=False,
                    width=0.5, fliersize=0)
        sns.stripplot(data=sub, x="rung", y="value", order=order, ax=ax,
                      color="black", size=3, jitter=0.12)
        ax.set_title(f"{m} across seeds"); ax.set_xlabel(""); ax.set_ylabel(m)
        ax.tick_params(axis="x", labelrotation=20, labelsize=6)
    fig.suptitle(f"Per-seed spread -- {ds} (paper-faithful protocol)", y=1.02)
    return _save(fig, f"{ds}_seedspread_RQ3.png")


def fig_train_process(ds, cache_prefix, top):
    present = _present(ds, cache_prefix)
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    plotted = False
    for rung in present:
        curves, eps = [], None
        for p, y, d in _seed_probs(ds, rung, cache_prefix):
            if "val_probs_per_epoch" not in d.files:
                continue
            pe = d["val_probs_per_epoch"]
            e = d["ep_index"] if "ep_index" in d.files else np.arange(pe.shape[0])
            aucs = [roc_auc_score(y, pe[i]) * 100 for i in range(pe.shape[0])]
            curves.append((np.asarray(e), np.asarray(aucs)))
        if not curves:
            continue
        plotted = True
        c = COLORS[rung]
        lens = {len(a) for a, _ in curves}
        if len(lens) == 1:                   # aligned -> mean+/-std band
            eps = curves[0][0]
            M = np.vstack([a for _, a in curves])
            ax.plot(eps, M.mean(0), color=c, lw=1.4, marker="o", ms=3,
                    label=f"{_tag(rung, top)}")
            ax.fill_between(eps, M.mean(0) - M.std(0), M.mean(0) + M.std(0),
                            color=c, alpha=0.18)
        else:                                # ragged -> thin per-seed lines
            for e, a in curves:
                ax.plot(e, a, color=c, lw=0.7, alpha=0.6)
            ax.plot([], [], color=c, lw=1.4, label=f"{_tag(rung, top)}")
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val ROC-AUC")
    ax.set_title(f"Training process -- {ds} (val ROC-AUC per epoch)")
    ax.legend(loc="lower right", fontsize=6, frameon=True)
    return _save(fig, f"{ds}_train_process_RQ3.png")


def fig_expl_contribution(ds, cache_prefix, top):
    """RQ1: does adding the explanation channel improve detection? L1 (code-only)
    vs L2 (+explanation), ROC and PR side by side with the AUC lift annotated."""
    l1, l2 = _seed_probs(ds, "L1", cache_prefix), _seed_probs(ds, "L2", cache_prefix)
    if not (l1 and l2):
        return None
    arms = [("L1", "code-only", l1), ("L2", "+explanation", l2)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for rung, label, sp in arms:                       # ROC panel
        mu, sd, a_mu, a_sd = _roc_band(sp)
        c = COLORS[rung]
        axes[0].plot(GRID, mu, color=c, lw=1.4, label=f"{label}  AUC={a_mu:.3f}")
        axes[0].fill_between(GRID, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                             color=c, alpha=0.18)
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1); axes[0].set_title("ROC")
    axes[0].legend(loc="lower right", fontsize=6, frameon=True)
    prev = None
    for rung, label, sp in arms:                       # PR panel
        mu, sd, ap_mu, ap_sd, prev = _pr_band(sp)
        c = COLORS[rung]
        axes[1].plot(GRID, mu, color=c, lw=1.4, label=f"{label}  AP={ap_mu:.3f}")
        axes[1].fill_between(GRID, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                             color=c, alpha=0.18)
    if prev is not None:
        axes[1].axhline(prev, ls="--", color="grey", lw=0.8,
                        label=f"no-skill ({prev:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1); axes[1].set_title("Precision-Recall")
    axes[1].legend(loc="upper right", fontsize=6, frameon=True)
    a1 = _roc_band(l1)[2]; a2 = _roc_band(l2)[2]
    fig.suptitle(f"Explanation contribution -- {ds}: +explanation lifts ROC-AUC "
                 f"{a1:.3f}$\\to${a2:.3f} ($\\Delta${a2-a1:+.3f})", y=1.03, fontsize=9)
    return _save(fig, f"{ds}_expl_contribution_RQ1.png")


def fig_threshold_sweep(ds, cache_prefix, top):
    """RQ4: precision / recall / F1 vs decision threshold for SemanticVul (top
    rung), mean over seeds, marking the val-tuned threshold and 0.5 -- the
    imbalance / threshold trade-off, pictorially."""
    sp = _seed_probs(ds, top, cache_prefix)
    if not sp:
        return None
    P, R, F = [], [], []
    for p, y, _ in sp:
        y = y.astype(int); npos = int(y.sum())
        pr, rc, f1 = [], [], []
        for t in THRS:
            yp = p >= t
            tp = int((yp & (y == 1)).sum()); fp = int((yp & (y == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / npos if npos else 0.0
            pr.append(prec); rc.append(rec)
            f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        P.append(pr); R.append(rc); F.append(f1)
    P, R, F = 100*np.array(P), 100*np.array(R), 100*np.array(F)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(THRS, P.mean(0), color="#4C72B0", lw=1.3, label="Precision")
    ax.plot(THRS, R.mean(0), color="#DD8452", lw=1.3, label="Recall")
    ax.plot(THRS, F.mean(0), color="#55A868", lw=1.7, label="F1")
    ax.fill_between(THRS, F.mean(0) - F.std(0), F.mean(0) + F.std(0),
                    color="#55A868", alpha=0.15)
    tuned = _seed_json_vals(ds, top, cache_prefix, "tuned_on_tune.threshold")
    if tuned:
        tt = float(np.mean(tuned))
        ax.axvline(tt, ls=":", color="black", lw=1.0, label=f"val-tuned thr={tt:.2f}")
    ax.axvline(0.5, ls="--", color="grey", lw=0.8, label="thr=0.5")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score (%)")
    ax.set_xlim(THRS[0], THRS[-1]); ax.set_ylim(0, 100)
    ax.set_title(f"Threshold sweep -- {ds} ({top}, mean over {len(sp)} seeds)")
    ax.legend(loc="center right", fontsize=6, frameon=True)
    return _save(fig, f"{ds}_threshold_sweep_RQ4.png")


# --------------------------------------------- RQ4 imbalance-strategy maths
def _ens_probs(sp):
    """(val_y, [val_probs], ens_val, tune_y, ens_tune) from _seed_probs output.
    Averages val probs across seeds; tune probs only across aligned tune_idx so
    the tuned threshold stays honest. Mirrors src/rqs/rq4.py::_ensemble."""
    val_y, vps, tps, tune_y, tune_idx = None, [], [], None, None
    for p, y, d in sp:
        if val_y is None:
            val_y = y
        elif len(y) != len(val_y) or not np.array_equal(y, val_y):
            continue
        vps.append(p)
        if "tune_prob" in d.files and "tune_y" in d.files:
            if "tune_idx" in d.files:
                ti = d["tune_idx"]
                if tune_idx is None:
                    tune_idx, tune_y = ti, d["tune_y"]; tps.append(d["tune_prob"])
                elif np.array_equal(ti, tune_idx):
                    tps.append(d["tune_prob"])
            elif tune_y is None:
                tune_y = d["tune_y"]; tps.append(d["tune_prob"])
    if not vps:
        return None, None, None, None, None
    ens_tune = np.mean(tps, axis=0) if tps else None
    return val_y, vps, np.mean(vps, axis=0), tune_y, ens_tune


def _best_f1_thr(prob, y):
    best, bs = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        s = f1_score(y, (np.asarray(prob) >= t).astype(int), zero_division=0)
        if s > bs:
            bs, best = s, float(t)
    return best


def _prf(prob, y, thr):
    yh = (np.asarray(prob) >= thr).astype(int)
    return (100 * recall_score(y, yh, zero_division=0),
            100 * precision_score(y, yh, zero_division=0),
            100 * f1_score(y, yh, zero_division=0))


def fig_imbalance_strategy(ds, cache_prefix, top):
    """RQ4: minority (positive-class) Recall / Precision / F1 across the imbalance
    strategy stack -- single@0.5 -> single tuned -> ensemble@0.5 -> ensemble tuned
    -- isolating the threshold-tuning and multi-seed-ensembling levers. Read from
    the cached per-seed probs; no retraining. Loss-agnostic (the loss each run
    used is a config fact, reported in rq4.py bundle A)."""
    sp = _seed_probs(ds, top, cache_prefix)
    if not sp:
        return None
    val_y, vps, ens_val, tune_y, ens_tune = _ens_probs(sp)
    if val_y is None:
        return None
    # per-seed tuned thresholds (each on its own tune probs); ensemble tuned on
    # the averaged tune probs -> non-circular.
    seed_thr = [_best_f1_thr(d["tune_prob"], d["tune_y"])
                if "tune_prob" in d.files else 0.5 for _, _, d in sp]
    ens_thr = _best_f1_thr(ens_tune, tune_y) if ens_tune is not None else 0.5

    def _single_mean(thr_of):
        vals = np.array([_prf(vp, val_y, thr_of(i)) for i, vp in enumerate(vps)])
        return vals.mean(0), (vals.std(0) if len(vps) > 1 else np.zeros(3))

    s05_mu, s05_sd = _single_mean(lambda i: 0.5)
    stn_mu, stn_sd = _single_mean(lambda i: seed_thr[i])
    e05 = np.array(_prf(ens_val, val_y, 0.5))
    etn = np.array(_prf(ens_val, val_y, ens_thr))

    configs = [f"single\n@0.5", f"single\ntuned\n({np.mean(seed_thr):.2f})",
               f"ensemble\n@0.5", f"ensemble\ntuned\n({ens_thr:.2f})"]
    mus = np.vstack([s05_mu, stn_mu, e05, etn])          # rows=config, cols=R/P/F1
    sds = np.vstack([s05_sd, stn_sd, np.zeros(3), np.zeros(3)])
    metrics = ["Recall", "Precision", "F1"]
    mcolors = ["#DD8452", "#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = np.arange(len(configs)); w = 0.26
    for j, (m, c) in enumerate(zip(metrics, mcolors)):
        ax.bar(x + (j - 1) * w, mus[:, j], w, yerr=sds[:, j], capsize=2,
               color=c, label=m, error_kw={"lw": 0.7})
    n = len(vps)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=6.5)
    ax.set_ylabel("Minority-class score (%)")
    ax.set_ylim(0, max(100, float(mus.max()) * 1.15))
    ax.set_title(f"Imbalance strategy -- {ds} ({top}, {n} seed(s); "
                 f"err=seed std)")
    ax.legend(loc="upper left", fontsize=6, frameon=True, ncol=3)
    return _save(fig, f"{ds}_imbalance_strategy_RQ4.png")


FIGURES = [fig_roc, fig_pr, fig_seedspread, fig_train_process,
           fig_expl_contribution, fig_threshold_sweep, fig_imbalance_strategy]


# --------------------------------------------------------------------- driver
def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def run(ds, cache_prefix):
    present = _present(ds, cache_prefix)
    print(f"[{ds}] rungs with probs: {present or 'NONE'}  [cache {cache_prefix}_*]")
    if not present:
        print(f"  [skip] no {cache_prefix}_{ds}_* probs found.")
        return
    top = present[-1]
    for fn in FIGURES:
        try:
            path = fn(ds, cache_prefix, top)
        except Exception as e:                # one bad figure never kills the run
            print(f"  [warn] {fn.__name__} failed: {e}")
            continue
        if path:
            print(f"  wrote {os.path.relpath(path, ROOT)}")


def _choose():
    p = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        p = sys.argv[1].strip().lower()
        if p in DATASETS:
            return [p]
        if p in ("both", "all"):
            return list(DATASETS)
        print(f"unknown dataset '{p}' (use: reveal | devign | both)")
        sys.exit(2)
    print("RQ plots -- select dataset:")
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
                    help="reveal | devign | both (omit for menu)")
    ap.add_argument("--cache-prefix", default="final",
                    help="cache family -> <prefix>_<ds>_<rung>_cache (default 'final')")
    args, _ = ap.parse_known_args()
    datasets = ([args.dataset] if args.dataset in DATASETS else
                list(DATASETS) if args.dataset in ("both", "all") else _choose())
    for ds in datasets:
        run(ds, args.cache_prefix)


if __name__ == "__main__":
    main()
