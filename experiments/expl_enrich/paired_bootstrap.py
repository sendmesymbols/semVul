"""Paired bootstrap ROC-delta between two ladder members (baseline vs treated).

The honest A/B readout for the ReVeal treatment: dual_eval POOLS all
runs/enriched* members into one ensemble, which blends arms. This instead
compares two matched members (same rung + seed) on the SAME row-aligned val set
and bootstraps the paired ROC difference with a 95% CI.

  .venv/Scripts/python.exe experiments/expl_enrich/paired_bootstrap.py \
      --base-sub enriched_real --treat-sub enriched_real_tail_a85 \
      --ds reveal --rung L2 --seed 1337
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(ROOT, "experiments", "runs")


def paired_roc_delta(y, p_base, p_treat, n_boot=2000, seed=1337):
    y = np.asarray(y); p_base = np.asarray(p_base); p_treat = np.asarray(p_treat)
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, p_base)
    treat = roc_auc_score(y, p_treat)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = y[idx].sum()
        if s == 0 or s == n:            # need both classes for ROC
            continue
        deltas.append(roc_auc_score(y[idx], p_treat[idx])
                      - roc_auc_score(y[idx], p_base[idx]))
    lo, hi = (np.percentile(deltas, [2.5, 97.5]) if deltas else (0.0, 0.0))
    return dict(base=100 * base, treat=100 * treat,
                delta=100 * (treat - base),
                ci=[100 * float(lo), 100 * float(hi)], n_boot=len(deltas))


def _load(sub, ds, rung, seed):
    """sub = run subdir name directly under experiments/runs (e.g. enriched_real)."""
    p = os.path.join(RUNS, sub, f"s{seed}", f"fusevul_ladder_{ds}_{rung}_probs.npz")
    d = np.load(p)
    return d["val_prob"], d["val_y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-sub", default="enriched_real",
                    help="baseline run subdir under experiments/runs")
    ap.add_argument("--treat-sub", required=True,
                    help="treated run subdir, e.g. enriched_real_tail_a85")
    ap.add_argument("--ds", default="reveal")
    ap.add_argument("--rung", default="L2")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    pb, yb = _load(args.base_sub, args.ds, args.rung, args.seed)
    pt, yt = _load(args.treat_sub, args.ds, args.rung, args.seed)
    assert np.array_equal(yb, yt), "val labels/order differ -> not paired"
    r = paired_roc_delta(yb, pb, pt, n_boot=args.n_boot, seed=args.seed)
    print(f"{args.ds} {args.rung} s{args.seed}: base ROC={r['base']:.2f}  "
          f"treat ROC={r['treat']:.2f}  delta={r['delta']:+.2f}  "
          f"95% CI=[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]  (n_boot={r['n_boot']})")


if __name__ == "__main__":
    main()
