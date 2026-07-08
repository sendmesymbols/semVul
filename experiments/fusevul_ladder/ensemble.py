"""Multi-seed ensemble for the component ladder (RQ4 lever).

Trains extra seeds of chosen rungs (resumable — a finished seed is skipped),
then averages val probabilities across all members and reports:
  - ensemble ROC/PR vs best single member,
  - argmax@0.5 (FuSEVul-comparable),
  - threshold tuned on the SHARED tune slice (honest, non-circular),
  - a joint sweep: does ANY threshold beat stated acc AND F1 simultaneously?

All members share split_seed=1337, so the tune slice is identical across seeds
and tune probabilities can be averaged for honest threshold selection. The
original single-seed run (seed 1337) is picked up automatically as a member.

  # train 3 extra seeds of L3 on both datasets (~6-9 h), then report:
  python experiments/fusevul_ladder/ensemble.py
  # report-only from existing prob files (seconds):
  python experiments/fusevul_ladder/ensemble.py --eval-only
  # include prob files copied from another machine:
  python experiments/fusevul_ladder/ensemble.py --eval-only --extra-dirs path\\to\\copied
"""
from __future__ import annotations
import os
import sys
import glob
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

RUNS = os.path.join(ROOT, "experiments", "runs")
SEED_DIR = os.path.join(RUNS, "seeds")
SPLIT_SEED = 1337
STATED = {"devign": {"acc": 60.39, "f1": 55.91},
          "reveal": {"acc": 91.68, "f1": 46.76}}


def member_files(dataset, rung, extra_dirs):
    pats = [os.path.join(RUNS, f"fusevul_ladder_{dataset}_{rung}_probs.npz"),
            os.path.join(SEED_DIR, "s*", f"fusevul_ladder_{dataset}_{rung}_probs.npz"),
            # enriched-data retrains (2026-07-08): runs/enriched (reveal 320-tok)
            # and runs/enriched512 (devign 512-tok), incl. per-seed subfolders
            # copied back from other machines (retrain_remote.py). Their tune
            # slices differ in size from the baseline members (cleaned train),
            # so they contribute to the val ensemble but are auto-skipped for
            # tune averaging.
            os.path.join(RUNS, "enriched*", "**",
                         f"fusevul_ladder_{dataset}_{rung}_probs.npz"),
            # laptop members consolidated from ladder_probs_laptop.zip (2026-07-08)
            os.path.join(RUNS, "laptop", f"fusevul_ladder_{dataset}_{rung}_probs.npz")]
    for d in extra_dirs:
        pats.append(os.path.join(d, f"fusevul_ladder_{dataset}_{rung}_probs.npz"))
        pats.append(os.path.join(d, "**", f"fusevul_ladder_{dataset}_{rung}_probs.npz"))
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))


def _metrics_at(thr, prob1, y):
    yh = (prob1 >= thr).astype(int)
    return dict(threshold=round(float(thr), 3),
                acc=100 * accuracy_score(y, yh),
                f1=100 * f1_score(y, yh, zero_division=0),
                prec=100 * precision_score(y, yh, zero_division=0),
                rec=100 * recall_score(y, yh, zero_division=0))


def _best_f1_thr(prob1, y):
    best, bs = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        s = f1_score(y, (prob1 >= t).astype(int), zero_division=0)
        if s > bs:
            bs, best = s, float(t)
    return best


def beat_both_sweep(prob1, y, sa, sf):
    """Return (found, best) where best maximizes the worst-side margin."""
    best = None
    for t in np.linspace(0.02, 0.98, 385):
        yh = (prob1 >= t).astype(int)
        a = 100 * accuracy_score(y, yh)
        s = 100 * f1_score(y, yh, zero_division=0)
        margin = min(a - sa, s - sf)
        if best is None or margin > best[0]:
            best = (margin, float(t), a, s)
    margin, t, a, s = best
    return margin > 0, dict(threshold=round(t, 3), acc=a, f1=s,
                            worst_margin=round(margin, 2))


def _best_both_thr(prob1, y, sa, sf):
    """Threshold maximizing the worst-side margin vs stated targets,
    computed on the TUNE slice only (non-circular)."""
    best = (None, -1e9)
    for t in np.linspace(0.02, 0.98, 385):
        yh = (prob1 >= t).astype(int)
        m = min(100 * accuracy_score(y, yh) - sa,
                100 * f1_score(y, yh, zero_division=0) - sf)
        if m > best[1]:
            best = (float(t), m)
    return best[0]


def evaluate(dataset, rungs, extra_dirs):
    """Ensemble all prob files for the given rung(s). Passing several rungs
    pools them into one ensemble (the RQ4 'ensemble' component: seeds x rungs)."""
    files = []
    for rung in rungs:
        files.extend(member_files(dataset, rung, extra_dirs))
    files = sorted(set(files))
    label = "+".join(rungs)
    if not files:
        print(f"[{dataset} {label}] no prob files found")
        return
    val_probs, tune_probs, val_y, tune_y, tune_idx = [], [], None, None, None
    rocs = []
    skipped_tune = 0
    for f in files:
        d = np.load(f)
        vy = d["val_y"]
        if val_y is None:
            val_y = vy
        elif len(vy) != len(val_y) or not np.array_equal(vy, val_y):
            print(f"  [skip member] val split mismatch: {f}")
            continue
        val_probs.append(d["val_prob"])
        rocs.append(100 * roc_auc_score(val_y, d["val_prob"]))
        # tune probs are only averageable if the tune slice is the same
        ok_tune = True
        if "tune_idx" in d.files:
            if tune_idx is None:
                tune_idx = d["tune_idx"]
            elif not np.array_equal(d["tune_idx"], tune_idx):
                ok_tune = False
        else:
            ty = d["tune_y"]
            if tune_y is None:
                tune_y = ty
            elif len(ty) != len(tune_y) or not np.array_equal(ty, tune_y):
                ok_tune = False
        if ok_tune:
            if tune_y is None:
                tune_y = d["tune_y"]
            tune_probs.append(d["tune_prob"])
        else:
            skipped_tune += 1

    n = len(val_probs)
    ens = np.mean(val_probs, axis=0)
    st = STATED[dataset]
    print(f"\n=== {dataset} {label} — ensemble of {n} member(s) "
          f"(member ROCs: {', '.join(f'{r:.2f}' for r in rocs)})")
    print(f"  ensemble ROC={100 * roc_auc_score(val_y, ens):.2f} "
          f"PR={100 * average_precision_score(val_y, ens):.2f} "
          f"(best single: {max(rocs):.2f})")
    a = _metrics_at(0.5, ens, val_y)
    print(f"  argmax@0.5:    acc={a['acc']:.2f} f1={a['f1']:.2f} "
          f"(stated acc {st['acc']}, f1 {st['f1']})")
    if tune_probs:
        tens = np.mean(tune_probs, axis=0)
        thr = _best_f1_thr(tens, tune_y)
        t = _metrics_at(thr, ens, val_y)
        print(f"  tuned-on-tune: thr={t['threshold']} acc={t['acc']:.2f} "
              f"f1={t['f1']:.2f} p={t['prec']:.2f} r={t['rec']:.2f} "
              f"({n - skipped_tune} aligned tune members)")
        thr_b = _best_both_thr(tens, tune_y, st["acc"], st["f1"])
        tb = _metrics_at(thr_b, ens, val_y)
        both = tb["acc"] > st["acc"] and tb["f1"] > st["f1"]
        print(f"  both-tuned:    thr={tb['threshold']} acc={tb['acc']:.2f} "
              f"f1={tb['f1']:.2f} p={tb['prec']:.2f} r={tb['rec']:.2f} "
              f"-> {'BEAT BOTH (threshold chosen on tune)' if both else 'short'}")
    found, bb = beat_both_sweep(ens, val_y, st["acc"], st["f1"])
    verdict = "BEAT BOTH" if found else "no single threshold beats both"
    print(f"  joint sweep:   thr={bb['threshold']} acc={bb['acc']:.2f} "
          f"f1={bb['f1']:.2f} worst-margin={bb['worst_margin']:+.2f} -> {verdict}")


def main():
    # Defaults are set so the BARE command:
    #     python experiments/fusevul_ladder/ensemble.py
    # reproduces the best result we have: an eval-only, pooled ensemble over
    # every saved L1/L2/L3 member (all seeds), on both datasets, no GPU.
    # Opt in to training extra seeds with --train; disable pooling with --no-pool.
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["reveal", "devign"])
    ap.add_argument("--rungs", nargs="*", default=["L1", "L2", "L3"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--fusion", default="self")
    ap.add_argument("--train", dest="train", action="store_true",
                    help="train the seed members first (default: eval only)")
    ap.add_argument("--extra-dirs", nargs="*", default=[],
                    help="extra dirs to scan for *_probs.npz (e.g. copied "
                         "from another machine)")
    ap.add_argument("--no-pool", dest="pool", action="store_false",
                    help="report each rung separately instead of one pooled "
                         "ensemble (pooling is the default)")
    ap.set_defaults(pool=True)
    args = ap.parse_args()

    if args.train:
        from train import train_rung
        for seed in args.seeds:
            out_dir = os.path.join(SEED_DIR, f"s{seed}")
            for ds in args.datasets:
                for rung in args.rungs:
                    done = os.path.join(out_dir, f"fusevul_ladder_{ds}_{rung}.json")
                    if os.path.exists(done):
                        print(f"[skip] {ds} {rung} seed={seed} already done", flush=True)
                        continue
                    print(f"\n===== {ds} {rung} seed={seed} =====", flush=True)
                    train_rung(ds, rung, epochs=args.epochs, batch=args.batch,
                               fusion=args.fusion, seed=seed,
                               split_seed=SPLIT_SEED, out_dir=out_dir)

    for ds in args.datasets:
        if args.pool:
            evaluate(ds, args.rungs, args.extra_dirs)
        else:
            for rung in args.rungs:
                evaluate(ds, [rung], args.extra_dirs)


if __name__ == "__main__":
    main()
