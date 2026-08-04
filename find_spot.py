"""Brute-force sweep to find the confidence-threshold sweet spot for the
hard code / code+explanation+quality switch (model.py's hard_conf_switch,
env-gated via SEMVUL_HARD_CONF_SWITCH / SEMVUL_HARD_CONF_THRESH): per
sample, confidence < threshold -> code alone; else -> code+explanation
fusion with the 22 quality features concatenated (plain ladder L3).

Presents an interactive menu (dataset, subset size, threshold grid, seeds,
epochs, ranking metric), then calls experiments/fusevul_ladder/train.py's
train_rung() once per (threshold, seed) -- the real training pipeline, not
a reimplementation -- and reports mean +/- std per threshold across seeds.

Why seeds matter here: a single-seed sweep on this codebase cannot tell a
real sweet spot from noise. Training isn't fully deterministic per seed
(no global cuDNN-determinism pin), and at smoke-test subset sizes seed-to-
seed variance alone can swing ROC/PR/F1 by double digits (verified this
session: an "all-code" control run scored WORSE than several intermediate
thresholds, which is only possible if per-run noise dominates the trend).
Use --seeds with 3+ values (or the menu prompt) before trusting a winner;
this script flags it when it can't tell either.

Usage:
    python find_spot.py
    python find_spot.py --dataset reveal --subset 300 --lo 5 --hi 95 --step 10 --seeds 1,3,7
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LADDER_DIR = ROOT / "experiments" / "fusevul_ladder"
for _p in (str(ROOT), str(LADDER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Identical 8-column text channel used across final_reveal_l3.ps1 /
# final_devign_l3.ps1 this session -- $Cols is the same for both datasets.
FIELDS = ("confidence,risky_operations,missing_checks,function_name,"
          "called_functions,risky_apis,risk_summary,purpose")
DATASETS = ["reveal", "devign"]
OUT_ROOT = ROOT / "experiments" / "runs" / "find_spot"
METRICS = ["roc", "pr", "acc", "f1"]


def _ask(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else str(default)


def _menu():
    print("find_spot.py -- brute-force sweet-spot search for the hard")
    print("confidence switch (code-only below threshold, code+explanation")
    print("+quality above it, on top of the plain ladder L3).\n")
    print("Datasets:")
    for i, name in enumerate(DATASETS, 1):
        print(f"  {i}) {name}")
    ds_idx = _ask("Pick dataset", "1")
    try:
        dataset = DATASETS[int(ds_idx) - 1]
    except (ValueError, IndexError):
        dataset = DATASETS[0]

    subset = int(_ask("Subset size (samples; 0 = full dataset, SLOW)", 300))
    lo = int(_ask("Threshold sweep start (confidence 0-100)", 5))
    hi = int(_ask("Threshold sweep end", 95))
    step = int(_ask("Threshold sweep step", 10))
    seeds_raw = _ask("Seeds, comma-separated (use 3+ to trust the winner)", "1")
    seeds = [int(s) for s in seeds_raw.split(",") if s.strip()]
    epochs = int(_ask("Epochs", 12))
    metric = _ask(f"Rank by ({'/'.join(METRICS)})", "pr").lower()
    if metric not in METRICS:
        metric = "pr"
    return dict(dataset=dataset, subset=subset or None, lo=lo, hi=hi,
                step=step, seeds=seeds, epochs=epochs, metric=metric)


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--subset", type=int, default=300,
                    help="0 = full dataset (slow, no smoke-test skip)")
    ap.add_argument("--lo", type=int, default=5)
    ap.add_argument("--hi", type=int, default=95)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--seeds", type=str, default="1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--metric", choices=METRICS, default="pr")
    args = ap.parse_args()
    if args.dataset is None:
        return _menu()
    return dict(dataset=args.dataset, subset=args.subset or None, lo=args.lo,
                hi=args.hi, step=args.step,
                seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
                epochs=args.epochs, metric=args.metric)


def _run_one(dataset, threshold, seed, subset, epochs, out_dir):
    os.environ["SEMVUL_QUAL_V2"] = "0"
    os.environ["SEMVUL_EXPL_FIELDS"] = FIELDS
    os.environ["SEMVUL_HARD_CONF_SWITCH"] = "1"
    os.environ["SEMVUL_HARD_CONF_THRESH"] = str(threshold)
    os.environ["SEMVUL_QUAL_GATE"] = "0"
    for var in ("SEMVUL_CONF_SWITCH", "SEMVUL_TEXT_ONLY", "SEMVUL_FROZEN",
                "SEMVUL_QUAL_RICH"):
        os.environ.pop(var, None)

    tag = f"{dataset}_L3" + ("_smoke" if subset else "")
    result_path = Path(out_dir) / f"semanticvul_{tag}.json"
    if result_path.exists():
        print(f"[skip] thr={threshold} seed={seed} already done", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    from train import train_rung  # experiments/fusevul_ladder/train.py
    train_rung(dataset, "L3", out_dir=str(out_dir), seed=seed, subset=subset,
               code_enc="codet5p", max_text=512, max_code=512,
               batch=2, grad_accum=16, epochs=epochs,
               focal_alpha=0.85, focal_gamma=2.0)
    return json.loads(result_path.read_text(encoding="utf-8"))


def _extract(payload):
    a, t = payload["argmax"], payload["tuned_on_tune"]
    return dict(roc=payload["val_roc_auc"], pr=payload["val_pr_auc"],
                acc=a["acc"], f1=a["f1"], tuned_acc=t["acc"], tuned_f1=t["f1"])


def _aggregate(rows, thresholds):
    agg = {}
    for thr in thresholds:
        vals = [m for (t, _s, m) in rows if t == thr]
        if not vals:
            continue
        agg[thr] = {
            key: (stats.mean(v[key] for v in vals),
                  stats.pstdev(v[key] for v in vals) if len(vals) > 1 else 0.0)
            for key in vals[0]
        }
    return agg


def main():
    cfg = _parse_args()
    dataset, subset, epochs, metric = (cfg["dataset"], cfg["subset"],
                                       cfg["epochs"], cfg["metric"])
    thresholds = list(range(cfg["lo"], cfg["hi"] + 1, cfg["step"]))
    seeds = cfg["seeds"]
    print(f"\n{dataset}: sweeping thresholds {thresholds} x seeds {seeds} "
          f"(subset={subset}, epochs={epochs}, ranking by {metric})\n", flush=True)

    out_root = OUT_ROOT / dataset
    rows = []  # [(threshold, seed, metrics_dict), ...]
    t0 = time.time()
    for thr in thresholds:
        for seed in seeds:
            out_dir = out_root / f"thr{thr}_s{seed}"
            payload = _run_one(dataset, thr, seed, subset, epochs, out_dir)
            m = _extract(payload)
            rows.append((thr, seed, m))
            print(f"[find_spot] thr={thr:3d} seed={seed} "
                  f"roc={m['roc']:.2f} pr={m['pr']:.2f} "
                  f"acc={m['acc']:.2f} f1={m['f1']:.2f} "
                  f"(tuned acc={m['tuned_acc']:.2f} f1={m['tuned_f1']:.2f})",
                  flush=True)
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    agg = _aggregate(rows, thresholds)
    if not agg:
        print("No results produced.")
        return

    print("\n===== summary (mean +/- std across seeds) =====")
    header = f"{'thr':>5} | {'roc':>13} | {'pr':>13} | {'acc':>13} | {'f1':>13}"
    print(header)
    print("-" * len(header))
    for thr in thresholds:
        if thr not in agg:
            continue
        a = agg[thr]
        print(f"{thr:5d} | {a['roc'][0]:6.2f}+/-{a['roc'][1]:5.2f} | "
              f"{a['pr'][0]:6.2f}+/-{a['pr'][1]:5.2f} | "
              f"{a['acc'][0]:6.2f}+/-{a['acc'][1]:5.2f} | "
              f"{a['f1'][0]:6.2f}+/-{a['f1'][1]:5.2f}")

    best_thr = max(agg, key=lambda t: agg[t][metric][0])
    best_mean, best_std = agg[best_thr][metric]
    metric_means = [a[metric][0] for a in agg.values()]
    spread = max(metric_means) - min(metric_means)
    print(f"\nBest threshold by mean {metric.upper()}: {best_thr} "
          f"({best_mean:.2f} +/- {best_std:.2f})")
    if len(seeds) == 1:
        print("NOTE: single seed per threshold -- this sweep cannot distinguish "
              "a real sweet spot from seed noise (training isn't fully "
              "deterministic per seed in this codebase). Rerun with "
              "--seeds 1,3,7 (or more) before trusting the winner.")
    elif best_std >= spread / 2:
        print("WARNING: this threshold's own seed-to-seed std is at least half "
              "the total spread across thresholds -- the 'best' pick is likely "
              "noise, not a real sweet spot. Use more seeds before trusting it.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = OUT_ROOT / f"{dataset}_sweep_results.json"
    out_json.write_text(json.dumps({
        "dataset": dataset, "subset": subset, "epochs": epochs,
        "thresholds": thresholds, "seeds": seeds, "metric": metric,
        "rows": [{"threshold": t, "seed": s, **m} for (t, s, m) in rows],
        "summary": {str(t): {k: v for k, (v, _sd) in a.items()}
                    for t, a in agg.items()},
        "best_threshold": best_thr,
    }, indent=2), encoding="utf-8")
    print(f"\nFull results written to {out_json}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
