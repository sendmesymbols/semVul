"""Remote-GPU worker: train extra seeds of the treatment configs.

Meant for a second machine. The laptop (2026-07-08) is running seed 1337 of:
  reveal L2/L3 enriched+clean      -> runs/enriched/
  devign L1/L2/L3 enriched 512-tok -> runs/enriched512/  (queued)
This script trains OTHER seeds of the same configs, so every run anywhere is a
new ensemble member. Results land in runs/enriched512/s<seed>/ and
runs/enriched/s<seed>/ — zip those folders and drop them into the same path on
the main machine; ensemble.py / dual_eval.py auto-scan them.

Prereqs on this machine (deterministic, CPU, ~2 min total) if the derived
JSONLs are not already present:
  python experiments/expl_enrich/run_enrich.py
  python experiments/expl_enrich/augment_train.py --variant enriched --aug-copies 1

Usage (defaults = the highest-value pending work):
  python experiments/expl_enrich/retrain_remote.py --seeds 2024 2025 2026
  # devign-512 only / reveal only:
  python experiments/expl_enrich/retrain_remote.py --only devign --seeds 2024 2025
  # bigger GPU (>=16GB): --batch512 4 roughly halves devign wall-clock
"""
from __future__ import annotations
import argparse
import os
import sys

os.environ["SEMVUL_EXPL_VARIANT"] = "enriched"
os.environ["SEMVUL_QUAL_V2"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LADDER = os.path.join(ROOT, "experiments", "fusevul_ladder")
for _p in (ROOT, LADDER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNS = os.path.join(ROOT, "experiments", "runs")
SPLIT_SEED = 1337  # shared tune carve -> tune probs averageable across seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--only", choices=["devign", "reveal"], default=None)
    ap.add_argument("--batch512", type=int, default=2,
                    help="batch for the 512-token devign runs (2 fits 8GB; "
                         "4 on >=16GB)")
    ap.add_argument("--rungs", nargs="*", default=None,
                    help="override rung list, e.g. --rungs L1 L3")
    args = ap.parse_args()

    from train import train_rung  # after sys.path setup

    # (dataset, rungs, out_subdir, train_suffix, kwargs) — devign first: it is
    # the dataset where extra 512-token members have the highest expected value.
    jobs = [
        ("devign", args.rungs or ["L1", "L2", "L3"], "enriched512", "clean.aug",
         dict(max_code=512, batch=args.batch512,
              grad_accum=max(1, 32 // args.batch512))),
        ("reveal", args.rungs or ["L2", "L3", "L1"], "enriched", "clean",
         dict()),
    ]
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]

    for seed in args.seeds:
        for ds, rungs, sub, suffix, kw in jobs:
            os.environ["SEMVUL_TRAIN_SUFFIX"] = suffix
            out_dir = os.path.join(RUNS, sub, f"s{seed}")
            for rung in rungs:
                done = os.path.join(out_dir, f"fusevul_ladder_{ds}_{rung}.json")
                if os.path.exists(done):
                    print(f"[skip] {ds} {rung} s{seed} done", flush=True)
                    continue
                print(f"\n===== {ds} {rung} seed={seed} ({sub}) =====", flush=True)
                train_rung(ds, rung, out_dir=out_dir, seed=seed,
                           split_seed=SPLIT_SEED, **kw)


if __name__ == "__main__":
    main()
