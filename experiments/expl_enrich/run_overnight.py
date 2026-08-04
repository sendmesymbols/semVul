"""One-command unattended launcher for a spare GPU.

Chains the full treatment pipeline end to end, resumable, fail-soft:
  1. regenerate enriched JSONLs        (deterministic, CPU, ~1 min)
  2. clean + augment train sets        (deterministic, CPU, ~1 min)
  3. train the treatment configs        (GPU, the long part)
       reveal L2/L3 enriched+clean       -> runs/enriched/s<seed>/
       devign L1/L2/L3 enriched+512+aug   -> runs/enriched512/s<seed>/
  4. print a desktop-local ensemble verdict so morning greets you with numbers

Every (dataset, rung, seed) that already has a JSON is skipped, so a crash or a
re-launch never loses or repeats finished work. A single rung failing (e.g. OOM)
is logged and does not abort the rest.

  # self-sufficient overnight run incl. the primary seed:
  .venv/Scripts/python.exe experiments/expl_enrich/run_overnight.py --seeds 1337 2024 2025
  # extra seeds only (laptop still owns 1337 -> no duplication):
  .venv/Scripts/python.exe experiments/expl_enrich/run_overnight.py --seeds 2024 2025 2026
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(step, argv):
    print(f"\n{'='*70}\n[overnight] {step}\n{'='*70}", flush=True)
    r = subprocess.run([PY] + argv)
    if r.returncode != 0:
        print(f"[overnight] WARNING: {step} exited {r.returncode} "
              f"(continuing)", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--only", choices=["devign", "reveal"], default=None)
    ap.add_argument("--batch512", type=int, default=2,
                    help="devign 512-token batch (2 fits 8GB; 4 on >=16GB)")
    ap.add_argument("--code-window", choices=["head", "evidence"], default="head",
                    help="'evidence' = evidence-centered 512 window (treatment "
                         "arm for ReVeal tail truncation)")
    args = ap.parse_args()

    run("1/4 enrich explanations",
        [os.path.join(HERE, "run_enrich.py")])
    run("2/4 clean + augment train",
        [os.path.join(HERE, "augment_train.py"),
         "--variant", "enriched", "--aug-copies", "1"])

    train = [os.path.join(HERE, "retrain_remote.py"),
             "--seeds", *[str(s) for s in args.seeds],
             "--batch512", str(args.batch512),
             "--code-window", args.code_window]
    if args.only:
        train += ["--only", args.only]
    run("3/4 train treatment configs (GPU, long)", train)

    run("4/4 desktop-local ensemble verdict",
        [os.path.join(HERE, "dual_eval.py")])
    print("\n[overnight] done. Zip experiments/runs/enriched* and copy to the "
          "main machine; re-run ensemble.py there for the merged verdict.",
          flush=True)


if __name__ == "__main__":
    main()
