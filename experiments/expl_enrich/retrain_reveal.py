"""Validation retrain: reveal L2+L3 on enriched + cleaned data with qual_v2.

Writes to experiments/runs/enriched/ (never touches the baseline runs).
  .venv/Scripts/python.exe experiments/expl_enrich/retrain_reveal.py
"""
import os
import sys

os.environ["SEMVUL_EXPL_VARIANT"] = "enriched"
os.environ["SEMVUL_TRAIN_SUFFIX"] = "clean"
os.environ["SEMVUL_QUAL_V2"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LADDER = os.path.join(ROOT, "experiments", "fusevul_ladder")
for _p in (ROOT, LADDER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train import train_rung  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "runs", "enriched")

if __name__ == "__main__":
    for rung in ("L2", "L3"):
        tag = os.path.join(OUT, f"fusevul_ladder_reveal_{rung}.json")
        if os.path.exists(tag):
            print(f"[skip] {tag} exists", flush=True)
            continue
        train_rung("reveal", rung, out_dir=OUT, split_seed=1337)
