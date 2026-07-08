"""RO4 ablation driver: Devign with focal loss + capped class weights ON.

Devign was trained with plain cross-entropy (train.py: focal auto = ReVeal only).
This runs L1/L2/L3 on Devign with focal="on", split_seed=1337 (same tune carve as
the existing ensemble members, so probs are directly comparable / poolable), writing
to experiments/runs/focal_devign/ so nothing existing is clobbered.

Resumable: a rung whose JSON already exists is skipped.
Run (background):  .venv/Scripts/python.exe experiments/fusevul_ladder/run_focal_devign.py
"""
from __future__ import annotations
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train import train_rung  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "runs", "focal_devign")
os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    for rung in ("L1", "L2", "L3"):
        done = os.path.join(OUT, f"fusevul_ladder_devign_{rung}.json")
        if os.path.exists(done):
            print(f"[skip] devign {rung} already done -> {done}", flush=True)
            continue
        print(f"\n===== devign {rung} focal=ON seed=1337 =====", flush=True)
        train_rung("devign", rung, focal="on", seed=1337, split_seed=1337, out_dir=OUT)
    print("\n[focal_devign] all rungs done.", flush=True)
