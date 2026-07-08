"""Queued Devign treatment: L1/L2/L3 at max_code=512 on cleaned+augmented
enriched data with qual_v2. Waits for the ReVeal enriched retrain to free the
GPU first (polls for its final JSON; falls back to 'no progress for 45 min').

Motivation (2026-07-08 stratified analysis, scratchpad devign_deep.py):
  - val stratum that FITS the 320-token window: ens ROC 67.75, acc@0.5 61.92
    -> ALREADY beats the stated 60.39 target on 67% of val;
  - stratum truncated at 320: ROC 53.03 (chance) -> the entire Devign gap
    lives in the truncated third; frozen 512-token LoRA embeddings score ~57
    there, so window length (not the text channel) is the lever.

Writes to experiments/runs/enriched512/ (nothing overwritten). Resumable.
  .venv/Scripts/python.exe experiments/expl_enrich/retrain_devign512.py
"""
import os
import sys
import time

os.environ["SEMVUL_EXPL_VARIANT"] = "enriched"
os.environ["SEMVUL_TRAIN_SUFFIX"] = "clean.aug"
os.environ["SEMVUL_QUAL_V2"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LADDER = os.path.join(ROOT, "experiments", "fusevul_ladder")
for _p in (ROOT, LADDER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNS_ENR = os.path.join(ROOT, "experiments", "runs", "enriched")
OUT = os.path.join(ROOT, "experiments", "runs", "enriched512")


def wait_for_gpu():
    """Block until the reveal retrain is done (final JSON) or stalled 45 min."""
    final = os.path.join(RUNS_ENR, "fusevul_ladder_reveal_L3.json")
    partials = [os.path.join(RUNS_ENR, f"fusevul_ladder_reveal_{r}_partial.json")
                for r in ("L2", "L3")]
    while True:
        if os.path.exists(final):
            print("[queue] reveal retrain finished, GPU free", flush=True)
            return
        mtimes = [os.path.getmtime(p) for p in partials if os.path.exists(p)]
        if mtimes and time.time() - max(mtimes) > 45 * 60:
            print("[queue] reveal retrain stalled >45min, proceeding", flush=True)
            return
        if not mtimes:  # reveal job not even started -> assume GPU free
            print("[queue] no reveal partials found, proceeding", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    wait_for_gpu()
    from train import train_rung  # noqa: E402  (imports torch -> after wait)
    for rung in ("L1", "L2", "L3"):
        tag = os.path.join(OUT, f"fusevul_ladder_devign_{rung}.json")
        if os.path.exists(tag):
            print(f"[skip] {tag} exists", flush=True)
            continue
        # batch 2 x accum 16 keeps the effective batch (32) of the 320-token
        # baseline while fitting 512-token sequences in 8 GB.
        train_rung("devign", rung, out_dir=OUT, split_seed=1337,
                   max_code=512, batch=2, grad_accum=16)
