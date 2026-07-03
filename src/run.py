"""ENTRY POINT: full campaign for one dataset -- encode, train all 3 ladders,
run the component ablation matrix, ensemble to L3, write reports.

Nothing here does anything mysterious. Every step is a single function call
in a fixed order. Read top to bottom.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from src.config import CACHE_DIR, RUNS_DIR
from src.encode_code import encode as encode_code_frozen
from src.encode_text import encode as encode_text
from src.eval import report as make_report
from src.lora_finetune import encode as encode_code_lora
from src.lora_finetune import train as train_lora
from src.reports import all_reports
from src.train import run as train_head


def _stage_1_cache_features(dataset: str, do_l2: bool):
    print("\n[stage-1] cache text (MiniLM) embeddings")
    for split in ("train", "val"):
        encode_text(dataset, split)

    print("\n[stage-2] fine-tune GraphCodeBERT (LoRA) and cache embeddings")
    train_lora(dataset, encoder="graphcodebert")
    for split in ("train", "val"):
        encode_code_lora(dataset, split, encoder="graphcodebert")

    if do_l2:
        print("\n[stage-3] cache UniXcoder frozen embeddings (for L2)")
        for split in ("train", "val"):
            encode_code_frozen(dataset, split, encoder="unixcoder")


def _stage_train_ladders(dataset: str, ladders: List[str], ablate: bool):
    tags_full = [("full", "gated", True, True)]
    tags_ablate = [
        ("no_expl", "gated", False, True),
        ("no_qual", "gated", True, False),
        ("concat",  "concat", True, True),
    ]
    for lad in ladders:
        print(f"\n[train] ladder={lad}")
        for tag, fusion, use_expl, use_qual in tags_full:
            train_head(dataset, lad, tag=tag, fusion=fusion,
                       use_expl=use_expl, use_qual=use_qual)
        if ablate:
            for tag, fusion, use_expl, use_qual in tags_ablate:
                train_head(dataset, lad, tag=tag, fusion=fusion,
                           use_expl=use_expl, use_qual=use_qual)


def _stage_l3_ensemble(dataset: str):
    """Cross-family probability ensemble of L1 (LoRA-GCB) + L2 (LoRA-GCB + CodeT5+).
    We only ensemble the 'full' variants -- L3 is the money shot."""
    l1 = RUNS_DIR / f"{dataset}_L1_full_probs.npz"
    l2 = RUNS_DIR / f"{dataset}_L2_full_probs.npz"
    if not (l1.exists() and l2.exists()):
        print("[L3] skip -- need both L1 and L2 probs first")
        return
    a = np.load(l1); b = np.load(l2)
    probs_val  = 0.5 * (a["probs_val"]  + b["probs_val"])
    probs_tune = 0.5 * (a["probs_tune"] + b["probs_tune"])
    y_val, y_tune = a["y_val"], a["y_tune"]

    rep = make_report(probs_val, y_val, probs_tune, y_tune)
    payload = {
        "run_id": f"{dataset}_L3_full",
        "dataset": dataset,
        "ladder": "L3",
        "tag": "full",
        "config": {"members": ["L1_full", "L2_full"], "weights": [0.5, 0.5]},
        "per_seed": [],
        "ensemble_metrics": {k: v.as_dict() for k, v in rep.items()},
    }
    out = RUNS_DIR / f"{dataset}_L3_full.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    np.savez_compressed(RUNS_DIR / f"{dataset}_L3_full_probs.npz",
                        probs_val=probs_val, y_val=y_val,
                        probs_tune=probs_tune, y_tune=y_tune)
    print(f"[L3] wrote {out.name}")


def campaign(dataset: str, ladders: List[str], ablate: bool):
    _stage_1_cache_features(dataset, do_l2=("L2" in ladders or "L3" in ladders))
    train_ladders = [l for l in ladders if l in ("L1", "L2")]
    _stage_train_ladders(dataset, train_ladders, ablate=ablate)
    if "L3" in ladders:
        _stage_l3_ensemble(dataset)
    all_reports(dataset, ladders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["devign", "reveal"])
    ap.add_argument("--ladders", nargs="*", default=["L1", "L2", "L3"],
                    choices=["L1", "L2", "L3"])
    ap.add_argument("--ablate", action="store_true",
                    help="also train no_expl/no_qual/concat variants per ladder")
    args = ap.parse_args()
    campaign(args.dataset, args.ladders, args.ablate)


if __name__ == "__main__":
    main()
