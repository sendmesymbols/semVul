"""Corrected-val track: drop provably-broken benchmark val rows.

Criteria (all exact-code-string based, label-blind decisions except (b) which
uses label DISAGREEMENT, never the label's value):
  (a) LEAK: the val row's exact code also appears in TRAIN (any label, any
      sample_id) -> the model can have memorized it; keeping it inflates val.
  (b) CONTRADICTED: the exact code appears elsewhere (train or val) with the
      OPPOSITE label -> the row's label is unknowable; no model can be scored
      on it meaningfully.
  (c) WITHIN-VAL DUP: the exact code appears earlier in val -> double counting;
      keep the first occurrence only.

Outputs (originals untouched):
  explanations/SemanticVul/<ds>/<ds>_val[.enriched].clean.jsonl
  experiments/runs/val_clean_mask_<ds>.npz   keep-mask aligned to the ORIGINAL
                                             val row order (so every saved
                                             val_prob vector can be re-scored
                                             on corrected val without retraining)

IMPORTANT: numbers on corrected val are NOT comparable to FuSEVul's stated
numbers (those live on the benchmark split). Report both tracks, labeled.

  .venv/Scripts/python.exe experiments/expl_enrich/correct_val.py
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from src.config import EXPL_DIR, RUNS_DIR


def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    for ds in ("devign", "reveal"):
        train = load_jsonl(EXPL_DIR / ds / f"{ds}_train.jsonl")
        val = load_jsonl(EXPL_DIR / ds / f"{ds}_val.jsonl")

        train_codes = defaultdict(set)
        for r in train:
            train_codes[r["raw_code"]].add(int(r["label"]))
        val_code_labels = defaultdict(set)
        for r in val:
            val_code_labels[r["raw_code"]].add(int(r["label"]))

        keep = np.ones(len(val), dtype=bool)
        reasons = {"leak": 0, "contradicted": 0, "within_val_dup": 0}
        seen = set()
        for i, r in enumerate(val):
            c, lab = r["raw_code"], int(r["label"])
            labels_elsewhere = train_codes.get(c, set()) | val_code_labels[c]
            if (1 - lab) in labels_elsewhere:
                keep[i] = False
                reasons["contradicted"] += 1
                continue
            if c in train_codes:
                keep[i] = False
                reasons["leak"] += 1
                continue
            if c in seen:
                keep[i] = False
                reasons["within_val_dup"] += 1
                continue
            seen.add(c)

        kept = [r for r, k in zip(val, keep) if k]
        pos = np.mean([r["label"] for r in kept])
        print(f"[{ds}] val {len(val)} -> {len(kept)} "
              f"(dropped: {reasons})  pos={pos*100:.1f}% "
              f"(was {np.mean([r['label'] for r in val])*100:.1f}%)", flush=True)

        np.savez_compressed(RUNS_DIR / f"val_clean_mask_{ds}.npz",
                            keep=keep,
                            sample_ids=np.array([r["sample_id"] for r in val]))
        for variant in ("", ".enriched"):
            src = EXPL_DIR / ds / f"{ds}_val{variant}.jsonl"
            if not src.exists():
                continue
            rows_v = load_jsonl(src)
            assert len(rows_v) == len(val)
            dst = EXPL_DIR / ds / f"{ds}_val{variant}.clean.jsonl"
            with dst.open("w", encoding="utf-8") as f:
                for r, k in zip(rows_v, keep):
                    if k:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{ds}] wrote {dst.name}", flush=True)


if __name__ == "__main__":
    main()
