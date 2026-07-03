"""Data for the FuSEVul component ladder.

Reuses the project's canonical loaders so code / explanation / label / quality
stay aligned by sample_id. Dedup is applied to TRAIN ONLY (drop within-train
duplicate functions and any train function that also appears in val); the val
set is left identical to the benchmark so "beats stated results" is a direct
same-split comparison.
"""
from __future__ import annotations
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # D:\Projects\SemVul
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from src.data_io import load_split
from src.quality_features import compute_batch


def _dedup_train(train, val_sids):
    seen, keep = set(), []
    vs = set(val_sids)
    for i, s in enumerate(train):
        sid = s.sample_id
        if sid in vs or sid in seen:      # drop train∩val leak + within-train dups
            continue
        seen.add(sid)
        keep.append(i)
    return keep


def _pack(samples):
    return dict(
        code=[s.code for s in samples],
        expl=[s.explanation_text for s in samples],
        y=np.asarray([int(s.label) for s in samples], dtype=np.int64),
        qual=compute_batch(samples).astype(np.float32),
        sids=[s.sample_id for s in samples],
    )


def load(dataset: str, subset: int | None = None):
    tr = load_split(dataset, "train")
    va = load_split(dataset, "val")
    keep = _dedup_train(tr, [s.sample_id for s in va])
    tr = [tr[i] for i in keep]
    if subset:
        tr = tr[:subset]
        va = va[:max(50, subset // 4)]
    train, val = _pack(tr), _pack(va)
    print(f"[data] {dataset}: train={len(train['y'])} "
          f"(pos {train['y'].mean()*100:.1f}%)  val={len(val['y'])} "
          f"(pos {val['y'].mean()*100:.1f}%)  qual_dim={train['qual'].shape[1]}",
          flush=True)
    return train, val
