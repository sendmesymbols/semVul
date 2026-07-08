"""Data for the FuSEVul component ladder.

Reuses the project's canonical loaders so code / explanation / label / quality
stay aligned by sample_id. Dedup is applied to TRAIN ONLY (drop within-train
duplicate functions and any train function that also appears in val); the val
set is left identical to the benchmark so "beats stated results" is a direct
same-split comparison.

Env knobs (all default off -> behavior identical to before):
  SEMVUL_EXPL_VARIANT=enriched   load *.enriched.jsonl (see src/data_io.py)
  SEMVUL_TRAIN_SUFFIX=clean.aug  load <ds>_train[.<variant>].clean.aug.jsonl
                                 for TRAIN only (val untouched); produced by
                                 experiments/expl_enrich/augment_train.py
  SEMVUL_QUAL_V2=1               44-dim quality block (v1 22 + static-v1 22)
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
from src.config import EXPL_DIR

if os.environ.get("SEMVUL_QUAL_V2") == "1":
    from src.quality_features_v2 import compute_batch
else:
    from src.quality_features import compute_batch


def _load_train(dataset: str):
    """Train split, honoring SEMVUL_TRAIN_SUFFIX (cleaned/augmented files)."""
    suffix = os.environ.get("SEMVUL_TRAIN_SUFFIX", "").strip()
    if not suffix:
        return load_split(dataset, "train")
    variant = os.environ.get("SEMVUL_EXPL_VARIANT", "").strip()
    vsfx = f".{variant}" if variant else ""
    path = EXPL_DIR / dataset / f"{dataset}_train{vsfx}.{suffix}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run experiments/expl_enrich/augment_train.py "
            f"{'--variant ' + variant if variant else ''}")
    import json
    from src.data_io import Sample
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(Sample(sample_id=str(row.get("sample_id", "")),
                              label=int(row["label"]),
                              code=row.get("raw_code", "") or "",
                              explanation=row.get("explanation", {}) or {}))
    print(f"[data] train override: {path.name} (n={len(out)})", flush=True)
    return out


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
    tr = _load_train(dataset)
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
