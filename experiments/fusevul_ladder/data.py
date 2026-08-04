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
  SEMVUL_CONF_SWITCH=1          expose raw explanation.confidence as a separate
                                 scalar prior for L3 so low-confidence rows can
                                 close the explanation gate and lean on code
  SEMVUL_CODE_WINDOW=evidence    for functions longer than the code window,
                                 keep the span CENTERED on the explanation's
                                 verbatim evidence instead of the function head
                                 (attacks ReVeal's 58%-of-vulnerables truncation,
                                 where the defect is usually in the tail). Applied
                                 label-blind to train AND val. Window size in
                                 words: SEMVUL_CODE_WINDOW_WORDS (default 340,
                                 ~512 GraphCodeBERT tokens).
"""
from __future__ import annotations
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # D:\Projects\SemVul
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from src.data_io import load_split, _to_str, _active_path
from src.config import EXPL_DIR

_CONF_ORD = {"low": 25.0, "medium": 50.0, "high": 75.0}


def _evidence_window(code: str, expl: dict, budget_words: int) -> str:
    """Return a <=budget_words span of `code` centered on the explanation's
    verbatim evidence. Falls back to the full code (tokenizer head-truncates)
    when the function is short or no evidence string matches verbatim."""
    words = code.split()
    if len(words) <= budget_words:
        return code
    cands = [_to_str(x) for x in (expl.get("evidence_tokens") or [])]
    for g in (expl.get("safety_indicators") or []):
        if isinstance(g, dict):
            cands.append(_to_str(g.get("evidence")))
    for ro in (expl.get("risky_operations") or []):
        m = re.search(r"\[evidence:\s*(.*?)\]", _to_str(ro))
        if m:
            cands.append(m.group(1))
    positions = []
    for c in cands:
        c = c.strip()
        if len(c) < 4:
            continue
        idx = code.find(c)
        if idx >= 0:
            positions.append(len(code[:idx].split()))
    if not positions:
        return code
    center = int(np.median(positions))
    hi = min(len(words), max(center + budget_words // 2, budget_words))
    lo = max(0, hi - budget_words)
    return " ".join(words[lo:hi])


def _pack_code_ev(samples):
    """Explanation-guided code windows. Part of the EXPLANATION component
    (L2/L3 only — the window position is derived from the explanation, so
    giving it to L1 would contaminate the code-only baseline; supervisor
    2026-07-09). train.py selects 'code_ev' for L2/L3 and 'code' for L1."""
    if os.environ.get("SEMVUL_CODE_WINDOW", "").strip() != "evidence":
        return None
    budget = int(os.environ.get("SEMVUL_CODE_WINDOW_WORDS", "340"))
    out, recentered = [], 0
    for s in samples:
        w = _evidence_window(s.code, s.explanation or {}, budget)
        recentered += (w != s.code)
        out.append(w)
    print(f"[data] evidence-window (L2/L3 channel): recentered "
          f"{recentered}/{len(samples)} (budget {budget}w)", flush=True)
    return out


def _confidence_raw(expl: dict) -> float:
    """Confidence on a stable 0..100 scale.

    ACTIVE Reveal rows currently store numeric confidence, while older enriched
    rows may still carry low|medium|high strings. Keep both compatible.
    """
    v = (expl or {}).get("confidence")
    if v is None:
        return 50.0
    if isinstance(v, (int, float)):
        return float(np.clip(v, 0.0, 100.0))
    s = _to_str(v).strip().lower()
    try:
        return float(np.clip(float(s), 0.0, 100.0))
    except ValueError:
        return float(_CONF_ORD.get(s, 50.0))

if os.environ.get("SEMVUL_QUAL_V2") == "1":
    from src.quality_features_v2 import compute_batch
else:
    from src.quality_features import compute_batch


def _load_train(dataset: str):
    """Train split. Prefers the ACTIVE consolidated file when SEMVUL_ACTIVE_DIR
    is set, or falls back to it when the SEMVUL_TRAIN_SUFFIX long-name is absent
    on this machine — so copying only explanations/SemanticVul/ACTIVE/ is enough."""
    ap = _active_path(dataset, "train")
    force_active = bool(os.environ.get("SEMVUL_ACTIVE_DIR", "").strip())
    suffix = os.environ.get("SEMVUL_TRAIN_SUFFIX", "").strip()
    if force_active and ap.exists():
        path = ap
    elif not suffix:
        return load_split(dataset, "train")
    else:
        variant = os.environ.get("SEMVUL_EXPL_VARIANT", "").strip()
        vsfx = f".{variant}" if variant else ""
        path = EXPL_DIR / dataset / f"{dataset}_train{vsfx}.{suffix}.jsonl"
        if not path.exists():
            if ap.exists():
                path = ap                      # ACTIVE fallback
            else:
                raise FileNotFoundError(
                    f"{path} missing (and no ACTIVE fallback at {ap}) - run "
                    f"experiments/expl_enrich/apply_real_enrichment.py, or copy "
                    f"explanations/SemanticVul/ACTIVE/ onto this machine")
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
    print(f"[data] train: {path} (n={len(out)})", flush=True)
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
    # SEMVUL_QUAL_RICH=1 swaps in the RQ2 expanded label-free quality set (see
    # src/quality_features.compute_rich); default path is unchanged for the ladder.
    if os.environ.get("SEMVUL_QUAL_RICH") == "1":
        from src.quality_features import compute_batch_rich
        qual = compute_batch_rich(samples).astype(np.float32)
    else:
        qual = compute_batch(samples).astype(np.float32)
    d = dict(
        code=[s.code for s in samples],   # pure code: the L1 input, always
        expl=[s.explanation_text for s in samples],
        y=np.asarray([int(s.label) for s in samples], dtype=np.int64),
        qual=qual,
        conf=np.asarray([_confidence_raw(s.explanation) for s in samples],
                        dtype=np.float32),
        sids=[s.sample_id for s in samples],
    )
    ev = _pack_code_ev(samples)
    d["code_ev"] = ev if ev is not None else d["code"]
    return d


def load(dataset: str, subset: int | None = None):
    tr = _load_train(dataset)
    va = load_split(dataset, "val")
    keep = _dedup_train(tr, [s.sample_id for s in va])
    tr = [tr[i] for i in keep]
    if subset:
        # SEMVUL_RANDOM_SUBSET=1: sample randomly instead of taking the first N
        # rows (the file may be grouped/sorted, biasing a first-N smoke test).
        # Seeded independently of the training seed (SEMVUL_RANDOM_SUBSET_SEED)
        # so re-running with different training seeds still draws the SAME
        # subset -- otherwise threshold/config comparisons would be confounded
        # by comparing different data, not just different training runs.
        if os.environ.get("SEMVUL_RANDOM_SUBSET") == "1":
            rng = np.random.default_rng(
                int(os.environ.get("SEMVUL_RANDOM_SUBSET_SEED", "0")))
            tr = [tr[i] for i in rng.permutation(len(tr))[:subset]]
            va = [va[i] for i in rng.permutation(len(va))[:max(50, subset // 4)]]
        elif os.environ.get("SEMVUL_SUBSET_FROM_END") == "1":
            # Take the LAST N rows instead of the first N -- checks whether the
            # file's ordering (possibly grouped/sorted) biases a first-N slice,
            # without the extra randomization SEMVUL_RANDOM_SUBSET introduces.
            tr = tr[-subset:]
            va = va[-max(50, subset // 4):]
        else:
            tr = tr[:subset]
            va = va[:max(50, subset // 4)]
    train, val = _pack(tr), _pack(va)
    print(f"[data] {dataset}: train={len(train['y'])} "
          f"(pos {train['y'].mean()*100:.1f}%)  val={len(val['y'])} "
          f"(pos {val['y'].mean()*100:.1f}%)  qual_dim={train['qual'].shape[1]}",
          flush=True)
    return train, val
