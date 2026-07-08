"""Statistically-motivated TRAIN-set cleaning + augmentation.

Diagnosis this responds to (2026-07-08 scratchpad dataset stats):
  - devign: 185 train code strings carry BOTH labels (contradictory gradients);
    300 train rows are exact-code copies of val rows under DIFFERENT sample_ids
    (the ladder's sample_id dedup does not catch them -> train->val leakage);
  - reveal: 213 conflicting-label code strings; 69 exact-code leaks;
    409 cross-label duplicate explanation rows.
  - devign is 100% VARn/FUNn-anonymized -> index permutation is an exactly
    label-preserving augmentation (pure renaming of already-anonymous names);
    it teaches the encoder that the index digits carry no meaning.

What it writes (originals untouched):
  <ds>_train.clean.jsonl        cleaning only
  <ds>_train.clean.aug.jsonl    cleaning + augmentation (devign only by default)

Run AFTER run_enrich.py if you want enriched+clean+aug (use --variant enriched).

  .venv/Scripts/python.exe experiments/expl_enrich/augment_train.py
  .venv/Scripts/python.exe experiments/expl_enrich/augment_train.py --variant enriched --aug-copies 1
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from src.config import EXPL_DIR

_VARFUN = re.compile(r"\b(VAR|FUN)(\d+)\b")


def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean(train, val, tag):
    """Drop conflicting-label code groups and exact-code train->val leaks."""
    by_code = defaultdict(set)
    for r in train:
        by_code[r["raw_code"]].add(int(r["label"]))
    conflict = {c for c, ls in by_code.items() if len(ls) > 1}
    val_codes = {r["raw_code"] for r in val}
    kept, n_conf, n_leak, n_dup = [], 0, 0, 0
    seen = set()
    for r in train:
        c = r["raw_code"]
        if c in conflict:
            n_conf += 1
            continue
        if c in val_codes:
            n_leak += 1
            continue
        key = (c, int(r["label"]))
        if key in seen:          # exact same-label duplicate adds nothing
            n_dup += 1
            continue
        seen.add(key)
        kept.append(r)
    print(f"[{tag}] clean: kept {len(kept)}/{len(train)} "
          f"(dropped conflict={n_conf}, val-leak={n_leak}, same-label-dup={n_dup})",
          flush=True)
    return kept


def _remap_str(s: str, vmap, fmap) -> str:
    def rep(m):
        kind, idx = m.group(1), int(m.group(2))
        m2 = vmap if kind == "VAR" else fmap
        return f"{kind}{m2.get(idx, idx)}"
    return _VARFUN.sub(rep, s)


def _remap_any(v, vmap, fmap):
    if isinstance(v, str):
        return _remap_str(v, vmap, fmap)
    if isinstance(v, list):
        return [_remap_any(x, vmap, fmap) for x in v]
    if isinstance(v, dict):
        return {k: _remap_any(x, vmap, fmap) for k, x in v.items()}
    return v


def permute_row(row, rng):
    """Bijective permutation of VARn/FUNn indices, applied consistently to code
    AND every explanation string (evidence quotes stay verbatim-consistent)."""
    idxs = {"VAR": set(), "FUN": set()}
    for kind, i in _VARFUN.findall(row["raw_code"]):
        idxs[kind].add(int(i))
    maps = {}
    for kind in ("VAR", "FUN"):
        src = sorted(idxs[kind])
        tgt = src[:]
        rng.shuffle(tgt)
        maps[kind] = dict(zip(src, tgt))
    out = dict(row)
    out["raw_code"] = _remap_str(row["raw_code"], maps["VAR"], maps["FUN"])
    out["explanation"] = _remap_any(row.get("explanation") or {},
                                    maps["VAR"], maps["FUN"])
    out["sample_id"] = f"{row['sample_id']}_p{rng.integers(0, 10**6)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="", help="'' = original jsonl, "
                    "'enriched' = *.enriched.jsonl input")
    ap.add_argument("--aug-copies", type=int, default=1,
                    help="extra permuted copies per devign train row (0=off)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    sfx = f".{args.variant}" if args.variant else ""

    for ds in ("devign", "reveal"):
        rng = np.random.default_rng(args.seed)
        train = load_jsonl(EXPL_DIR / ds / f"{ds}_train{sfx}.jsonl")
        val = load_jsonl(EXPL_DIR / ds / f"{ds}_val{sfx}.jsonl")
        kept = clean(train, val, ds)
        out_clean = EXPL_DIR / ds / f"{ds}_train{sfx}.clean.jsonl"
        with out_clean.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{ds}] wrote {out_clean.name} "
              f"(pos {100*np.mean([r['label'] for r in kept]):.1f}%)", flush=True)

        if ds == "devign" and args.aug_copies > 0:
            aug = list(kept)
            for _ in range(args.aug_copies):
                for r in kept:
                    aug.append(permute_row(r, rng))
            out_aug = EXPL_DIR / ds / f"{ds}_train{sfx}.clean.aug.jsonl"
            with out_aug.open("w", encoding="utf-8") as f:
                for r in aug:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{ds}] wrote {out_aug.name} (n={len(aug)})", flush=True)


if __name__ == "__main__":
    main()
