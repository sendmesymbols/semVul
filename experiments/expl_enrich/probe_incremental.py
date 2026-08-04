"""Task-3 gate: does an explanation variant add signal OVER the code channel?

This is the guardrail that must be passed BEFORE spending on any re-annotation
(Part C). It answers the only question that matters for prediction: not "does the
text have signal" (a length/redundancy proxy can fake that) but "does the text
add AUC on top of the code embedding the model already has."

Method (CPU, no GPU, seconds): cached GraphCodeBERT-LoRA code embeddings are the
baseline. For a candidate explanation variant, TF-IDF the chosen field set and
fit logistic regression on [code] vs [code (+) text]; report val ROC for each and
a 2000-rep bootstrap CI on the delta. If the delta CI includes 0 (or is
negative), the variant adds nothing over code -> do NOT re-annotate on that basis.

  # current enriched channel (default trimmed field set):
  .venv/Scripts/python.exe experiments/expl_enrich/probe_incremental.py
  # score a re-annotation pilot written as <ds>_val.<variant>.jsonl etc.:
  .venv/Scripts/python.exe experiments/expl_enrich/probe_incremental.py --variant enriched_v2
  # full field set / custom fields:
  SEMVUL_EXPL_FIELDS=full  .venv/Scripts/python.exe experiments/expl_enrich/probe_incremental.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import CACHE_DIR, EXPL_DIR
from src.data_io import Sample


def _read(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _texts(rows):
    return [Sample(str(r.get("sample_id", "")), int(r["label"]),
                   r.get("raw_code", "") or "", r.get("explanation") or {}
                   ).explanation_text for r in rows]


def _code(dataset, split):
    d = np.load(CACHE_DIR / f"{dataset}_{split}_code_graphcodebert_lora.npz",
                allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["labels"].astype(np.int64)


def _auc(Xtr, ytr, Xva, yva):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1]


def _boot_delta(y, p_code, p_both, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y)
    d = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(100 * (roc_auc_score(y[idx], p_both[idx])
                        - roc_auc_score(y[idx], p_code[idx])))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="enriched",
                    help="jsonl suffix: reads <ds>_<split>.<variant>.jsonl "
                         "(original row order, to align with the code cache)")
    args = ap.parse_args()
    vsfx = f".{args.variant}" if args.variant else ""

    for ds in ("devign", "reveal"):
        ec, yc = _code(ds, "train")
        ev, yv = _code(ds, "val")
        tr = _read(EXPL_DIR / ds / f"{ds}_train{vsfx}.jsonl")
        va = _read(EXPL_DIR / ds / f"{ds}_val{vsfx}.jsonl")
        if len(tr) != len(yc) or len(va) != len(yv):
            print(f"[{ds}] SKIP: variant rows ({len(tr)}/{len(va)}) != code cache "
                  f"({len(yc)}/{len(yv)}). Use an original-order (non-clean) file.")
            continue
        assert all(int(r["label"]) == int(l) for r, l in zip(tr, yc))
        assert all(int(r["label"]) == int(l) for r, l in zip(va, yv))

        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
        Xt = vec.fit_transform(_texts(tr))
        Xv = vec.transform(_texts(va))
        mu, sd = ec.mean(0, keepdims=True), ec.std(0, keepdims=True) + 1e-6
        Ct, Cv = (ec - mu) / sd, (ev - mu) / sd

        p_code = _auc(Ct, yc, Cv, yv)
        p_both = _auc(hstack([csr_matrix(Ct), Xt]).tocsr(), yc,
                      hstack([csr_matrix(Cv), Xv]).tocsr(), yv)
        a_code = 100 * roc_auc_score(yv, p_code)
        a_both = 100 * roc_auc_score(yv, p_both)
        m, lo, hi = _boot_delta(yv, p_code, p_both)
        adds = lo > 0
        print(f"\n=== {ds}  variant='{args.variant}'  fields="
              f"{os.environ.get('SEMVUL_EXPL_FIELDS','trim')} ===")
        print(f"  code-only ROC        = {a_code:.2f}")
        print(f"  code + text ROC      = {a_both:.2f}")
        print(f"  delta (text adds)    = {m:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"  VERDICT: {'ADDS over code (CI>0)' if adds else 'does NOT add over code -> do not re-annotate on this basis'}")


if __name__ == "__main__":
    main()
