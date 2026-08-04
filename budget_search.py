"""Truncation-aware proxy: score exactly what the 512-token encoder will see.

Earlier proxy runs used FULL text (no length limit) — optimistic. Here every
input is built the way the encoder will receive it:

    <s> prefix(K tokens) </s></s> code(509-K tokens) </s>

using the real GraphCodeBERT/RoBERTa tokenizer to truncate, then decoded back
to text for the TF-IDF+LR proxy. Two searches:

  A. BUDGET SEARCH - prefix cap K in {0,64,96,128,160,192,256} (prefix-first)
     plus append-layout reference at K=128. Prefix field priority:
     lexical_digest > nonzero code_metrics > called_functions > evidence_tokens
     (highest measured value first, so low-value text is what gets cut).
  B. WINDOW SWEEP - code-only proxy at window 509/1024/2048/unlimited: how
     much signal lives beyond 512? (= is a long-context encoder worth it?)

Usage: python budget_search.py [dataset_dir]
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("HF_HOME", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"))
from transformers import AutoTokenizer  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)

WINDOW = 512 - 3  # content tokens after <s> </s></s> </s>
KS = [0, 64, 96, 128, 160, 192, 256]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def prefix_text(rec):
    """Suffix columns serialized in value-priority order."""
    e = rec["explanation"]
    cm = e.get("code_metrics") or {}
    mets = " ".join(f"{k[2:] if k.startswith('n_') else k}={v}"
                    for k, v in cm.items() if v)
    calls = e.get("called_functions") or []
    ev = e.get("evidence_tokens") or []
    parts = [
        f"digest: {e.get('lexical_digest', '')}",
        f"metrics: {mets}",
        "calls: " + " ".join(str(c) for c in calls[:15]),
        "evidence: " + " ; ".join(str(t) for t in ev[:8]),
    ]
    return " | ".join(parts)


def encode_all(tok, texts):
    ids = []
    for i in range(0, len(texts), 512):
        ids.extend(tok(texts[i:i + 512], add_special_tokens=False)["input_ids"])
    return ids


def build_views(tok, rows):
    """Pre-tokenize prefix and code once; return token-id lists."""
    return (encode_all(tok, [prefix_text(r) for r in rows]),
            encode_all(tok, [r["raw_code"] for r in rows]))


def make_inputs(tok, pre_ids, code_ids, k, layout="prefix", window=WINDOW):
    out = []
    for p, c in zip(pre_ids, code_ids):
        if layout == "prefix":
            p_t = p[:k]
            c_t = c[:window - len(p_t)]
            ids = p_t + c_t
        else:  # append: code first, prefix gets leftovers
            c_t = c[:window]
            room = window - len(c_t)
            ids = c_t + p[:min(room, k)]
        out.append(tok.decode(ids))
    return out


def proxy(train_texts, y_tr, val_texts, y_va):
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                          max_features=30000, min_df=2)
    Xtr = vec.fit_transform(train_texts)
    Xva = vec.transform(val_texts)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0,
                             solver="liblinear")
    clf.fit(Xtr, y_tr)
    prob = clf.predict_proba(Xva)[:, 1]
    pred = (prob >= 0.5).astype(int)
    f1, acc = f1_score(y_va, pred), accuracy_score(y_va, pred)
    return {"f1": f1, "acc": acc, "score": (f1 + acc) / 2,
            "precision": precision_score(y_va, pred, zero_division=0),
            "recall": recall_score(y_va, pred),
            "auc": roc_auc_score(y_va, prob)}


def show(tag, m):
    print(f"  {tag:34s} F1={m['f1']:.4f} acc={m['acc']:.4f} "
          f"score={m['score']:.4f} P={m['precision']:.3f} "
          f"R={m['recall']:.3f} AUC={m['auc']:.4f}")


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "explanations/SemanticVul/reveal"
    tr_p = os.path.join(ds, "train_improved.jsonl")
    va_p = os.path.join(ds, "val_improved.jsonl")
    train = read_jsonl(tr_p if os.path.exists(tr_p) else os.path.join(ds, "train.jsonl"))
    val = read_jsonl(va_p if os.path.exists(va_p) else os.path.join(ds, "val.jsonl"))
    y_tr = np.array([r["label"] for r in train])
    y_va = np.array([r["label"] for r in val])

    tok = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base",
                                        local_files_only=True)
    print("tokenizing...")
    pre_tr, code_tr = build_views(tok, train)
    pre_va, code_va = build_views(tok, val)

    print(f"\nA. BUDGET SEARCH (window={WINDOW} content tokens, "
          f"input = prefix[:K] + code[:window-K])")
    best = None
    for k in KS:
        m = proxy(make_inputs(tok, pre_tr, code_tr, k),
                  y_tr,
                  make_inputs(tok, pre_va, code_va, k),
                  y_va)
        show(f"prefix K={k:3d} / code {WINDOW - k}", m)
        if best is None or m["score"] > best[1]["score"]:
            best = (k, m)
    m = proxy(make_inputs(tok, pre_tr, code_tr, 128, layout="append"),
              y_tr,
              make_inputs(tok, pre_va, code_va, 128, layout="append"),
              y_va)
    show("APPEND ref (code first, K=128)", m)
    print(f"\n  => BEST: prefix K={best[0]} / code {WINDOW - best[0]}  "
          f"(F1={best[1]['f1']:.4f} acc={best[1]['acc']:.4f})")

    print("\nB. WINDOW SWEEP (code only - marginal value of a longer window)")
    for w in [WINDOW, 1024, 2048, None]:
        cut_tr = [tok.decode(c[:w]) if w else tok.decode(c) for c in code_tr]
        cut_va = [tok.decode(c[:w]) if w else tok.decode(c) for c in code_va]
        m = proxy(cut_tr, y_tr, cut_va, y_va)
        show(f"code window = {w or 'unlimited'}", m)

    print("\nC. FULL-TEXT REFERENCE (what earlier proxy runs measured)")
    m = proxy([prefix_text(r) + " | " + r["raw_code"] for r in train], y_tr,
              [prefix_text(r) + " | " + r["raw_code"] for r in val], y_va)
    show("prefix + code, no truncation", m)


if __name__ == "__main__":
    main()
