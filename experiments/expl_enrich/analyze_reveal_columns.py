"""Statistical analysis of the ReVeal text-channel columns (2026-07).

Motivation: the frozen-feature pilot showed the ~14-pt precision gap to the base
paper is CODE-SIDE, not in the prose explanation. The new channel front-loads the
code identifiers (function_name, called_functions, risky_apis, string_literals,
lexical_digest) ahead of the risk fields so RoBERTa's 256-token truncation keeps
them. This script measures whether that ordering is (a) safe w.r.t. truncation and
(b) carries more linear label signal than the old CORE channel.

Reports, on the deduped train + full val (same split the trainer uses):
  1. per-column coverage + word-length, split by label
  2. univariate label signal per column (AUC of column word-count; vuln-rate when
     the column is populated vs empty)
  3. token-length distribution of the NEW channel under the RoBERTa tokenizer and
     the % of samples truncated at 256 (+ which columns survive the cut)
  4. TF-IDF + logistic-regression proxy (seconds, no fine-tuning) on val:
     OLD channel vs NEW channel vs identifiers-only vs raw_code
     scored on ROC / PR / precision@recall=39.52% (the paper's operating point)
"""
from __future__ import annotations
import os, sys, json
import numpy as np

ROOT = r"D:\Projects\SemVul"
LADDER = os.path.join(ROOT, "experiments", "fusevul_ladder")
for p in (ROOT, LADDER):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["SEMVUL_ACTIVE_DIR"] = "1"

from src.data_io import Sample, _to_str, _to_list
import data as data_mod

COLS = ["function_name", "called_functions", "risky_apis", "string_literals",
        "lexical_digest", "risky_operations", "missing_checks",
        "safety_indicators", "evidence_tokens"]
NEW_FIELDS = ",".join(COLS)
OLD_FIELDS = ("risky_operations,missing_checks,evidence_tokens,safety_indicators,"
              "tail_facts,lexical_digest,tail_digest")
IDS_FIELDS = "function_name,called_functions,risky_apis,string_literals,lexical_digest"
PAPER_REC = 0.3952


def _load_raw(split):
    path = os.path.join(ROOT, "explanations", "SemanticVul", "ACTIVE", "reveal",
                        f"{split}.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _samples(rows):
    return [Sample(sample_id=str(r.get("sample_id", "")), label=int(r["label"]),
                   code=r.get("raw_code", "") or "",
                   explanation=r.get("explanation", {}) or {}) for r in rows]


def _col_text(e, c):
    """Same coercion the loader uses, per column."""
    v = e.get(c)
    if isinstance(v, list):
        return " ".join(_to_list(v))
    return _to_str(v)


def prec_at_rec(prob, y, rec=PAPER_REC):
    from sklearn.metrics import precision_recall_curve
    P, R, _ = precision_recall_curve(y, prob)
    return 100 * P[np.argmin(np.abs(R - rec))]


def render(samples, fields):
    os.environ["SEMVUL_EXPL_FIELDS"] = fields
    return [s.explanation_text for s in samples]


def main():
    tr_rows, va_rows = _load_raw("train"), _load_raw("val")
    tr_all = _samples(tr_rows)
    va = _samples(va_rows)
    keep = data_mod._dedup_train(tr_all, [s.sample_id for s in va])
    tr = [tr_all[i] for i in keep]
    ytr = np.array([s.label for s in tr]); yva = np.array([s.label for s in va])
    print(f"[data] train(deduped)={len(tr)} pos={100*ytr.mean():.1f}%  "
          f"val={len(va)} pos={100*yva.mean():.1f}%")

    # ---- 1 & 2: per-column coverage, length, univariate signal (train) -------
    from sklearn.metrics import roc_auc_score
    print("\n" + "="*94)
    print("PER-COLUMN STATS (deduped train)   [order = NEW serialization order]")
    print("="*94)
    print(f"{'column':18s} {'nonempty%':>9s} {'words(mean)':>11s} "
          f"{'w|vuln':>7s} {'w|safe':>7s} {'AUC(wc)':>8s} "
          f"{'vulnrate:full':>13s} {'vulnrate:empty':>14s}")
    for c in COLS:
        wc = np.array([len(_col_text(s.explanation, c).split()) for s in tr], float)
        ne = wc > 0
        auc = roc_auc_score(ytr, wc) * 100 if len(np.unique(wc)) > 1 else 50.0
        vr_full = 100 * ytr[ne].mean() if ne.any() else float("nan")
        vr_empty = 100 * ytr[~ne].mean() if (~ne).any() else float("nan")
        print(f"{c:18s} {100*ne.mean():8.1f}% {wc.mean():11.2f} "
              f"{wc[ytr==1].mean():7.2f} {wc[ytr==0].mean():7.2f} {auc:8.2f} "
              f"{vr_full:12.1f}% {vr_empty:13.1f}%")
    print("  AUC(wc): ROC-AUC using the column's word-count as the sole ranker "
          "(50=no signal). vulnrate:full/empty = vuln %% among rows where the "
          "column is populated / empty.")

    # ---- 3: token-length + truncation under RoBERTa, NEW order ---------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("roberta-base")
    new_tr = render(tr, NEW_FIELDS)
    lens = np.array([len(tok(t, truncation=False)["input_ids"]) for t in new_tr])
    print("\n" + "="*94)
    print("NEW-CHANNEL TOKEN LENGTH (RoBERTa, no truncation)  budget=256")
    print("="*94)
    for p in (50, 75, 90, 95, 99):
        print(f"  p{p:<2d} = {np.percentile(lens, p):6.0f} tokens")
    print(f"  mean = {lens.mean():.0f}   max = {lens.max()}   "
          f">256 (truncated) = {100*(lens>256).mean():.1f}%")
    # cumulative prefix: how many tokens each column consumes on average, in order
    print("\n  cumulative tokens by column prefix (mean) -> what survives 256:")
    cum = 0
    for i, c in enumerate(COLS):
        prefix = ",".join(COLS[:i+1])
        pt = render(tr, prefix)
        m = np.mean([len(tok(t, truncation=False)["input_ids"]) for t in pt[:2000]])
        flag = "  <-- crosses 256 here" if (cum <= 256 < m) else ""
        print(f"    +{c:18s} cum~{m:6.0f}{flag}")
        cum = m

    # ---- 4: TF-IDF + logistic proxy on val -----------------------------------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    print("\n" + "="*94)
    print("TF-IDF + LOGISTIC PROXY (fit train, eval val)  -- cheap linear signal check")
    print("="*94)
    print(f"{'representation':22s} {'ROC':>7s} {'PR':>7s} {'prec@rec39.5':>13s}  (paper 57.24)")
    def tfidf_eval(name, tr_txt, va_txt):
        vec = TfidfVectorizer(min_df=3, max_features=50000, ngram_range=(1, 2),
                              token_pattern=r"(?u)\b\w+\b")
        Xtr = vec.fit_transform(tr_txt); Xva = vec.transform(va_txt)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xva)[:, 1]
        print(f"{name:22s} {100*roc_auc_score(yva,p):7.2f} "
              f"{100*average_precision_score(yva,p):7.2f} {prec_at_rec(p,yva):12.2f}")
    tfidf_eval("OLD channel", render(tr, OLD_FIELDS), render(va, OLD_FIELDS))
    tfidf_eval("NEW channel (9-col)", render(tr, NEW_FIELDS), render(va, NEW_FIELDS))
    tfidf_eval("identifiers only", render(tr, IDS_FIELDS), render(va, IDS_FIELDS))
    tfidf_eval("raw_code", [s.code for s in tr], [s.code for s in va])


if __name__ == "__main__":
    main()
