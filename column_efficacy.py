"""Column-efficacy analysis for SemanticVul explanation datasets.

Ranks each explanation column (and greedy combinations) by validation F1 /
accuracy using a fast TF-IDF + logistic-regression proxy, to decide which
columns are worth feeding to RoBERTa / GraphCodeBERT.

Usage:
    python column_efficacy.py explanations/SemanticVul/reveal
    python column_efficacy.py explanations/SemanticVul/devign
    python column_efficacy.py --all
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

METRIC_KEYS = ["n_words", "n_stmts", "n_if", "n_loops", "n_switch", "n_goto",
               "n_return", "n_calls", "n_deref", "n_index", "n_alloc", "n_free",
               "n_unsafe_str", "n_bounded_copy", "truncated", "n_findings",
               "n_guards", "n_findings_tail"]

GREEDY_POOL = 14  # top-N single columns eligible for greedy combination
GREEDY_MAX = 5    # max columns in a combination


def as_text(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ; ".join(as_text(x) for x in v)
    if isinstance(v, dict):
        return " ; ".join(f"{k}: {as_text(x)}" for k, x in v.items())
    return str(v)


def load(path):
    """Load a jsonl split, flattening `explanation` into flat columns."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            e = r.get("explanation") or {}
            llm = e.get("llm_v1") or {}
            row = {
                "label": r["label"],
                "purpose": as_text(e.get("purpose")),
                "data_flow": as_text(e.get("data_flow")),
                "risky_operations": as_text(e.get("risky_operations")),
                "missing_checks": as_text(e.get("missing_checks")),
                "evidence_tokens": as_text(e.get("evidence_tokens")),
                "risk_summary": as_text(e.get("risk_summary")),
                "safety_indicators": as_text(e.get("safety_indicators")),
                "llm_risky_operations": as_text(llm.get("risky_operations")),
                "llm_missing_checks": as_text(llm.get("missing_checks")),
                "llm_risk_summary": as_text(llm.get("risk_summary")),
                "tail_facts": as_text(e.get("tail_facts")),
                "risk_level": as_text(e.get("risk_level")),
                "confidence": as_text(e.get("confidence")),
                "function_name": as_text(e.get("function_name")),
                "called_functions": as_text(e.get("called_functions")),
                "risky_apis": as_text(e.get("risky_apis")),
                "string_literals": as_text(e.get("string_literals")),
                "lexical_digest": as_text(e.get("lexical_digest")),
                "tail_digest": as_text(e.get("tail_digest")),
                "code_metrics": [float((e.get("code_metrics") or {}).get(k, 0) or 0)
                                 for k in METRIC_KEYS],
                "raw_code": r.get("raw_code", ""),
            }
            rows.append(row)
    return rows


TEXT_COLS = ["purpose", "data_flow", "risky_operations", "missing_checks",
             "evidence_tokens", "risk_summary", "safety_indicators",
             "llm_risky_operations", "llm_missing_checks", "llm_risk_summary",
             "tail_facts", "risk_level", "confidence", "function_name",
             "called_functions", "risky_apis", "string_literals",
             "lexical_digest", "tail_digest"]


def featurize(cols, train_rows, val_rows):
    """Build a train/val feature matrix from a list of column names."""
    blocks_tr, blocks_va = [], []
    text_cols = [c for c in cols if c != "code_metrics"]
    if text_cols:
        def join(row):
            return " [SEP] ".join(f"{c}: {row[c]}" for c in text_cols)
        vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                              max_features=30000, min_df=2)
        blocks_tr.append(vec.fit_transform(join(r) for r in train_rows))
        blocks_va.append(vec.transform(join(r) for r in val_rows))
    if "code_metrics" in cols:
        sc = StandardScaler()
        blocks_tr.append(csr_matrix(sc.fit_transform([r["code_metrics"] for r in train_rows])))
        blocks_va.append(csr_matrix(sc.transform([r["code_metrics"] for r in val_rows])))
    if len(blocks_tr) == 1:
        return blocks_tr[0], blocks_va[0]
    return hstack(blocks_tr).tocsr(), hstack(blocks_va).tocsr()


def evaluate(cols, train_rows, val_rows, y_tr, y_va):
    Xtr, Xva = featurize(cols, train_rows, val_rows)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="liblinear")
    clf.fit(Xtr, y_tr)
    prob = clf.predict_proba(Xva)[:, 1]
    pred = (prob >= 0.5).astype(int)
    f1 = f1_score(y_va, pred)
    acc = accuracy_score(y_va, pred)
    return {
        "f1": f1,
        "acc": acc,
        "score": (f1 + acc) / 2,  # selection objective: F1 alone rewards
        # degenerate all-positive predictors on balanced data (devign trap)
        "precision": precision_score(y_va, pred, zero_division=0),
        "recall": recall_score(y_va, pred),
        "auc": roc_auc_score(y_va, prob),
    }


def mean_words(rows, col):
    return float(np.mean([len(r[col].split()) for r in rows])) if col != "code_metrics" else float(len(METRIC_KEYS))


def analyze(ds_dir, out_dir):
    name = os.path.basename(os.path.normpath(ds_dir))
    print(f"\n{'='*70}\nDATASET: {name}\n{'='*70}")
    train_rows = load(os.path.join(ds_dir, "train.jsonl"))
    val_rows = load(os.path.join(ds_dir, "val.jsonl"))
    y_tr = np.array([r["label"] for r in train_rows])
    y_va = np.array([r["label"] for r in val_rows])
    print(f"train={len(y_tr)} (pos {y_tr.mean():.1%})  val={len(y_va)} (pos {y_va.mean():.1%})")
    maj_acc = max(y_va.mean(), 1 - y_va.mean())
    allpos_f1 = f1_score(y_va, np.ones_like(y_va))
    print(f"majority-class accuracy baseline: {maj_acc:.4f}")
    print(f"all-positive F1 baseline:         {allpos_f1:.4f} "
          f"(any F1 below this with acc < {maj_acc:.2f} is degenerate)\n")

    # ---- Pass 1: single columns -------------------------------------------
    results = []
    candidates = TEXT_COLS + ["code_metrics"]
    for col in candidates:
        m = evaluate([col], train_rows, val_rows, y_tr, y_va)
        m.update(column=col, n_cols=1, mean_words=round(mean_words(train_rows, col), 1))
        results.append(m)
        print(f"  {col:22s} F1={m['f1']:.4f} acc={m['acc']:.4f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f} AUC={m['auc']:.4f} ~{m['mean_words']}w")

    # reference: raw_code proxy (what the transformer sees anyway)
    ref = evaluate(["raw_code"], train_rows, val_rows, y_tr, y_va)
    ref.update(column="raw_code (reference)", n_cols=1,
               mean_words=round(mean_words(train_rows, "raw_code"), 1))
    print(f"  {'raw_code (reference)':22s} F1={ref['f1']:.4f} acc={ref['acc']:.4f} "
          f"AUC={ref['auc']:.4f}")

    results.sort(key=lambda m: m["score"], reverse=True)

    # ---- Pass 2: greedy forward selection on top columns ------------------
    # Objective is (F1 + acc)/2: F1 alone is gamed by predict-all-positive on
    # near-balanced data. Near-empty columns are excluded (degenerate fits).
    print("\nGreedy forward selection (objective: (F1 + acc) / 2):")
    pool = [m["column"] for m in results
            if m["mean_words"] >= 0.1 and m["auc"] > 0.52][:GREEDY_POOL]
    chosen, best = [], None
    while len(chosen) < GREEDY_MAX:
        trial_best = None
        for col in pool:
            if col in chosen:
                continue
            m = evaluate(chosen + [col], train_rows, val_rows, y_tr, y_va)
            if trial_best is None or m["score"] > trial_best[1]["score"]:
                trial_best = (col, m)
        if best is not None and trial_best[1]["score"] <= best["score"] + 1e-4:
            print(f"  stop: adding '{trial_best[0]}' gives score={trial_best[1]['score']:.4f} (no gain)")
            break
        chosen.append(trial_best[0])
        best = trial_best[1]
        print(f"  + {trial_best[0]:22s} -> F1={best['f1']:.4f} acc={best['acc']:.4f} "
              f"P={best['precision']:.3f} R={best['recall']:.3f} AUC={best['auc']:.4f}")
    if best:
        best.update(column=" + ".join(chosen), n_cols=len(chosen),
                    mean_words=round(sum(mean_words(train_rows, c) for c in chosen), 1))
        results.insert(0, best)
    results.append(ref)

    # ---- Save --------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{name}_column_efficacy.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["column", "n_cols", "f1", "acc", "score",
                                          "precision", "recall", "auc", "mean_words"])
        w.writeheader()
        for m in results:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    print(f"\nsaved -> {out_csv}")
    return name, results, chosen, maj_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", help="dataset dir containing train.jsonl/val.jsonl")
    ap.add_argument("--all", action="store_true", help="run reveal + devign")
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "analysis_results")
    dirs = ([os.path.join(base, "explanations", "SemanticVul", d) for d in ("reveal", "devign")]
            if args.all else [args.dataset])
    if not dirs or dirs == [None]:
        ap.error("give a dataset dir or --all")
    for d in dirs:
        analyze(d, out_dir)


if __name__ == "__main__":
    main()
