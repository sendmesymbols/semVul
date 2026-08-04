"""
reveal_cl.py — Column-value analysis for the Reveal SemanticVul explanation dataset.

Question: which explanation *columns* actually add predictive signal for `label`,
and do the long free-text columns hurt?

We never touch `raw_code`. `label` is the target. Everything else in `explanation`
is a candidate input column. We measure value three ways:

  1. Univariate signal   — per numeric/count/length feature: AUROC, AUPRC-lift, MI.
  2. Categorical signal   — variance / constant detection + mutual information.
  3. Learned-channel value — embed each text/code column with the real encoders
                             (codet5p for code-ish, roberta for NL), fit a cheap
                             linear probe on train, score on val (AUROC + AUPRC),
                             and measure *incremental* lift over the structured
                             baseline. This is what tells us whether a column is
                             worth a whole encoder branch.

Models (as specified):
    CODE_MODEL = "Salesforce/codet5p-110m-embedding"
    TEXT_MODEL = "roberta-base"

Usage:
    .venv\\Scripts\\python.exe reveal_cl.py            # full analysis (caches embeddings)
    .venv\\Scripts\\python.exe reveal_cl.py --no-embed  # structured-only, fast
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# HF caches must be set before transformers import; reuse the project layout.
import src.config as C  # noqa: F401  (side effect: sets HF_HOME etc.)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "explanations" / "SemanticVul" / "reveal" / "ACTIVE"
TRAIN = DATA / "reveal_train.jsonl"
VAL = DATA / "reveal_val.jsonl"
CACHE = C.CACHE_DIR / "reveal_cl"
CACHE.mkdir(parents=True, exist_ok=True)

CODE_MODEL = "Salesforce/codet5p-110m-embedding"
TEXT_MODEL = "roberta-base"

# ---- column groups -------------------------------------------------------
CODE_METRICS = [
    "n_words", "n_stmts", "n_if", "n_loops", "n_switch", "n_goto", "n_return",
    "n_calls", "n_deref", "n_index", "n_alloc", "n_free", "n_unsafe_str",
    "n_bounded_copy", "truncated", "n_findings", "n_guards", "n_findings_tail",
]
# list columns -> we derive a length feature for each
LIST_FIELDS = [
    "risky_operations", "missing_checks", "evidence_tokens", "safety_indicators",
    "called_functions", "risky_apis", "string_literals",
]
# free-text columns -> we derive char/word lengths AND embed them
TEXT_FIELDS = ["purpose", "data_flow", "risk_summary",
               "lexical_digest", "tail_facts", "tail_digest"]
CATEG_FIELDS = ["risk_level", "confidence", "enrich", "real_enrich"]

# which encoder each embeddable column belongs to
NL_COLS = ["purpose", "data_flow", "risk_summary", "concat_nl"]        # roberta
CODE_COLS = ["lexical_digest", "evidence_joined", "called_joined"]     # codet5p


# ==========================================================================
# Loading + feature extraction
# ==========================================================================
def load_records(path: Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _txt(v) -> str:
    return v if isinstance(v, str) else ""


def _list(v) -> list:
    return v if isinstance(v, list) else []


def extract(recs: list[dict]):
    """Return (y, structured_matrix, feat_names, text_cols_dict)."""
    y = np.array([int(r["label"]) for r in recs], dtype=np.int64)

    feat_names: list[str] = []
    rows: list[list[float]] = []
    text_cols = {c: [] for c in TEXT_FIELDS}
    text_cols["concat_nl"] = []
    text_cols["evidence_joined"] = []
    text_cols["called_joined"] = []
    categ = {c: [] for c in CATEG_FIELDS}

    for r in recs:
        e = r.get("explanation", {}) or {}
        cm = e.get("code_metrics", {}) or {}
        row: list[float] = []

        # 18 numeric code metrics
        for k in CODE_METRICS:
            row.append(float(cm.get(k, 0) or 0))

        # list-length features
        for lf in LIST_FIELDS:
            row.append(float(len(_list(e.get(lf)))))

        # text length features (chars + words)
        for tf in TEXT_FIELDS:
            s = _txt(e.get(tf))
            text_cols[tf].append(s)
            row.append(float(len(s)))                 # chars
            row.append(float(len(s.split())))         # words

        rows.append(row)

        # composite text channels
        nl = " ".join(_txt(e.get(f)) for f in ("purpose", "data_flow", "risk_summary"))
        text_cols["concat_nl"].append(nl.strip())
        text_cols["evidence_joined"].append(" \n ".join(_list(e.get("evidence_tokens"))))
        text_cols["called_joined"].append(" ".join(_list(e.get("called_functions"))))

        for cf in CATEG_FIELDS:
            categ[cf].append(str(e.get(cf, "")))

    if not feat_names:
        feat_names = list(CODE_METRICS)
        feat_names += [f"len_{lf}" for lf in LIST_FIELDS]
        for tf in TEXT_FIELDS:
            feat_names += [f"{tf}_chars", f"{tf}_words"]

    X = np.array(rows, dtype=np.float32)
    return y, X, feat_names, text_cols, categ


# ==========================================================================
# Univariate signal
# ==========================================================================
def univariate(y, X, names):
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import mutual_info_classif

    base_rate = y.mean()
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    out = []
    for j, nm in enumerate(names):
        col = X[:, j]
        if np.ptp(col) == 0:
            out.append((nm, 0.5, 0.0, mi[j], col.mean(), "CONSTANT"))
            continue
        auc = roc_auc_score(y, col)
        direction = "+" if auc >= 0.5 else "-"
        auc_abs = max(auc, 1 - auc)
        # mean in pos vs neg for interpretability
        mp, mn = col[y == 1].mean(), col[y == 0].mean()
        out.append((nm, auc_abs, mp - mn, mi[j], (mp, mn), direction))
    out.sort(key=lambda t: t[1], reverse=True)
    return out, base_rate


def categorical(y, categ):
    from sklearn.metrics import roc_auc_score
    rows = []
    for cf, vals in categ.items():
        uniq = sorted(set(vals))
        if len(uniq) <= 1:
            rows.append((cf, uniq, "CONSTANT", None))
            continue
        # per-value positive rate + a one-vs-rest AUC using value pos-rate as score
        rate = {u: y[[i for i, v in enumerate(vals) if v == u]].mean() for u in uniq}
        score = np.array([rate[v] for v in vals])
        try:
            auc = roc_auc_score(y, score)
        except ValueError:
            auc = float("nan")
        counts = {u: vals.count(u) for u in uniq}
        rows.append((cf, uniq, {u: (counts[u], round(rate[u], 3)) for u in uniq}, auc))
    return rows


# ==========================================================================
# Embedding channels
# ==========================================================================
def _mean_pool(last_hidden, mask):
    import torch
    m = mask.unsqueeze(-1).float()
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def embed_column(texts: list[str], model_id: str, tag: str,
                 batch_size: int = 64, max_len: int = 256) -> np.ndarray:
    """Embed a text column with the given HF model. Cached to disk by tag."""
    import torch
    cache = CACHE / f"emb_{tag}.npy"
    if cache.exists():
        return np.load(cache)

    from transformers import AutoTokenizer, AutoModel, T5EncoderModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_codet5p = "codet5p" in model_id
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=is_codet5p)
    if is_codet5p:
        # The custom CodeT5pEmbedding* classes break on transformers 5.x; load the
        # underlying T5 encoder and mean-pool -> same code encoder, 768-dim.
        model = T5EncoderModel.from_pretrained(model_id).to(device).eval()
    else:
        model = AutoModel.from_pretrained(model_id).to(device).eval()
    if device == "cuda":
        model = model.half()

    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t if t else " " for t in texts[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            out = model(**enc)
            if torch.is_tensor(out):                       # codet5p returns emb tensor
                pooled = out
            elif hasattr(out, "last_hidden_state"):
                pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            else:
                h = out[0]
                pooled = _mean_pool(h, enc["attention_mask"]) if h.ndim == 3 else h
            embs.append(pooled.float().cpu().numpy())
            print(f"\r  [{tag}] {min(i + batch_size, len(texts))}/{len(texts)}",
                  end="", flush=True)
    print()
    arr = np.concatenate(embs, 0).astype(np.float32)
    np.save(cache, arr)
    del model, tok
    if device == "cuda":
        torch.cuda.empty_cache()
    return arr


def probe(Xtr, ytr, Xva, yva):
    """Balanced logistic probe. Returns (auroc, auprc, ap_lift)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xva))[:, 1]
    ap = average_precision_score(yva, p)
    return roc_auc_score(yva, p), ap, ap / yva.mean()


# ==========================================================================
# Main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true", help="skip encoder channels")
    args = ap.parse_args()

    print("Loading ...")
    tr, va = load_records(TRAIN), load_records(VAL)
    ytr, Xtr, names, txt_tr, cat_tr = extract(tr)
    yva, Xva, _, txt_va, cat_va = extract(va)

    n_pos_tr, n_pos_va = int(ytr.sum()), int(yva.sum())
    print(f"\ntrain: {len(ytr):5d}  pos={n_pos_tr:4d} ({ytr.mean():.2%})  "
          f"imbalance={((len(ytr)-n_pos_tr)/max(n_pos_tr,1)):.1f}:1")
    print(f"val:   {len(yva):5d}  pos={n_pos_va:4d} ({yva.mean():.2%})  "
          f"imbalance={((len(yva)-n_pos_va)/max(n_pos_va,1)):.1f}:1")

    report = {"class_balance": {
        "train": {"n": len(ytr), "pos": n_pos_tr, "pos_rate": float(ytr.mean())},
        "val": {"n": len(yva), "pos": n_pos_va, "pos_rate": float(yva.mean())},
    }}

    # ---- 1. univariate structured signal --------------------------------
    uni, base = univariate(ytr, Xtr, names)
    print(f"\n{'='*72}\nUNIVARIATE SIGNAL (train, {len(names)} structured features)\n{'='*72}")
    print(f"{'feature':22s} {'AUC':>6s} {'MI':>7s} {'dir':>3s}  pos_mean / neg_mean")
    for nm, auc, _delta, mi, means, d in uni:
        mtxt = "const" if means == "CONSTANT" or d == "CONSTANT" else \
               (f"{means[0]:8.2f} / {means[1]:8.2f}" if isinstance(means, tuple) else "")
        print(f"{nm:22s} {auc:6.3f} {mi:7.4f} {d:>3s}  {mtxt}")
    report["univariate"] = [
        {"feature": nm, "auc": round(a, 4), "mi": round(m, 5), "direction": d}
        for nm, a, _dl, m, _mn, d in uni
    ]

    # ---- 2. categorical -------------------------------------------------
    cats = categorical(ytr, cat_tr)
    print(f"\n{'='*72}\nCATEGORICAL COLUMNS (train)\n{'='*72}")
    for cf, uniq, info, auc in cats:
        if info == "CONSTANT":
            print(f"{cf:14s} CONSTANT -> zero value  (only: {uniq})")
        else:
            print(f"{cf:14s} AUC={auc:.3f}  {info}")
    report["categorical"] = [
        {"field": cf, "constant": info == "CONSTANT",
         "auc": (None if auc is None else round(auc, 4)),
         "levels": (uniq if info == "CONSTANT" else info)}
        for cf, uniq, info, auc in cats
    ]

    # ---- 3. structured baseline model -----------------------------------
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, l2_regularization=1.0,
        class_weight="balanced", random_state=0)
    hgb.fit(Xtr, ytr)
    pv = hgb.predict_proba(Xva)[:, 1]
    base_auc = roc_auc_score(yva, pv)
    base_ap = average_precision_score(yva, pv)
    print(f"\n{'='*72}\nSTRUCTURED BASELINE (all {len(names)} numeric cols, HGB, val)\n{'='*72}")
    print(f"AUROC={base_auc:.4f}   AUPRC={base_ap:.4f}  "
          f"(random AUPRC={yva.mean():.4f}, lift={base_ap/yva.mean():.2f}x)")
    report["structured_baseline"] = {
        "auroc": round(base_auc, 4), "auprc": round(base_ap, 4),
        "auprc_lift": round(base_ap / yva.mean(), 2)}

    # ---- 4. learned text/code channels ----------------------------------
    if not args.no_embed:
        print(f"\n{'='*72}\nLEARNED CHANNELS (probe on val: standalone + incremental)\n{'='*72}")
        print(f"{'column':18s} {'model':8s} {'AUROC':>6s} {'AUPRC':>6s} "
              f"{'lift':>5s} | {'+struct AUROC':>13s} {'+struct AUPRC':>13s} {'d_lift':>6s}")
        chan = []
        for col in NL_COLS + CODE_COLS:
            model_id = TEXT_MODEL if col in NL_COLS else CODE_MODEL
            short = "roberta" if col in NL_COLS else "codet5p"
            e_tr = embed_column(txt_tr[col], model_id, f"{col}_tr_{short}")
            e_va = embed_column(txt_va[col], model_id, f"{col}_va_{short}")
            a, p, lift = probe(e_tr, ytr, e_va, yva)
            # incremental over structured
            ca, cp, clift = probe(np.hstack([Xtr, e_tr]), ytr, np.hstack([Xva, e_va]), yva)
            print(f"{col:18s} {short:8s} {a:6.3f} {p:6.3f} {lift:5.2f} | "
                  f"{ca:13.3f} {cp:13.3f} {cp/yva.mean()-base_ap/yva.mean():+6.2f}")
            chan.append({"column": col, "model": short,
                         "standalone_auroc": round(a, 4), "standalone_auprc": round(p, 4),
                         "with_struct_auroc": round(ca, 4), "with_struct_auprc": round(cp, 4)})
        report["learned_channels"] = chan
        print(f"\n(reference: structured-only AUPRC={base_ap:.3f}, "
              f"AUPRC lift over random = {base_ap/yva.mean():.2f}x)")

    out = CACHE / "reveal_cl_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport written: {out}")


if __name__ == "__main__":
    main()
