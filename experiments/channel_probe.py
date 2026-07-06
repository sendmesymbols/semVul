"""STANDALONE, READ-ONLY channel-isolation probe.

Question it answers: does each input channel carry standalone label signal,
and does adding the explanation help or hurt on top of code?

It ONLY reads the cached .npz embeddings (never re-encodes, never touches
data/ or the runs/). For every feature set it trains two probes on TRAIN and
reports threshold-independent ROC-AUC / PR-AUC on VAL:

  * LR  : logistic regression (linear signal, deterministic)
  * MLP : 1-hidden-layer net matching the project head dims, 3-seed ensemble
          (nonlinear signal, directly comparable to the pipeline's ~63 AUC)

Run:  .venv/Scripts/python.exe experiments/channel_probe.py
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# Resolve relative to this file so the probe works regardless of repo location.
_HERE = Path(__file__).resolve().parent
CACHE = _HERE / "cache"
REPORT = _HERE / "reports" / "devign_channel_probe.md"
DATASET = "devign"
SEEDS = (1337, 2024, 42)


def _load(name: str, key: str = "embeddings"):
    d = np.load(CACHE / name, allow_pickle=True)
    return d[key], (d["sample_ids"] if "sample_ids" in d.files else None), \
        (d["labels"] if "labels" in d.files else None)


def load_split(split: str):
    """Return dict of aligned channels + labels for a split, aligned by sample_id."""
    gcb, ids_g, y_g = _load(f"{DATASET}_{split}_code_graphcodebert_lora.npz")
    uni, ids_u, _ = _load(f"{DATASET}_{split}_code_unixcoder_frozen.npz")
    txt, ids_t, y_t = _load(f"{DATASET}_{split}_text_minilm.npz")
    qual = np.load(CACHE / f"{DATASET}_{split}_quality.npz")["quality"]

    # sanity: all channels same length
    n = len(gcb)
    assert len(uni) == len(txt) == len(qual) == n, "channel length mismatch"

    # align by sample_ids if present and not already identical
    def aligned(a, b):
        return a is not None and b is not None and np.array_equal(a, b)
    if not (aligned(ids_g, ids_u) and aligned(ids_g, ids_t)):
        # build order by gcb's ids
        def order_to(ids_ref, ids_other, arr):
            pos = {s: i for i, s in enumerate(ids_other)}
            idx = np.array([pos[s] for s in ids_ref])
            return arr[idx]
        if ids_u is not None:
            uni = order_to(ids_g, ids_u, uni)
        if ids_t is not None:
            txt = order_to(ids_g, ids_t, txt)
        # quality has no ids -> assume it follows gcb order (same builder)
    y = y_g if y_g is not None else y_t
    y = y.astype(np.int64)
    return {
        "code1": gcb.astype(np.float32),          # GraphCodeBERT-LoRA (L1 code)
        "code2": uni.astype(np.float32),           # UniXcoder-frozen (L2 adds this)
        "expl":  txt.astype(np.float32),           # MiniLM explanation embedding
        "qual":  qual.astype(np.float32),          # 22 handcrafted quality feats
    }, y


def standardize(train_x, val_x):
    mu = train_x.mean(0, keepdims=True)
    sd = train_x.std(0, keepdims=True) + 1e-6
    return (train_x - mu) / sd, (val_x - mu) / sd


def build(sets: dict, feats: list):
    return np.concatenate([sets[f] for f in feats], axis=1)


def lr_auc(Xtr, ytr, Xva, yva):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    return 100 * roc_auc_score(yva, p), 100 * average_precision_score(yva, p)


def mlp_auc(Xtr, ytr, Xva, yva):
    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = Xtr.shape[1]
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                         dtype=torch.float32, device=device)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    Xva_t = torch.from_numpy(Xva).float().to(device)
    probs = np.zeros(len(yva), dtype=np.float64)
    for s in SEEDS:
        torch.manual_seed(s)
        net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3),
                            nn.Linear(256, 1)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        bs = 256
        idx = np.arange(len(ytr))
        for ep in range(30):
            rng = np.random.default_rng(s * 1000 + ep)
            rng.shuffle(idx)
            net.train()
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                opt.zero_grad()
                logit = net(Xtr_t[b]).squeeze(-1)
                loss = lossf(logit, ytr_t[b])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            probs += torch.sigmoid(net(Xva_t).squeeze(-1)).cpu().numpy()
    probs /= len(SEEDS)
    return 100 * roc_auc_score(yva, probs), 100 * average_precision_score(yva, probs)


def main():
    tr, ytr = load_split("train")
    va, yva = load_split("val")
    print(f"train n={len(ytr)}  pos={ytr.mean()*100:.1f}%   "
          f"val n={len(yva)}  pos={yva.mean()*100:.1f}%")
    print(f"dims: code1={tr['code1'].shape[1]} code2={tr['code2'].shape[1]} "
          f"expl={tr['expl'].shape[1]} qual={tr['qual'].shape[1]}\n")

    # (label, feature-list). codeL1 = code1 ; codeL2 = code1+code2
    FEATURE_SETS = [
        ("code_L1 (GCB)",            ["code1"]),
        ("code_L2 (GCB+UniX)",       ["code1", "code2"]),
        ("expl_only",                ["expl"]),
        ("qual_only",                ["qual"]),
        ("code_L1 + expl",           ["code1", "expl"]),
        ("code_L1 + qual",           ["code1", "qual"]),
        ("expl + qual",              ["expl", "qual"]),
        ("code_L1 + expl + qual (=full L1)", ["code1", "expl", "qual"]),
        ("code_L2 + expl + qual (=full L2)", ["code1", "code2", "expl", "qual"]),
    ]

    rows = []
    for label, feats in FEATURE_SETS:
        Xtr = build(tr, feats)
        Xva = build(va, feats)
        Xtr, Xva = standardize(Xtr, Xva)
        lr_roc, lr_pr = lr_auc(Xtr, ytr, Xva, yva)
        ml_roc, ml_pr = mlp_auc(Xtr, ytr, Xva, yva)
        rows.append((label, Xtr.shape[1], lr_roc, lr_pr, ml_roc, ml_pr))
        print(f"{label:38s} dim={Xtr.shape[1]:5d} | "
              f"LR roc={lr_roc:5.2f} pr={lr_pr:5.2f} | "
              f"MLP roc={ml_roc:5.2f} pr={ml_pr:5.2f}")

    # sort by MLP ROC-AUC for the report
    rows_sorted = sorted(rows, key=lambda r: r[4], reverse=True)
    lines = ["# Channel-isolation probe -- devign (READ-ONLY, cached embeddings)", "",
             f"train n={len(ytr)} (pos {ytr.mean()*100:.1f}%), "
             f"val n={len(yva)} (pos {yva.mean()*100:.1f}%). "
             "Probes trained on TRAIN, AUC reported on VAL. Threshold-independent.", "",
             "| Feature set | dim | LR ROC | LR PR | MLP ROC | MLP PR |",
             "|---|---|---|---|---|---|"]
    for label, dim, lr_roc, lr_pr, ml_roc, ml_pr in rows_sorted:
        lines.append(f"| {label} | {dim} | {lr_roc:.2f} | {lr_pr:.2f} | "
                     f"{ml_roc:.2f} | {ml_pr:.2f} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
