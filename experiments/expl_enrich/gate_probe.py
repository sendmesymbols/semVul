"""Gate 2 (decisive, cheap): frozen-embedding channel probe for the ENRICHED
explanation channel, on both datasets.

Embeds the enriched explanation text with the project's MiniLM text encoder,
then reuses the cached GraphCodeBERT-LoRA code embeddings and asks: does
code + enriched-expl (+ qual_v2) beat code + original-expl (+ qual) in probe
ROC/PR-AUC?  If yes -> retraining the L2/L3 rungs on the enriched JSONL is
worth the GPU hours; if no -> it is not, and no amount of retraining will
make the rung climb.

  .venv/Scripts/python.exe experiments/expl_enrich/gate_probe.py
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import CACHE_DIR, EXPL_DIR, REPORTS_DIR, TEXT_ENCODER
from static_enrich import enriched_text

SEEDS = (1337, 2024, 42)


def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def qual_v2(rows):
    """code_metrics + risk_level/confidence as a feature block (label-blind)."""
    lvl = {"none": 0, "low": 1, "medium": 2, "high": 3}
    conf = {"low": 0, "medium": 1, "high": 2}
    keys = ["n_words", "n_stmts", "n_if", "n_loops", "n_switch", "n_goto",
            "n_return", "n_calls", "n_deref", "n_index", "n_alloc", "n_free",
            "n_unsafe_str", "n_bounded_copy", "truncated", "n_findings",
            "n_guards", "n_findings_tail"]
    out = []
    for r in rows:
        e = r["explanation"]
        m = e.get("code_metrics", {})
        out.append([float(m.get(k, 0)) for k in keys] +
                   [float(lvl.get(e.get("risk_level"), 0)),
                    float(conf.get(e.get("confidence"), 1)),
                    float(len(e.get("safety_indicators") or [])),
                    float(bool(e.get("tail_facts")))])
    return np.asarray(out, dtype=np.float32)


def embed_texts(texts, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(TEXT_ENCODER, device=device)
    return model.encode(texts, batch_size=256, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)


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
        idx = np.arange(len(ytr))
        for ep in range(30):
            rng = np.random.default_rng(s * 1000 + ep)
            rng.shuffle(idx)
            net.train()
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                opt.zero_grad()
                loss = lossf(net(Xtr_t[b]).squeeze(-1), ytr_t[b])
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            probs += torch.sigmoid(net(Xva_t).squeeze(-1)).cpu().numpy()
    probs /= len(SEEDS)
    return 100 * roc_auc_score(yva, probs), 100 * average_precision_score(yva, probs)


def standardize(a, b):
    mu, sd = a.mean(0, keepdims=True), a.std(0, keepdims=True) + 1e-6
    return (a - mu) / sd, (b - mu) / sd


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lines = ["# Enriched-explanation gate probe (frozen embeddings)", "",
             "Probe trained on TRAIN, ROC/PR-AUC on VAL. Same protocol as "
             "devign_channel_probe.md. 'enr' = static-v1 enriched channel.", ""]
    for ds in ("devign", "reveal"):
        print(f"\n===== {ds} =====", flush=True)
        rows = {s: load_jsonl(EXPL_DIR / ds / f"{ds}_{s}.enriched.jsonl")
                for s in ("train", "val")}
        emb, y, ids = {}, {}, {}
        for s in ("train", "val"):
            d = np.load(CACHE_DIR / f"{ds}_{s}_code_graphcodebert_lora.npz",
                        allow_pickle=True)
            emb[s] = d["embeddings"].astype(np.float32)
            ids[s] = [str(i) for i in d["sample_ids"]]
            y[s] = d["labels"].astype(np.int64)
            # sample_ids are NOT unique (845 dup rows in devign train), so align
            # positionally: the cache was built in JSONL file order.
            assert [r["sample_id"] for r in rows[s]] == ids[s], \
                f"{ds}/{s}: JSONL row order != cache order"
            assert all(int(r["label"]) == int(l) for r, l in zip(rows[s], y[s]))
        txt_orig = {s: np.load(CACHE_DIR / f"{ds}_{s}_text_minilm.npz",
                               allow_pickle=True) for s in ("train", "val")}
        # original minilm cache order should match gcb order (same builder);
        # verify via labels
        for s in ("train", "val"):
            if "sample_ids" in txt_orig[s].files:
                assert [str(i) for i in txt_orig[s]["sample_ids"]] == ids[s], \
                    "text cache misaligned"
        orig = {s: txt_orig[s]["embeddings"].astype(np.float32) for s in ("train", "val")}
        qold = {s: np.load(CACHE_DIR / f"{ds}_{s}_quality.npz")["quality"].astype(np.float32)
                for s in ("train", "val")}
        cache_enr = {s: CACHE_DIR / f"{ds}_{s}_text_minilm_enriched.npz"
                     for s in ("train", "val")}
        enr = {}
        for s in ("train", "val"):
            if cache_enr[s].exists():
                enr[s] = np.load(cache_enr[s])["embeddings"].astype(np.float32)
            else:
                texts = [enriched_text(r) for r in rows[s]]
                enr[s] = embed_texts(texts, device)
                np.savez_compressed(cache_enr[s], embeddings=enr[s],
                                    sample_ids=np.array(ids[s]))
        q2 = {s: qual_v2(rows[s]) for s in ("train", "val")}

        sets = {
            "code_L1": (emb["train"], emb["val"]),
            "code_L1 + expl_orig": (np.hstack([emb["train"], orig["train"]]),
                                    np.hstack([emb["val"], orig["val"]])),
            "code_L1 + expl_enr": (np.hstack([emb["train"], enr["train"]]),
                                   np.hstack([emb["val"], enr["val"]])),
            "expl_orig only": (orig["train"], orig["val"]),
            "expl_enr only": (enr["train"], enr["val"]),
            "qual_v2 only": (q2["train"], q2["val"]),
            "code_L1 + qual_v2": (np.hstack([emb["train"], q2["train"]]),
                                  np.hstack([emb["val"], q2["val"]])),
            "code_L1 + expl_enr + qual_v2": (
                np.hstack([emb["train"], enr["train"], q2["train"]]),
                np.hstack([emb["val"], enr["val"], q2["val"]])),
            "code_L1 + expl_orig + qual (old full)": (
                np.hstack([emb["train"], orig["train"], qold["train"]]),
                np.hstack([emb["val"], orig["val"], qold["val"]])),
        }
        lines += [f"## {ds}", "",
                  "| Feature set | dim | LR ROC | LR PR | MLP ROC | MLP PR |",
                  "|---|---|---|---|---|---|"]
        for label, (Xtr, Xva) in sets.items():
            Xtr_s, Xva_s = standardize(Xtr, Xva)
            lr_r, lr_p = lr_auc(Xtr_s, y["train"], Xva_s, y["val"])
            ml_r, ml_p = mlp_auc(Xtr_s, y["train"], Xva_s, y["val"])
            print(f"{label:42s} dim={Xtr.shape[1]:5d} | LR roc={lr_r:5.2f} "
                  f"pr={lr_p:5.2f} | MLP roc={ml_r:5.2f} pr={ml_p:5.2f}", flush=True)
            lines.append(f"| {label} | {Xtr.shape[1]} | {lr_r:.2f} | {lr_p:.2f} "
                         f"| {ml_r:.2f} | {ml_p:.2f} |")
        lines.append("")
    out = REPORTS_DIR / "enrich_gate_probe.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
