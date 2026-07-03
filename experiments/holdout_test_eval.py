"""SELF-CONTAINED, SINGLE-FILE held-out TEST-fold builder + evaluator.

Delete this file and nothing else breaks. It writes only two artifacts, both
optional and deletable:
    experiments/runs/devign_holdout_split.json   (sample_id -> fold assignment)
    experiments/reports/devign_holdout_test.md   (metrics table)

WHY THIS EXISTS
---------------
FuSEVul reports on the Devign *test* fold of a stratified 80/10/10 split.
Your repo only ships train+val, and FuSEVul never released its (normalized)
test fold -- so its EXACT test fold cannot be reconstructed. This script does
the next best, fully reproducible thing: a fixed-seed stratified 80/10/10
re-partition of ALL your cached samples into train/val/test, then reports on a
TEST fold that is never seen during head training OR threshold selection.

HONESTY CAVEAT (state this in any writeup):
    This is NOT FuSEVul's exact test fold. It is a seeded stratified 80/10/10
    split of the released Devign train+val pool. Comparison to FuSEVul is
    "same dataset, same standard protocol, different random split."

It uses ONLY cached embeddings (no downloads, no models, no explanations
regeneration). Threshold is selected on VAL, metrics reported on TEST.

Run:
    .venv/Scripts/python.exe experiments/holdout_test_eval.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

# ---------------------------------------------------------------- config
CACHE = Path(r"D:\Projects\SemVul\experiments\cache")
RUNS = Path(r"D:\Projects\SemVul\experiments\runs")
REPORTS = Path(r"D:\Projects\SemVul\experiments\reports")
DATASET = "devign"
SPLIT_SEED = 2025          # fixes the 80/10/10 partition; change -> new split
FRACS = (0.80, 0.10, 0.10)  # train / val / test
HEAD_SEEDS = (1337, 2024, 42)
EPOCHS = 30
FUSEVUL = {"accuracy": 60.39, "f1": 55.91}   # Devign test-fold targets

# feature configs to evaluate on the held-out test fold
CONFIGS = {
    "code_only (GCB)":        ["code1"],
    "code + qual":            ["code1", "qual"],
    "full (code+expl+qual)":  ["code1", "expl", "qual"],
}


# ---------------------------------------------------------------- data load
def _load_channel(split, name, key="embeddings"):
    d = np.load(CACHE / f"{DATASET}_{split}_{name}.npz", allow_pickle=True)
    return d


def load_pool():
    """Combine cached train+val into one pool keyed by sample_id.
    Returns dict channel-> (N,d) arrays aligned to `sids`, plus labels."""
    channels = {}
    sids_all, y_all = [], []
    per_split = {}
    for split in ("train", "val"):
        gcb = _load_channel(split, "code_graphcodebert_lora")
        uni = _load_channel(split, "code_unixcoder_frozen")
        txt = _load_channel(split, "text_minilm")
        qual = np.load(CACHE / f"{DATASET}_{split}_quality.npz")["quality"]
        sids = gcb["sample_ids"].astype(str)
        y = gcb["labels"].astype(np.int64)
        n = len(sids)
        assert len(uni["embeddings"]) == len(txt["embeddings"]) == len(qual) == n, \
            f"{split}: channel length mismatch"
        per_split[split] = dict(
            code1=gcb["embeddings"].astype(np.float32),
            code2=uni["embeddings"].astype(np.float32),
            expl=txt["embeddings"].astype(np.float32),
            qual=qual.astype(np.float32),
            sids=sids, y=y,
        )
    # concat train+val, then dedupe by sample_id (keep first)
    for ch in ("code1", "code2", "expl", "qual"):
        channels[ch] = np.concatenate([per_split["train"][ch], per_split["val"][ch]], 0)
    sids_all = np.concatenate([per_split["train"]["sids"], per_split["val"]["sids"]])
    y_all = np.concatenate([per_split["train"]["y"], per_split["val"]["y"]])

    _, uniq_idx = np.unique(sids_all, return_index=True)
    uniq_idx.sort()
    dropped = len(sids_all) - len(uniq_idx)
    if dropped:
        print(f"[dedupe] dropped {dropped} duplicate sample_ids across train/val")
    for ch in channels:
        channels[ch] = channels[ch][uniq_idx]
    sids_all = sids_all[uniq_idx]
    y_all = y_all[uniq_idx]
    return channels, sids_all, y_all


# ---------------------------------------------------------------- split
def stratified_3way(y, fracs, seed):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * fracs[0]))
        n_va = int(round(n * fracs[1]))
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    tr = np.concatenate(tr); va = np.concatenate(va); te = np.concatenate(te)
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    # disjointness guarantee
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 \
        and len(set(va) & set(te)) == 0, "folds overlap!"
    return tr, va, te


# ---------------------------------------------------------------- model / eval
def train_probs(Xtr, ytr, Xva, Xte):
    import torch, torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = Xtr.shape[1]
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                         dtype=torch.float32, device=device)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    Xva_t = torch.from_numpy(Xva).float().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)
    pv = np.zeros(len(Xva), np.float64); pt = np.zeros(len(Xte), np.float64)
    for s in HEAD_SEEDS:
        torch.manual_seed(s)
        net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3),
                            nn.Linear(256, 1)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        idx = np.arange(len(ytr)); bs = 256
        for ep in range(EPOCHS):
            rng = np.random.default_rng(s * 1000 + ep); rng.shuffle(idx)
            net.train()
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]
                opt.zero_grad()
                loss = lossf(net(Xtr_t[b]).squeeze(-1), ytr_t[b])
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv += torch.sigmoid(net(Xva_t).squeeze(-1)).cpu().numpy()
            pt += torch.sigmoid(net(Xte_t).squeeze(-1)).cpu().numpy()
    return pv / len(HEAD_SEEDS), pt / len(HEAD_SEEDS)


def metrics_at(thr, p, y):
    yh = (p >= thr).astype(int)
    return dict(threshold=round(float(thr), 3),
                accuracy=100 * accuracy_score(y, yh),
                precision=100 * precision_score(y, yh, zero_division=0),
                recall=100 * recall_score(y, yh, zero_division=0),
                f1=100 * f1_score(y, yh, zero_division=0),
                roc_auc=100 * roc_auc_score(y, p),
                pr_auc=100 * average_precision_score(y, p))


def best_thr(p, y, objective):
    grid = np.linspace(0.05, 0.95, 91); best, bs = 0.5, -1
    for t in grid:
        yh = (p >= t).astype(int)
        if objective == "f1":
            sc = f1_score(y, yh, zero_division=0)
        else:
            tp = ((yh == 1) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
            fp = ((yh == 1) & (y == 0)).sum(); fn = ((yh == 0) & (y == 1)).sum()
            sc = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
        if sc > bs:
            bs, best = sc, float(t)
    return best


def main():
    channels, sids, y = load_pool()
    tr, va, te = stratified_3way(y, FRACS, SPLIT_SEED)
    print(f"pool n={len(y)} pos={y.mean()*100:.1f}%  ->  "
          f"train {len(tr)} ({y[tr].mean()*100:.1f}%)  "
          f"val {len(va)} ({y[va].mean()*100:.1f}%)  "
          f"test {len(te)} ({y[te].mean()*100:.1f}%)\n")

    # persist split assignment by sample_id (reproducible, reusable)
    fold = {}
    for name, arr in (("train", tr), ("val", va), ("test", te)):
        for i in arr:
            fold[str(sids[i])] = name
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{DATASET}_holdout_split.json").write_text(
        json.dumps({"seed": SPLIT_SEED, "fracs": FRACS, "fold": fold}, indent=0),
        encoding="utf-8")

    def mark(v, k):
        d = v - FUSEVUL[k]
        return f"WIN +{d:.2f}" if d > 0.05 else (f"LOSE {d:.2f}" if d < -0.05 else "TIE")

    lines = ["# Held-out TEST-fold evaluation -- devign", "",
             f"Seeded stratified 80/10/10 (seed={SPLIT_SEED}) over the released "
             f"train+val pool (n={len(y)}). Threshold picked on VAL, reported on TEST. "
             "NOT FuSEVul's exact fold -- same dataset & protocol, different split.", "",
             f"FuSEVul (Devign test): Acc={FUSEVUL['accuracy']}, F1={FUSEVUL['f1']}", "",
             "| Config | Policy | Acc | F1 | Prec | Rec | ROC-AUC | PR-AUC | Acc? | F1? |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    print(f"{'config':24s}{'policy':13s}{'Acc':>7}{'F1':>7}{'P':>7}{'R':>7}{'ROC':>7}{'PR':>7}")
    print("-" * 78)
    for cname, chs in CONFIGS.items():
        Xtr = np.concatenate([channels[c][tr] for c in chs], 1)
        Xva = np.concatenate([channels[c][va] for c in chs], 1)
        Xte = np.concatenate([channels[c][te] for c in chs], 1)
        mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        pv, pt = train_probs(Xtr, y[tr], Xva, Xte)
        t_bal = best_thr(pv, y[va], "bal"); t_f1 = best_thr(pv, y[va], "f1")
        for pol, thr in (("fixed_0.5", 0.5), ("max_bal_acc", t_bal), ("max_f1", t_f1)):
            m = metrics_at(thr, pt, y[te])
            print(f"{cname:24s}{pol:13s}{m['accuracy']:7.2f}{m['f1']:7.2f}"
                  f"{m['precision']:7.2f}{m['recall']:7.2f}{m['roc_auc']:7.2f}{m['pr_auc']:7.2f}")
            lines.append(f"| {cname} | {pol} | {m['accuracy']:.2f} | {m['f1']:.2f} | "
                         f"{m['precision']:.2f} | {m['recall']:.2f} | {m['roc_auc']:.2f} | "
                         f"{m['pr_auc']:.2f} | {mark(m['accuracy'],'accuracy')} | {mark(m['f1'],'f1')} |")
        print()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{DATASET}_holdout_test.md").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
    print(f"wrote {RUNS / (DATASET + '_holdout_split.json')}")
    print(f"wrote {REPORTS / (DATASET + '_holdout_test.md')}")


if __name__ == "__main__":
    main()
