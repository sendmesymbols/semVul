"""SELF-CONTAINED, SINGLE-FILE held-out TEST-fold builder + evaluator.

Delete this file and nothing else breaks. It writes only two artifacts, both
optional and deletable:
    experiments/runs/devign_holdout_split.json   (sample_id -> fold, first seed)
    experiments/reports/devign_holdout_test.md    (aggregated metrics table)

WHY THIS EXISTS
---------------
FuSEVul reports on the Devign *test* fold of a stratified 80/10/10 split.
Your repo only ships train+val, and FuSEVul never released its (normalized)
test fold -- so its EXACT test fold cannot be reconstructed. This does the next
best, fully reproducible thing: for EACH of several fixed seeds it makes a
stratified 80/10/10 train/val/test partition of ALL cached samples, trains the
head on train, picks the threshold on val, and reports on TEST. Aggregating
over seeds gives a mean +/- std baseline that is robust to split luck.

HONESTY CAVEAT (state this in any writeup): NOT FuSEVul's exact test fold --
same dataset & standard protocol, different random split. Uses ONLY cached
embeddings (no downloads, no models, no explanation regeneration).

Run:  .venv/Scripts/python.exe experiments/holdout_test_eval.py
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

CACHE = Path(r"D:\Projects\SemVul\experiments\cache")
RUNS = Path(r"D:\Projects\SemVul\experiments\runs")
REPORTS = Path(r"D:\Projects\SemVul\experiments\reports")
DATASET = "devign"
SPLIT_SEEDS = (2025, 7, 42, 1337, 99)   # one 80/10/10 partition each -> mean/std
FRACS = (0.80, 0.10, 0.10)
HEAD_SEEDS = (1337, 2024, 42)
EPOCHS = 30
FUSEVUL = {"accuracy": 60.39, "f1": 55.91}

CONFIGS = {
    "code_only (GCB)":        ["code1"],
    "code + qual":            ["code1", "qual"],
    "full (code+expl+qual)":  ["code1", "expl", "qual"],
}
POLICIES = ("fixed_0.5", "max_bal_acc", "max_f1")


# ---------------------------------------------------------------- data load
def load_pool():
    channels = {}
    per_split = {}
    for split in ("train", "val"):
        gcb = np.load(CACHE / f"{DATASET}_{split}_code_graphcodebert_lora.npz", allow_pickle=True)
        uni = np.load(CACHE / f"{DATASET}_{split}_code_unixcoder_frozen.npz", allow_pickle=True)
        txt = np.load(CACHE / f"{DATASET}_{split}_text_minilm.npz", allow_pickle=True)
        qual = np.load(CACHE / f"{DATASET}_{split}_quality.npz")["quality"]
        sids = gcb["sample_ids"].astype(str)
        n = len(sids)
        assert len(uni["embeddings"]) == len(txt["embeddings"]) == len(qual) == n
        per_split[split] = dict(
            code1=gcb["embeddings"].astype(np.float32),
            code2=uni["embeddings"].astype(np.float32),
            expl=txt["embeddings"].astype(np.float32),
            qual=qual.astype(np.float32),
            sids=sids, y=gcb["labels"].astype(np.int64))
    for ch in ("code1", "code2", "expl", "qual"):
        channels[ch] = np.concatenate([per_split["train"][ch], per_split["val"][ch]], 0)
    sids = np.concatenate([per_split["train"]["sids"], per_split["val"]["sids"]])
    y = np.concatenate([per_split["train"]["y"], per_split["val"]["y"]])
    _, uniq = np.unique(sids, return_index=True); uniq.sort()
    dropped = len(sids) - len(uniq)
    if dropped:
        print(f"[dedupe] dropped {dropped} duplicate sample_ids across train/val")
    for ch in channels:
        channels[ch] = channels[ch][uniq]
    return channels, sids[uniq], y[uniq]


def stratified_3way(y, fracs, seed):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]; rng.shuffle(idx); n = len(idx)
        n_tr = int(round(n * fracs[0])); n_va = int(round(n * fracs[1]))
        tr.append(idx[:n_tr]); va.append(idx[n_tr:n_tr + n_va]); te.append(idx[n_tr + n_va:])
    tr = np.concatenate(tr); va = np.concatenate(va); te = np.concatenate(te)
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    return tr, va, te


def train_probs(Xtr, ytr, Xva, Xte):
    import torch, torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = Xtr.shape[1]
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())],
                         dtype=torch.float32, device=device)
    Xtr_t = torch.from_numpy(Xtr).float().to(device); ytr_t = torch.from_numpy(ytr).float().to(device)
    Xva_t = torch.from_numpy(Xva).float().to(device); Xte_t = torch.from_numpy(Xte).float().to(device)
    pv = np.zeros(len(Xva)); pt = np.zeros(len(Xte))
    for s in HEAD_SEEDS:
        torch.manual_seed(s)
        net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 1)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        idx = np.arange(len(ytr)); bs = 256
        for ep in range(EPOCHS):
            rng = np.random.default_rng(s * 1000 + ep); rng.shuffle(idx)
            net.train()
            for i in range(0, len(idx), bs):
                b = idx[i:i + bs]; opt.zero_grad()
                lossf(net(Xtr_t[b]).squeeze(-1), ytr_t[b]).backward(); opt.step()
        net.eval()
        import torch as _t
        with _t.no_grad():
            pv += _t.sigmoid(net(Xva_t).squeeze(-1)).cpu().numpy()
            pt += _t.sigmoid(net(Xte_t).squeeze(-1)).cpu().numpy()
    return pv / len(HEAD_SEEDS), pt / len(HEAD_SEEDS)


def _m(thr, p, y):
    yh = (p >= thr).astype(int)
    return dict(accuracy=100 * accuracy_score(y, yh), f1=100 * f1_score(y, yh, zero_division=0),
                precision=100 * precision_score(y, yh, zero_division=0),
                recall=100 * recall_score(y, yh, zero_division=0),
                roc_auc=100 * roc_auc_score(y, p), pr_auc=100 * average_precision_score(y, p))


def best_thr(p, y, obj):
    best, bs = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        yh = (p >= t).astype(int)
        if obj == "f1":
            sc = f1_score(y, yh, zero_division=0)
        else:
            tp = ((yh == 1) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
            fp = ((yh == 1) & (y == 0)).sum(); fn = ((yh == 0) & (y == 1)).sum()
            sc = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
        if sc > bs:
            bs, best = sc, float(t)
    return best


def eval_seed(channels, sids, y, seed, write_split):
    tr, va, te = stratified_3way(y, FRACS, seed)
    if write_split:
        RUNS.mkdir(parents=True, exist_ok=True)
        fold = {str(sids[i]): nm for nm, arr in (("train", tr), ("val", va), ("test", te)) for i in arr}
        (RUNS / f"{DATASET}_holdout_split.json").write_text(
            json.dumps({"seed": seed, "fracs": FRACS, "fold": fold}), encoding="utf-8")
    res = {}
    for cname, chs in CONFIGS.items():
        Xtr = np.concatenate([channels[c][tr] for c in chs], 1)
        Xva = np.concatenate([channels[c][va] for c in chs], 1)
        Xte = np.concatenate([channels[c][te] for c in chs], 1)
        mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        pv, pt = train_probs(Xtr, y[tr], Xva, Xte)
        thr = {"fixed_0.5": 0.5, "max_bal_acc": best_thr(pv, y[va], "bal"), "max_f1": best_thr(pv, y[va], "f1")}
        res[cname] = {pol: _m(thr[pol], pt, y[te]) for pol in POLICIES}
    return res, (len(tr), len(va), len(te))


def main():
    channels, sids, y = load_pool()
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    sizes = None
    for i, seed in enumerate(SPLIT_SEEDS):
        res, sizes = eval_seed(channels, sids, y, seed, write_split=(i == 0))
        for cname, pols in res.items():
            for pol, m in pols.items():
                for k, v in m.items():
                    agg[cname][pol][k].append(v)
        print(f"seed {seed} done")
    print(f"\npool n={len(y)} pos={y.mean()*100:.1f}%  "
          f"train/val/test = {sizes}  over {len(SPLIT_SEEDS)} split seeds\n")

    def ms(cname, pol, k):
        a = np.array(agg[cname][pol][k]); return a.mean(), a.std()

    def mark(mean, k):
        d = mean - FUSEVUL[k]
        return f"WIN +{d:.2f}" if d > 0.05 else (f"LOSE {d:.2f}" if d < -0.05 else "TIE")

    lines = ["# Held-out TEST-fold evaluation -- devign (mean +/- std over "
             f"{len(SPLIT_SEEDS)} split seeds)", "",
             f"Stratified 80/10/10 per seed over the deduped train+val pool "
             f"(n={len(y)}). Threshold on VAL, metrics on TEST. NOT FuSEVul's exact "
             "fold -- same dataset & protocol, different split.", "",
             f"FuSEVul (Devign test): Acc={FUSEVUL['accuracy']}, F1={FUSEVUL['f1']}", "",
             "| Config | Policy | Acc | F1 | ROC-AUC | Acc vs base | F1 vs base |",
             "|---|---|---|---|---|---|---|"]
    hdr = f"{'config':24s}{'policy':13s}{'Acc':>14}{'F1':>14}{'ROC':>14}"
    print(hdr); print("-" * len(hdr))
    for cname in CONFIGS:
        for pol in POLICIES:
            am, asd = ms(cname, pol, "accuracy"); fm, fsd = ms(cname, pol, "f1")
            rm, rsd = ms(cname, pol, "roc_auc")
            print(f"{cname:24s}{pol:13s}{f'{am:.2f}+/-{asd:.2f}':>14}"
                  f"{f'{fm:.2f}+/-{fsd:.2f}':>14}{f'{rm:.2f}+/-{rsd:.2f}':>14}")
            lines.append(f"| {cname} | {pol} | {am:.2f} ± {asd:.2f} | {fm:.2f} ± {fsd:.2f} | "
                         f"{rm:.2f} ± {rsd:.2f} | {mark(am,'accuracy')} | {mark(fm,'f1')} |")
        print()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{DATASET}_holdout_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORTS / (DATASET + '_holdout_test.md')}")


if __name__ == "__main__":
    main()
