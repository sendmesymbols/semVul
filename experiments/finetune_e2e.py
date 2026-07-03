"""SELF-CONTAINED, SINGLE-FILE end-to-end fine-tune of GraphCodeBERT (step B).

Delete this file and nothing else breaks. Writes only:
    experiments/reports/devign_finetune_e2e.md
    experiments/runs/devign_finetune_e2e_probs.npz

WHY: the project's LoRA step trains a classifier end-to-end but then THROWS IT
AWAY, extracts frozen mean-pooled embeddings, and trains a separate head on top
(ceiling ROC-AUC ~64, acc ~59, ~1 pt short of FuSEVul). This instead fine-tunes
the encoder + classifier JOINTLY (full fine-tune), selects the best epoch on VAL,
and reports on the SAME held-out TEST fold as holdout_test_eval.py (seed-2025
split in experiments/runs/devign_holdout_split.json) -- so it is directly
comparable to the code-only baseline. No leakage: folds are disjoint unique
functions by sample_id.

Run (long; use background):
    .venv/Scripts/python.exe experiments/finetune_e2e.py
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(r"D:\Projects\SemVul")
# point HF at the project model cache and stay offline (models already cached)
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "models" / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

BASE_ID = "microsoft/graphcodebert-base"
JSONL = ROOT / "explanations" / "SemanticVul" / "devign"
SPLIT_JSON = ROOT / "experiments" / "runs" / "devign_holdout_split.json"
REPORT = ROOT / "experiments" / "reports" / "devign_finetune_e2e.md"
PROBS = ROOT / "experiments" / "runs" / "devign_finetune_e2e_probs.npz"
FUSEVUL = {"accuracy": 60.39, "f1": 55.91}

MAX_LEN = 400
BATCH = 8
GRAD_ACCUM = 4          # effective batch 32
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
PATIENCE = 2            # early stop on val ROC-AUC
SEED = 1337


# ---------------------------------------------------------------- data
def load_folds():
    fold = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))["fold"]  # sid -> split
    sid_code, sid_label = {}, {}
    for split in ("train", "val"):
        with (JSONL / f"devign_{split}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sid = str(r["sample_id"])
                if sid not in sid_code:               # sha1(code) => identical code per sid
                    sid_code[sid] = r.get("raw_code", "") or ""
                    sid_label[sid] = int(r["label"])
    buckets = {"train": [], "val": [], "test": []}
    for sid, sp in fold.items():
        if sid in sid_code:
            buckets[sp].append((sid_code[sid], sid_label[sid]))
    return buckets


# ---------------------------------------------------------------- model
class Classifier(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------- metrics
def metrics_at(thr, p, y):
    yh = (p >= thr).astype(int)
    return dict(accuracy=100 * accuracy_score(y, yh), f1=100 * f1_score(y, yh, zero_division=0),
                precision=100 * precision_score(y, yh, zero_division=0),
                recall=100 * recall_score(y, yh, zero_division=0),
                roc_auc=100 * roc_auc_score(y, p), pr_auc=100 * average_precision_score(y, p))


def best_thr(p, y, obj):
    best, bs = 0.5, -1.0
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


# ---------------------------------------------------------------- train / eval
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_ID)
    base = AutoModel.from_pretrained(BASE_ID)
    model = Classifier(base).to(device)
    model.base.gradient_checkpointing_enable()
    model.base.config.use_cache = False

    buckets = load_folds()
    tr, va, te = buckets["train"], buckets["val"], buckets["test"]
    ytr = np.array([l for _, l in tr]); yva = np.array([l for _, l in va]); yte = np.array([l for _, l in te])
    print(f"folds  train={len(tr)} ({ytr.mean()*100:.1f}%)  "
          f"val={len(va)} ({yva.mean()*100:.1f}%)  test={len(te)} ({yte.mean()*100:.1f}%)")

    pos = int(ytr.sum()); neg = len(ytr) - pos
    pos_w = torch.tensor([neg / max(1, pos)], dtype=torch.float32, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_total = (len(tr) // (BATCH * GRAD_ACCUM) + 1) * EPOCHS
    warmup = max(1, int(steps_total * WARMUP_RATIO))
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    def lr_at(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, steps_total - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))

    def encode_batch(pairs):
        codes = [c for c, _ in pairs]
        enc = tok(codes, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    @torch.no_grad()
    def predict(pairs):
        model.eval()
        out = []
        for i in range(0, len(pairs), 32):
            ii, am = encode_batch(pairs[i:i + 32])
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                out.append(torch.sigmoid(model(ii, am)).float().cpu().numpy())
        return np.concatenate(out)

    best_auc, best_state, patience, step = -1.0, None, 0, 0
    idx = np.arange(len(tr))
    for ep in range(EPOCHS):
        model.train(); np.random.shuffle(idx); opt.zero_grad()
        run = 0.0
        for bi, s in enumerate(range(0, len(idx), BATCH)):
            batch = [tr[j] for j in idx[s:s + BATCH]]
            ii, am = encode_batch(batch)
            yb = torch.tensor([l for _, l in batch], dtype=torch.float32, device=device)
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                loss = lossf(model(ii, am), yb) / GRAD_ACCUM
            (scaler.scale(loss) if scaler else loss).backward()
            run += loss.item() * GRAD_ACCUM
            if (bi + 1) % GRAD_ACCUM == 0:
                for pg in opt.param_groups:
                    pg["lr"] = LR * lr_at(step)
                if scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(); step += 1
        pva = predict(va); auc = roc_auc_score(yva, pva)
        print(f"epoch {ep+1}/{EPOCHS}  train_loss={run/max(1,len(idx)//BATCH):.4f}  val_roc_auc={auc*100:.2f}")
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("early stop"); break

    model.load_state_dict(best_state)
    pva, pte = predict(va), predict(te)
    t_bal, t_f1 = best_thr(pva, yva, "bal"), best_thr(pva, yva, "f1")

    def mark(v, k):
        d = v - FUSEVUL[k]
        return f"WIN +{d:.2f}" if d > 0.05 else (f"LOSE {d:.2f}" if d < -0.05 else "TIE")

    lines = ["# End-to-end fine-tune (GraphCodeBERT, full FT) -- devign held-out TEST", "",
             f"Same seed-2025 80/10/10 fold as holdout_test_eval.py. Encoder+classifier "
             f"fine-tuned JOINTLY, best epoch by val ROC-AUC. NOT FuSEVul's exact fold.", "",
             f"best val ROC-AUC={best_auc*100:.2f}", "",
             f"FuSEVul (Devign test): Acc={FUSEVUL['accuracy']}, F1={FUSEVUL['f1']}", "",
             "| Policy | Acc | F1 | Prec | Rec | ROC-AUC | PR-AUC | Acc? | F1? |",
             "|---|---|---|---|---|---|---|---|---|"]
    print(f"\n{'policy':13s}{'Acc':>7}{'F1':>7}{'P':>7}{'R':>7}{'ROC':>7}{'PR':>7}")
    for pol, thr in (("fixed_0.5", 0.5), ("max_bal_acc", t_bal), ("max_f1", t_f1)):
        m = metrics_at(thr, pte, yte)
        print(f"{pol:13s}{m['accuracy']:7.2f}{m['f1']:7.2f}{m['precision']:7.2f}"
              f"{m['recall']:7.2f}{m['roc_auc']:7.2f}{m['pr_auc']:7.2f}")
        lines.append(f"| {pol} | {m['accuracy']:.2f} | {m['f1']:.2f} | {m['precision']:.2f} | "
                     f"{m['recall']:.2f} | {m['roc_auc']:.2f} | {m['pr_auc']:.2f} | "
                     f"{mark(m['accuracy'],'accuracy')} | {mark(m['f1'],'f1')} |")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savez_compressed(PROBS, probs_test=pte, y_test=yte, probs_val=pva, y_val=yva)
    print(f"\nwrote {REPORT}\nwrote {PROBS}")


if __name__ == "__main__":
    main()
