"""Train one ladder rung end-to-end (both encoders fine-tuned). Writes one JSON
per (dataset, rung) so a crash never loses a completed rung.

CLI:
  python experiments/fusevul_ladder/train.py --dataset reveal --rung L2
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import data as data_mod
from model import LadderModel, focal_ce

# Code encoder. CodeT5+ (110m-embedding AND 220m) is incompatible with
# transformers 5.12 in this venv (custom-config + tokenizer add_tokens both break).
# We use GraphCodeBERT — a code encoder FuSEVul also reports (Devign 59.74/55.32,
# Reveal 90.37/45.39) — which loads and fine-tunes cleanly here. RoBERTa-based, 768-d.
CODE_ID = "microsoft/graphcodebert-base"
TEXT_ID = "roberta-base"
RUNS = os.path.join(ROOT, "experiments", "runs")
STATED = {"devign": {"acc": 60.39, "f1": 55.91},
          "reveal": {"acc": 91.68, "f1": 46.76, "prec": 57.24, "rec": 39.52}}


def _tok(tokenizer, texts, max_len):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def _metrics(logits, y):
    pred = logits.argmax(1)
    return dict(acc=100 * accuracy_score(y, pred),
                f1=100 * f1_score(y, pred, zero_division=0),
                prec=100 * precision_score(y, pred, zero_division=0),
                rec=100 * recall_score(y, pred, zero_division=0))


def train_rung(dataset, rung, *, epochs=5, batch=4, grad_accum=8, max_code=320,
               max_text=256, lr=2e-5, fusion="self", subset=None, seed=1337,
               out_dir=RUNS):
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    tag = f"{dataset}_{rung}" + ("" if subset is None else "_smoke")
    print(f"[{tag}] device={device} amp={'bf16' if bf16 else 'fp16'} "
          f"fusion={fusion} batch={batch}x{grad_accum}", flush=True)

    tr, va = data_mod.load(dataset, subset=subset)
    code_tok = AutoTokenizer.from_pretrained(CODE_ID)
    text_tok = AutoTokenizer.from_pretrained(TEXT_ID)
    code_enc = AutoModel.from_pretrained(CODE_ID)
    text_enc = AutoModel.from_pretrained(TEXT_ID)

    model = LadderModel(code_enc, text_enc, qual_dim=tr["qual"].shape[1],
                        rung=rung, fusion=fusion).to(device)
    model.enable_grad_checkpointing()

    # pre-tokenize once
    tr_ci, tr_cm = _tok(code_tok, tr["code"], max_code)
    va_ci, va_cm = _tok(code_tok, va["code"], max_code)
    tr_ti, tr_tm = _tok(text_tok, tr["expl"], max_text)
    va_ti, va_tm = _tok(text_tok, va["expl"], max_text)
    tr_q = torch.from_numpy(tr["qual"]); va_q = torch.from_numpy(va["qual"])
    tr_y = torch.from_numpy(tr["y"]); yva = va["y"]

    pos_rate = float(tr["y"].mean())
    alpha_pos = float(np.clip(1.0 - pos_rate, 0.5, 0.80))
    use_focal = dataset == "reveal"

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not bf16))
    n = len(tr_y)

    @torch.no_grad()
    def predict(ci, cm, ti, tm, q):
        model.eval()
        outs = []
        bs = max(2, batch)
        for i in range(0, len(ci), bs):
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                lo = model(ci[i:i+bs].to(device), cm[i:i+bs].to(device),
                           ti[i:i+bs].to(device), tm[i:i+bs].to(device),
                           q[i:i+bs].to(device))
            outs.append(lo.float().cpu())
        return torch.cat(outs)

    best = {"f1": -1.0}
    best_epoch = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        opt.zero_grad()
        losses = []
        for si, i in enumerate(range(0, n, batch)):
            idx = perm[i:i+batch]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                logits = model(tr_ci[idx].to(device), tr_cm[idx].to(device),
                               tr_ti[idx].to(device), tr_tm[idx].to(device),
                               tr_q[idx].to(device))
                yb = tr_y[idx].to(device)
                loss = (focal_ce(logits, yb, alpha_pos) if use_focal
                        else nn.functional.cross_entropy(logits, yb))
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            if (si + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            losses.append(loss.item() * grad_accum)
        vlogits = predict(va_ci, va_cm, va_ti, va_tm, va_q)
        m = _metrics(vlogits, yva)
        print(f"[{tag}] ep{ep}/{epochs} loss={np.mean(losses):.4f} "
              f"val acc={m['acc']:.2f} f1={m['f1']:.2f} p={m['prec']:.2f} r={m['rec']:.2f}",
              flush=True)
        if m["f1"] > best["f1"]:
            best, best_epoch = m, ep

    dur = time.time() - t0
    payload = {"dataset": dataset, "rung": rung, "fusion": fusion,
               "best_epoch": best_epoch, "metrics": best, "stated": STATED[dataset],
               "config": dict(epochs=epochs, batch=batch, grad_accum=grad_accum,
                              max_code=max_code, max_text=max_text, lr=lr, seed=seed,
                              subset=subset, use_focal=use_focal, alpha_pos=alpha_pos),
               "seconds": round(dur, 1)}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"fusevul_ladder_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[{tag}] DONE best@ep{best_epoch}: acc={best['acc']:.2f} f1={best['f1']:.2f} "
          f"(stated {STATED[dataset]}) in {dur/60:.1f} min", flush=True)
    del model, code_enc, text_enc
    if device == "cuda":
        torch.cuda.empty_cache()
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["devign", "reveal"])
    ap.add_argument("--rung", required=True, choices=["L1", "L2", "L3"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-code", type=int, default=320)
    ap.add_argument("--max-text", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--fusion", default="self", choices=["self", "cross"])
    ap.add_argument("--subset", type=int, default=None)
    args = ap.parse_args()
    train_rung(args.dataset, args.rung, epochs=args.epochs, batch=args.batch,
               grad_accum=args.grad_accum, max_code=args.max_code,
               max_text=args.max_text, lr=args.lr, fusion=args.fusion, subset=args.subset)


if __name__ == "__main__":
    main()
