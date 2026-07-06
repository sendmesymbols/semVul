"""Train one ladder rung end-to-end (both encoders fine-tuned).

Protocol (honest + comparable):
  - carve a stratified TUNE slice from train (threshold + epoch selection live here,
    never on val -> non-circular);
  - train on the rest, early-stop on TUNE PR-AUC (threshold-free);
  - report on val at THREE operating points: argmax@0.5 (directly FuSEVul-comparable),
    best-F1 threshold chosen on TUNE (honest tuned), and best-F1 threshold chosen on
    val (optimistic upper bound, labelled);
  - also report val ROC-AUC / PR-AUC (threshold-free -> the fair ladder-contribution
    measure);
  - save val/tune probabilities so any threshold can be recomputed without retraining.

One JSON per (dataset, rung); a crash never loses a completed rung.
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
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

import data as data_mod
from model import LadderModel, focal_ce

# Code encoder. CodeT5+ is incompatible with transformers 5.12 in this venv;
# GraphCodeBERT is a FuSEVul-reported encoder that loads + fine-tunes cleanly.
CODE_ID = "microsoft/graphcodebert-base"
TEXT_ID = "roberta-base"
RUNS = os.path.join(ROOT, "experiments", "runs")
STATED = {"devign": {"acc": 60.39, "f1": 55.91},
          "reveal": {"acc": 91.68, "f1": 46.76, "prec": 57.24, "rec": 39.52}}


def _tok(tokenizer, texts, max_len):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def _tune_mask(y, frac, seed):
    rng = np.random.default_rng(seed)
    m = np.zeros(len(y), dtype=bool)
    for c in (0, 1):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * frac)))
        m[idx[:k]] = True
    return ~m, m  # train_mask, tune_mask


def _best_thr(prob1, y, objective="f1"):
    best, bs = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        yh = (prob1 >= t).astype(int)
        if objective == "f1":
            s = f1_score(y, yh, zero_division=0)
        else:
            tp = ((yh == 1) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
            fp = ((yh == 1) & (y == 0)).sum(); fn = ((yh == 0) & (y == 1)).sum()
            s = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
        if s > bs:
            bs, best = s, float(t)
    return best


def _metrics_at(thr, prob1, y):
    yh = (prob1 >= thr).astype(int)
    return dict(threshold=round(float(thr), 3),
                acc=100 * accuracy_score(y, yh),
                f1=100 * f1_score(y, yh, zero_division=0),
                prec=100 * precision_score(y, yh, zero_division=0),
                rec=100 * recall_score(y, yh, zero_division=0))


def train_rung(dataset, rung, *, epochs=12, patience=3, batch=4, grad_accum=8,
               max_code=320, max_text=256, lr=2e-5, fusion="self", tune_frac=0.12,
               subset=None, seed=1337, split_seed=None, out_dir=RUNS):
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    # split_seed fixes the TUNE carve independently of training randomness so
    # multi-seed ensemble members share one tune slice (aligned tune probs).
    if split_seed is None:
        split_seed = seed
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    tag = f"{dataset}_{rung}" + ("_smoke" if subset else "")
    print(f"[{tag}] device={device} amp={'bf16' if bf16 else 'fp16'} fusion={fusion} "
          f"batch={batch}x{grad_accum} epochs<= {epochs}", flush=True)

    tr, va = data_mod.load(dataset, subset=subset)
    code_tok = AutoTokenizer.from_pretrained(CODE_ID)
    text_tok = AutoTokenizer.from_pretrained(TEXT_ID)
    code_enc = AutoModel.from_pretrained(CODE_ID)
    text_enc = AutoModel.from_pretrained(TEXT_ID)
    model = LadderModel(code_enc, text_enc, qual_dim=tr["qual"].shape[1],
                        rung=rung, fusion=fusion).to(device)
    model.enable_grad_checkpointing()

    ci, cm = _tok(code_tok, tr["code"], max_code)
    ti, tm = _tok(text_tok, tr["expl"], max_text)
    q = torch.from_numpy(tr["qual"]); y = tr["y"]
    va_ci, va_cm = _tok(code_tok, va["code"], max_code)
    va_ti, va_tm = _tok(text_tok, va["expl"], max_text)
    va_q = torch.from_numpy(va["qual"]); yva = va["y"]

    trm, tum = _tune_mask(y, tune_frac, split_seed)
    ytr, ytu = y[trm], y[tum]
    ytr_t = torch.from_numpy(ytr)
    print(f"[{tag}] train'={trm.sum()} tune={tum.sum()} val={len(yva)}", flush=True)

    pos_rate = float(ytr.mean())
    alpha_pos = float(np.clip(1.0 - pos_rate, 0.5, 0.80))
    use_focal = dataset == "reveal"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not bf16))

    tr_idx = np.where(trm)[0]
    tu_idx = np.where(tum)[0]

    @torch.no_grad()
    def prob1(ci_, cm_, ti_, tm_, q_):
        model.eval(); outs = []
        for i in range(0, len(ci_), max(2, batch)):
            s = slice(i, i + max(2, batch))
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                lo = model(ci_[s].to(device), cm_[s].to(device), ti_[s].to(device),
                           tm_[s].to(device), q_[s].to(device))
            outs.append(torch.softmax(lo.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(outs)

    best_ap, best = -1.0, None
    # Parallel tracker: the epoch with best val F1@0.5. This is the BASE PAPER's
    # selection rule (best epoch chosen on val, then reported on val -> circular).
    # We record it ONLY to produce a comparability column under their protocol;
    # our headline stays the non-circular tune-selected number above.
    best_valf1, best_val = -1.0, None
    # Faithful base-paper replication: FuSEVul selects the epoch by val ACCURACY.
    # Track that separately from best-val-F1, and log the full per-epoch val
    # trajectory + probs so the comparability column can't hide a cherry-pick.
    best_valacc, best_val_acc = -1.0, None
    ep_log = []            # [(ep, val_acc@0.5, val_f1@0.5, val_roc)]
    ep_val_probs = []      # per-epoch val_prob vectors, aligned to ep_log
    wait = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        opt.zero_grad()
        losses = []
        for si, i in enumerate(range(0, len(perm), batch)):
            bidx = perm[i:i + batch]
            bt = torch.as_tensor(bidx)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                logits = model(ci[bt].to(device), cm[bt].to(device), ti[bt].to(device),
                               tm[bt].to(device), q[bt].to(device))
                yb = torch.as_tensor(y[bidx], dtype=torch.long, device=device)
                loss = (focal_ce(logits, yb, alpha_pos) if use_focal
                        else nn.functional.cross_entropy(logits, yb))
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            if (si + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            losses.append(loss.item() * grad_accum)

        tu_p = prob1(ci[torch.as_tensor(tu_idx)], cm[torch.as_tensor(tu_idx)],
                     ti[torch.as_tensor(tu_idx)], tm[torch.as_tensor(tu_idx)],
                     q[torch.as_tensor(tu_idx)])
        va_p = prob1(va_ci, va_cm, va_ti, va_tm, va_q)
        ap = average_precision_score(ytu, tu_p) if ytu.sum() > 0 else 0.0
        va_f1_argmax = f1_score(yva, (va_p >= 0.5).astype(int), zero_division=0) * 100
        va_acc_argmax = accuracy_score(yva, (va_p >= 0.5).astype(int)) * 100
        va_roc = roc_auc_score(yva, va_p) * 100
        print(f"[{tag}] ep{ep}/{epochs} loss={np.mean(losses):.4f} tune_prauc={ap*100:.2f} "
              f"val_acc@0.5={va_acc_argmax:.2f} val_f1@0.5={va_f1_argmax:.2f} val_roc={va_roc:.2f}",
              flush=True)
        ep_log.append((ep, va_acc_argmax, va_f1_argmax, va_roc))
        ep_val_probs.append(va_p.copy())
        # Incremental crash-safe dump: the per-epoch acc/F1@0.5 trajectory is
        # enough to answer the base-paper-protocol question even if the run is
        # killed before the final JSON is written. Overwritten each epoch.
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"fusevul_ladder_{tag}_partial.json"),
                      "w", encoding="utf-8") as _pf:
                json.dump({"dataset": dataset, "rung": rung, "epochs_done": ep,
                           "stated": STATED[dataset],
                           "per_epoch": [{"epoch": e, "acc": round(a, 2),
                                          "f1": round(f, 2), "roc": round(r, 2)}
                                         for (e, a, f, r) in ep_log]}, _pf, indent=2)
        except OSError:
            pass
        if va_f1_argmax > best_valf1:
            best_valf1, best_val = va_f1_argmax, (ep, va_p.copy())
        if va_acc_argmax > best_valacc:
            best_valacc, best_val_acc = va_acc_argmax, (ep, va_p.copy())
        if ap > best_ap:
            best_ap, best, wait = ap, (ep, va_p, tu_p), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[{tag}] early stop @ep{ep}", flush=True)
                break

    ep_best, va_p, tu_p = best
    ep_bp, va_p_bp = best_val          # base-paper selection by best val F1@0.5
    ep_ba, va_p_ba = best_val_acc      # base-paper selection by best val ACC@0.5
    thr_tune = _best_thr(tu_p, ytu, "f1")
    thr_val = _best_thr(va_p, yva, "f1")
    payload = {
        "dataset": dataset, "rung": rung, "fusion": fusion, "best_epoch": ep_best,
        "val_roc_auc": 100 * roc_auc_score(yva, va_p),
        "val_pr_auc": 100 * average_precision_score(yva, va_p),
        "argmax": _metrics_at(0.5, va_p, yva),
        "tuned_on_tune": _metrics_at(thr_tune, va_p, yva),
        "tuned_on_val": _metrics_at(thr_val, va_p, yva),
        # Comparability column under the base paper's circular protocol (select
        # epoch on val, report val). Clearly labeled; NOT our headline number.
        # Both selection rules are reported at a SINGLE operating point (0.5) so
        # acc and F1 come from the same epoch (no double cherry-pick). FuSEVul's
        # stated rule is best val ACCURACY -> by_val_acc is the faithful match.
        "base_paper_protocol": {
            "by_val_acc": {
                "select": "best val ACC@0.5 epoch (faithful to base paper)",
                "epoch": ep_ba,
                "argmax": _metrics_at(0.5, va_p_ba, yva),
            },
            "by_val_f1": {
                "select": "best val F1@0.5 epoch (F1-favourable variant)",
                "epoch": ep_bp,
                "argmax": _metrics_at(0.5, va_p_bp, yva),
            },
            "per_epoch": [
                {"epoch": e, "acc": round(a, 2), "f1": round(f, 2), "roc": round(r, 2)}
                for (e, a, f, r) in ep_log
            ],
        },
        "stated": STATED[dataset],
        "config": dict(epochs=epochs, patience=patience, batch=batch,
                       grad_accum=grad_accum, max_code=max_code, max_text=max_text,
                       lr=lr, tune_frac=tune_frac, seed=seed, split_seed=split_seed,
                       subset=subset, use_focal=use_focal, alpha_pos=alpha_pos),
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"fusevul_ladder_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, f"fusevul_ladder_{tag}_probs.npz"),
                        val_prob=va_p, val_y=yva, tune_prob=tu_p, tune_y=ytu,
                        tune_idx=tu_idx, val_prob_bp=va_p_bp, val_prob_ba=va_p_ba,
                        val_probs_per_epoch=np.asarray(ep_val_probs),
                        ep_index=np.asarray([e for (e, *_ ) in ep_log]))
    a, t_, st = payload["argmax"], payload["tuned_on_tune"], STATED[dataset]
    bp = payload["base_paper_protocol"]["by_val_acc"]["argmax"]
    bpe = payload["base_paper_protocol"]["by_val_acc"]["epoch"]
    print(f"[{tag}] DONE @ep{ep_best}  ROC={payload['val_roc_auc']:.2f} PR={payload['val_pr_auc']:.2f} | "
          f"argmax acc={a['acc']:.2f} f1={a['f1']:.2f} | tuned acc={t_['acc']:.2f} f1={t_['f1']:.2f} "
          f"| base-paper-proto(by val acc) @ep{bpe} "
          f"acc={bp['acc']:.2f} f1={bp['f1']:.2f} "
          f"| stated {st} | {payload['seconds']/60:.1f} min", flush=True)
    del model, code_enc, text_enc
    if device == "cuda":
        torch.cuda.empty_cache()
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["devign", "reveal"])
    ap.add_argument("--rung", required=True, choices=["L1", "L2", "L3"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-code", type=int, default=320)
    ap.add_argument("--max-text", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--fusion", default="self", choices=["self", "cross"])
    ap.add_argument("--subset", type=int, default=None)
    args = ap.parse_args()
    train_rung(args.dataset, args.rung, epochs=args.epochs, patience=args.patience,
               batch=args.batch, grad_accum=args.grad_accum, max_code=args.max_code,
               max_text=args.max_text, lr=args.lr, fusion=args.fusion, subset=args.subset)


if __name__ == "__main__":
    main()
