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

# Code encoder is SELECTABLE per run via train_rung(code_enc=...):
#   "graphcodebert" (default) -> microsoft/graphcodebert-base
#   "codet5p"                  -> Salesforce/codet5p-110m-embedding (FuSEVul's
#                                 encoder; loaded via is_decoder shim, token
#                                 states read from .encoder -> code_kind="t5").
# GraphCodeBERT stays the default so existing runs are unchanged. The text
# channel is always RoBERTa (matches FuSEVul).
_CODE_ENC_IDS = {
    "graphcodebert": "microsoft/graphcodebert-base",
    "codet5p": "Salesforce/codet5p-110m-embedding",
}
TEXT_ID = "roberta-base"


def _resolve_code_id(code_enc):
    return _CODE_ENC_IDS.get(code_enc, code_enc)  # allow a raw HF id too


def _load_code_encoder(code_enc):
    """Return (encoder, tokenizer, code_kind). CodeT5+ needs a config shim on
    transformers 5.12 (CodeT5pEmbeddingConfig lacks is_decoder) and trust_remote_code."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    code_id = _resolve_code_id(code_enc)
    if code_enc == "codet5p":
        cfg = AutoConfig.from_pretrained(code_id, trust_remote_code=True)
        if not hasattr(cfg, "is_decoder"):
            cfg.is_decoder = False
        tok = AutoTokenizer.from_pretrained(code_id, trust_remote_code=True)
        enc = AutoModel.from_pretrained(code_id, config=cfg, trust_remote_code=True)
        return enc, tok, "t5"
    return (AutoModel.from_pretrained(code_id),
            AutoTokenizer.from_pretrained(code_id), "")
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
        elif objective == "acc":
            s = accuracy_score(y, yh)
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


def _hardware(device):
    """Hardware descriptor saved with every run so wall-time is auditable: a
    reviewer cannot interpret `seconds` without the GPU it ran on. Auto-detected;
    peak VRAM is filled at payload time (proves the run fits the card's budget)."""
    hw = {"device": device, "torch": torch.__version__}
    if device == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
            hw["cuda"] = torch.version.cuda
        except Exception:
            pass
    return hw


def _resolve_focal(use_focal, computed_alpha, focal_alpha, focal_gamma):
    """Actual (alpha_pos, gamma) used. Overrides apply ONLY when focal is on
    (ReVeal); Devign (use_focal=False) uses plain CE and is unaffected."""
    if not use_focal:
        return computed_alpha, float(focal_gamma)
    alpha = computed_alpha if focal_alpha is None else float(focal_alpha)
    return alpha, float(focal_gamma)


TAG_PREFIX = "semanticvul"   # output-file tag; was "fusevul_ladder" (leftover from the
                             # original FuSEVul-comparison scaffold). Historical L1/L2
                             # runs on disk still use the old tag -- reproduce_real.py's
                             # resume check and aggregate_seeds.py's reader both accept
                             # either, so nothing already trained gets silently redone.


def train_rung(dataset, rung, *, epochs=12, patience=3, batch=4, grad_accum=8,
               max_code=320, max_text=256, lr=2e-5, fusion="self", tune_frac=0.12,
               subset=None, seed=1337, split_seed=None, out_dir=RUNS, focal="auto",
               focal_alpha=None, focal_gamma=2.0, warmup=0.1, code_enc="graphcodebert"):
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    code_id = _resolve_code_id(code_enc)
    print(f"[{dataset}_{rung}] code_enc={code_enc} ({code_id})", flush=True)
    # split_seed fixes the TUNE carve independently of training randomness so
    # multi-seed ensemble members share one tune slice (aligned tune probs).
    if split_seed is None:
        split_seed = seed
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()   # -> peak VRAM for the resource claim
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    tag = f"{dataset}_{rung}" + ("_smoke" if subset else "")
    print(f"[{tag}] device={device} amp={'bf16' if bf16 else 'fp16'} fusion={fusion} "
          f"batch={batch}x{grad_accum} epochs<= {epochs}", flush=True)

    tr, va = data_mod.load(dataset, subset=subset)
    code_enc_model, code_tok, code_kind = _load_code_encoder(code_enc)
    text_tok = AutoTokenizer.from_pretrained(TEXT_ID)
    text_enc = AutoModel.from_pretrained(TEXT_ID)
    model = LadderModel(code_enc_model, text_enc, qual_dim=tr["qual"].shape[1],
                        rung=rung, fusion=fusion, code_kind=code_kind).to(device)
    # RQ2 frozen-encoder regime (SEMVUL_FROZEN=1): freeze both encoders and train
    # only the lightweight fusion + gate + head. Matches RO2 ("over frozen, cached
    # encoders"), and gives the quality gate real work to do (noisier frozen reps
    # -> explanations vary in usefulness). Grad-checkpointing is pointless without
    # an encoder backward pass, so skip it when frozen.
    frozen = os.environ.get("SEMVUL_FROZEN") == "1"
    if frozen:
        for p in model.code_enc.parameters():
            p.requires_grad_(False)
        if model.text_enc is not None:
            for p in model.text_enc.parameters():
                p.requires_grad_(False)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{tag}] FROZEN encoders -> training {n_train/1e3:.1f}K params "
              f"(head only)", flush=True)
    else:
        model.enable_grad_checkpointing()

    # Rung separation (supervisor 2026-07-09): the explanation-guided code
    # window ('code_ev') belongs to the EXPLANATION component, so L1 always
    # reads the pure head-truncated code; L2/L3 read the guided window when
    # SEMVUL_CODE_WINDOW=evidence (otherwise code_ev == code and this is a
    # no-op). Keeps L2-L1 attributable to explanations, L3-L2 to quality.
    code_key = "code" if rung == "L1" else "code_ev"
    ci, cm = _tok(code_tok, tr.get(code_key, tr["code"]), max_code)
    ti, tm = _tok(text_tok, tr["expl"], max_text)
    q = torch.from_numpy(tr["qual"])
    conf = torch.from_numpy(tr["conf"]).float()
    y = tr["y"]
    va_ci, va_cm = _tok(code_tok, va.get(code_key, va["code"]), max_code)
    va_ti, va_tm = _tok(text_tok, va["expl"], max_text)
    va_q = torch.from_numpy(va["qual"])
    va_conf = torch.from_numpy(va["conf"]).float()
    yva = va["y"]

    trm, tum = _tune_mask(y, tune_frac, split_seed)
    ytr, ytu = y[trm], y[tum]
    ytr_t = torch.from_numpy(ytr)
    print(f"[{tag}] train'={trm.sum()} tune={tum.sum()} val={len(yva)}", flush=True)

    pos_rate = float(ytr.mean())
    computed_alpha = float(np.clip(1.0 - pos_rate, 0.5, 0.80))
    # focal="auto" preserves prior behavior (focal on ReVeal only); "on"/"off" override
    # so the RO4 focal-loss ablation can be run on Devign (currently plain CE).
    use_focal = {"on": True, "off": False}.get(focal, dataset == "reveal")
    # ReVeal-only focal knobs (passed by the final ReVeal launchers).
    # focal_alpha=None keeps the auto alpha; Devign (use_focal=False) is untouched.
    alpha_pos, gamma = _resolve_focal(use_focal, computed_alpha, focal_alpha, focal_gamma)
    gate_lr_mult = float(os.environ.get("SEMVUL_GATE_LR_MULT", "1.0"))
    if getattr(model, "use_gate", False) and gate_lr_mult != 1.0:
        gate_params = list(model.gate.parameters())
        gate_ids = {id(p) for p in gate_params}
        other_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) not in gate_ids]
        opt = torch.optim.AdamW(
            [{"params": other_params, "lr": lr},
             {"params": gate_params, "lr": lr * gate_lr_mult}],
            weight_decay=1e-2)
        print(f"[{tag}] gate lr x{gate_lr_mult:g} -> {lr * gate_lr_mult:.2e}", flush=True)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=lr, weight_decay=1e-2)
    # 2026-07-15: linear warmup + decay (standard RoBERTa/GraphCodeBERT recipe;
    # previous behavior was constant LR). warmup=0 restores constant LR.
    # Floor 0.05*lr so early-stopped runs never train at ~zero LR.
    n_updates = max(1, epochs * max(1, int(np.ceil(trm.sum() / (batch * grad_accum)))))
    n_warm = int(warmup * n_updates)
    sched = None
    if warmup > 0:
        def _lr_lambda(step):
            if step < n_warm:
                return (step + 1) / max(1, n_warm)
            return max(0.05, (n_updates - step) / max(1, n_updates - n_warm))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not bf16))

    tr_idx = np.where(trm)[0]
    tu_idx = np.where(tum)[0]

    # Quality-feature standardization (2026-07-11 fix, widened to BOTH datasets
    # 2026-07-15 as the original comment sanctioned). The 44-dim qual block is
    # raw counts (per-col max up to ~1100, some col std ~85) and is concatenated
    # onto a LayerNorm'd ~unit-scale embedding in the L3 head; the scale
    # imbalance starves the encoder gradient so L3 warms up from ~chance ROC
    # over many epochs. Z-scoring with TRAIN'-split stats (fit on tr_idx only,
    # so tune/val are held out -> no leakage) removes it. No-op for L1/L2.
    _qmu = q[torch.as_tensor(tr_idx)].mean(0, keepdim=True)
    _qsd = q[torch.as_tensor(tr_idx)].std(0, keepdim=True).clamp_min(1e-6)
    q = (q - _qmu) / _qsd
    va_q = (va_q - _qmu) / _qsd
    print(f"[{tag}] qual z-scored (train'-stats) dim={q.shape[1]}", flush=True)

    @torch.no_grad()
    def prob1(ci_, cm_, ti_, tm_, q_, conf_):
        model.eval(); outs = []
        for i in range(0, len(ci_), max(2, batch)):
            s = slice(i, i + max(2, batch))
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                lo = model(ci_[s].to(device), cm_[s].to(device), ti_[s].to(device),
                           tm_[s].to(device), q_[s].to(device), conf_[s].to(device))
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
                               tm[bt].to(device), q[bt].to(device), conf[bt].to(device))
                yb = torch.as_tensor(y[bidx], dtype=torch.long, device=device)
                loss = (focal_ce(logits, yb, alpha_pos, gamma) if use_focal
                        else nn.functional.cross_entropy(logits, yb))
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            if (si + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
                if sched is not None:
                    sched.step()
            losses.append(loss.item() * grad_accum)

        tu_p = prob1(ci[torch.as_tensor(tu_idx)], cm[torch.as_tensor(tu_idx)],
                     ti[torch.as_tensor(tu_idx)], tm[torch.as_tensor(tu_idx)],
                     q[torch.as_tensor(tu_idx)], conf[torch.as_tensor(tu_idx)])
        va_p = prob1(va_ci, va_cm, va_ti, va_tm, va_q, va_conf)
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
            with open(os.path.join(out_dir, f"{TAG_PREFIX}_{tag}_partial.json"),
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
    # Platt calibration fitted on the TUNE slice (non-circular, monotone ->
    # ROC/PR unchanged). Rungs land at different miscalibrations, so raw @0.5
    # compares operating points, not rungs (Devign L3 raw 57.87 -> calibrated
    # 59.41 while its all-threshold acc ceiling 61.46 tops L2's 60.76).
    # calibrated@0.5 is the fair single-operating-point rung comparison.
    from sklearn.linear_model import LogisticRegression as _PlattLR
    _lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                           (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    _platt = _PlattLR(max_iter=1000).fit(_lo(tu_p).reshape(-1, 1), ytu)
    va_p_cal = _platt.predict_proba(_lo(va_p).reshape(-1, 1))[:, 1]
    payload = {
        "dataset": dataset, "rung": rung, "fusion": fusion, "best_epoch": ep_best,
        "val_roc_auc": 100 * roc_auc_score(yva, va_p),
        "val_pr_auc": 100 * average_precision_score(yva, va_p),
        "argmax": _metrics_at(0.5, va_p, yva),
        "calibrated_at_05": _metrics_at(0.5, va_p_cal, yva),
        "tuned_on_tune": _metrics_at(thr_tune, va_p, yva),
        # accuracy-objective threshold, still chosen on TUNE (non-circular):
        # the honest max-accuracy operating point next to the max-F1 one.
        "tuned_on_tune_acc": _metrics_at(_best_thr(tu_p, ytu, "acc"), va_p, yva),
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
        # Hardware + peak VRAM: attaches every `seconds` figure to the GPU it ran
        # on (RQ4 low-resource / resource-saver claim -> absolute feasibility).
        "hardware": {**_hardware(device),
                     "peak_vram_gb": (round(torch.cuda.max_memory_allocated()
                                            / (1024 ** 3), 2)
                                      if device == "cuda" else None)},
        "config": dict(epochs=epochs, patience=patience, batch=batch,
                       grad_accum=grad_accum, max_code=max_code, max_text=max_text,
                       lr=lr, tune_frac=tune_frac, seed=seed, split_seed=split_seed,
                       subset=subset, use_focal=use_focal, alpha_pos=alpha_pos,
                       focal_gamma=gamma, warmup=warmup,
                       code_enc=code_enc, code_id=code_id,
                       hard_conf_switch=(os.environ.get("SEMVUL_HARD_CONF_SWITCH") == "1"),
                       hard_conf_thresh=float(os.environ.get("SEMVUL_HARD_CONF_THRESH", "50")),
                       use_gate=getattr(model, "use_gate", False)),
        "seconds": round(time.time() - t0, 1),
    }
    if getattr(model, "use_gate", False):
        model.eval()
        w_parts = []
        with torch.no_grad():
            for i in range(0, len(va_ti), max(2, batch)):
                s = slice(i, i + max(2, batch))
                th = model.text_enc(input_ids=va_ti[s].to(device),
                                    attention_mask=va_tm[s].to(device)).last_hidden_state
                expl_pooled = model._pool(th, va_tm[s].to(device))
                cf = va_conf[s].to(device).to(expl_pooled.dtype).view(-1, 1)
                gate_in = torch.cat([expl_pooled, cf], dim=-1)
                w_parts.append(torch.sigmoid(model.gate(gate_in)).squeeze(-1).float().cpu())
        w_np = torch.cat(w_parts).numpy()
        payload["gate"] = {
            "w_mean": round(float(w_np.mean()), 4),
            "w_std": round(float(w_np.std()), 4),
            "w_min": round(float(w_np.min()), 4),
            "w_max": round(float(w_np.max()), 4),
            "pct_code_side": round(float((w_np > 0.5).mean()) * 100, 1),
        }
        print(f"[{tag}] gate w (code fraction): mean={w_np.mean():.3f} "
              f"std={w_np.std():.3f} range=[{w_np.min():.3f}, {w_np.max():.3f}] "
              f"| {(w_np > 0.5).mean()*100:.0f}% routed to code", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{TAG_PREFIX}_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, f"{TAG_PREFIX}_{tag}_probs.npz"),
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
    del model, code_enc_model, text_enc
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
    ap.add_argument("--code-enc", default="graphcodebert",
                    help="code encoder: graphcodebert (default) | codet5p | raw HF id")
    args = ap.parse_args()
    train_rung(args.dataset, args.rung, epochs=args.epochs, patience=args.patience,
               batch=args.batch, grad_accum=args.grad_accum, max_code=args.max_code,
               max_text=args.max_text, lr=args.lr, fusion=args.fusion, subset=args.subset,
               code_enc=args.code_enc)


if __name__ == "__main__":
    main()
