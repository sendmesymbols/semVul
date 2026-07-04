"""SELF-CONTAINED PILOT — does the ONE untested pathway carry signal?

Question this answers (and nothing else)
----------------------------------------
Every prior negative result on the explanation channel came from either a
label-free *standalone* signal probe (AUC ~0.51) or the *frozen-MiniLM* pipeline
that fused pooled vectors. The one architecture that has NEVER been tested with
regenerated v2 explanations is the FuSEVul-intended pathway:

    fine-tuned RoBERTa over the explanation TOKENS  +  multi-head self-attention
    fusion into the (fine-tuned) code encoder.

This script isolates exactly that variable on a fresh, deletable subset:

  L1 = code-only (GraphCodeBERT, fine-tuned)              -> mean-pool -> head
  L2 = L1 + fine-tuned RoBERTa on v2 explanation tokens, fused by self-attention

Both rungs are trained and evaluated on the SAME samples, SAME internal
train/tune/test split, SAME seed, SAME schedule. The only difference is whether
the v2 explanation channel exists. The verdict metric is threshold-free
val/test ROC-AUC and PR-AUC (a fair ladder-contribution measure); F1/acc at
0.5 and at a tune-chosen threshold are reported alongside.

Everything this writes lives under experiments/expl_v2_pilot/ — delete that one
folder and the pilot is gone. It reuses the reference v2 prompt (expl_v2/
prompt_v2.py) and the reference ladder model (fusevul_ladder/model.py), which
are the very implementations under test; it modifies neither.

Run it (attended) in PyCharm with no arguments, or:
    .venv/Scripts/python.exe experiments/expl_v2_pilot/run_pilot.py
Generation is resumable (interrupt and re-run; done sample_ids are skipped).
Pass --skip-gen to only (re)run the L1-vs-L2 comparison once explanations exist.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Paths + imports. src.config must import before transformers (it points the HF
# cache at project/models, matching every prior run so nothing re-downloads).
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # D:\Projects\SemVul
EXPL_V2 = os.path.join(ROOT, "experiments", "expl_v2")
LADDER = os.path.join(ROOT, "experiments", "fusevul_ladder")
for _p in (ROOT, EXPL_V2, LADDER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config  # noqa: F401,E402  (sets HF_HOME/HF_HUB_CACHE before transformers)
from src.data_io import load_split                       # noqa: E402
from prompt_v2 import JSON_SCHEMA, build_messages         # noqa: E402  (the v2 prompt under test)

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SUBSET_JSON = os.path.join(OUT, "subset.json")
EXPL_JSONL = os.path.join(OUT, "expl_v2_pilot.jsonl")
RESULTS_JSON = os.path.join(OUT, "pilot_results.json")

# ---------------------------------------------------------------------------
# Defaults (PyCharm "Run" with no args uses these). Override via CLI if needed.
# ---------------------------------------------------------------------------
MODEL = "qwen3.5:9b"          # exact Ollama tag as requested
HOST = "http://localhost:9999"
N_SUBSET = 3000               # total samples (N/2 per class); ~3-5k range
NO_THINK = True               # qwen3-family: send "think": false (huge speedup)
WORKERS = 1                   # raise only if OLLAMA_NUM_PARALLEL is set server-side
NUM_CTX = 8192
GEN_TIMEOUT = 600
MAX_CODE_CHARS = 12000        # ~3k tokens; matches generate_v2

GEN_SEED = 13                 # deterministic subset pick (gen and train see same set)
SPLIT_SEED = 1337             # internal train/tune/test carve
TRAIN_SEED = 1337
TEST_FRAC = 0.30              # internal held-out test (the L1-vs-L2 comparison set)
TUNE_FRAC = 0.12              # carved from the internal train remainder

EPOCHS = 12
PATIENCE = 3
BATCH = 4
GRAD_ACCUM = 8
MAX_CODE = 320
MAX_TEXT = 256
LR = 2e-5
CODE_ID = "microsoft/graphcodebert-base"   # FuSEVul-reported code encoder
TEXT_ID = "roberta-base"                    # FuSEVul-intended explanation encoder

RISK_MAP = {"none": 0, "low": 1, "medium": 2, "high": 3}


# ===========================================================================
# Phase 0 — deterministic stratified subset (dedup within train)
# ===========================================================================
def select_subset(n: int, seed: int):
    samples = load_split("devign", "train")
    seen, uniq = set(), []
    for s in samples:                       # drop within-train duplicate functions
        if s.sample_id in seen:
            continue
        seen.add(s.sample_id)
        uniq.append(s)
    by_label = {0: [], 1: []}
    for s in uniq:
        if s.label in by_label:
            by_label[s.label].append(s)
    rng = random.Random(seed)
    picked = []
    for lab in (0, 1):
        pool = sorted(by_label[lab], key=lambda s: s.sample_id)
        rng.shuffle(pool)
        picked.extend(pool[: n // 2])
    picked.sort(key=lambda s: s.sample_id)
    with open(SUBSET_JSON, "w", encoding="utf-8") as fh:
        json.dump([{"sample_id": s.sample_id, "label": s.label} for s in picked],
                  fh, indent=2)
    npos = sum(s.label for s in picked)
    print(f"[subset] picked {len(picked)} unique samples "
          f"(pos {npos}, neg {len(picked) - npos}) -> {SUBSET_JSON}", flush=True)
    return picked


# ===========================================================================
# Phase 1 — regenerate v2 explanations for exactly that subset (resumable)
# ===========================================================================
def ollama_chat(host, model, messages, num_ctx, timeout, no_think):
    payload = {
        "model": model, "messages": messages, "stream": False,
        "format": JSON_SCHEMA,
        "options": {"temperature": 0.0, "seed": 1234, "num_ctx": num_ctx},
    }
    if no_think:
        payload["think"] = False
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate(samples, args):
    done = set()
    if os.path.exists(EXPL_JSONL):
        with open(EXPL_JSONL, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    todo = [s for s in samples if s.sample_id not in done]
    print(f"[gen] model={args.model} host={args.host} think={not args.no_think} "
          f"total={len(samples)} done={len(done)} todo={len(todo)} -> {EXPL_JSONL}",
          flush=True)
    if not todo:
        print("[gen] nothing to generate.", flush=True)
        return

    def gen_one(s):
        code = s.code[:MAX_CODE_CHARS]
        err = None
        for attempt in (1, 2):
            try:
                t1 = time.time()
                resp = ollama_chat(args.host, args.model, build_messages(code),
                                   args.num_ctx, args.timeout, args.no_think)
                dur = time.time() - t1
                expl = json.loads(resp["message"]["content"])
                return {"sample_id": s.sample_id, "label": s.label,
                        "raw_code": s.code, "explanation": expl,
                        "meta": {"model": args.model, "prompt": "v2",
                                 "gen_seconds": round(dur, 2)}}, None
            except (urllib.error.URLError, urllib.error.HTTPError,
                    json.JSONDecodeError, KeyError, TimeoutError, OSError) as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 * attempt)
        return None, f"{s.sample_id}: {err}"

    from concurrent.futures import ThreadPoolExecutor, as_completed
    n_ok, n_fail, t0 = 0, 0, time.time()
    with open(EXPL_JSONL, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = [pool.submit(gen_one, s) for s in todo]
            for i, fut in enumerate(as_completed(futs)):
                row, err = fut.result()
                if row is None:
                    n_fail += 1
                    print(f"[gen] FAIL {err}", flush=True)
                    continue
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n_ok += 1
                if n_ok % 10 == 0 or i == len(todo) - 1:
                    rate = n_ok / max(time.time() - t0, 1e-9)
                    eta = (len(todo) - i - 1) / max(rate, 1e-9) / 60
                    print(f"[gen] {n_ok}/{len(todo)} ok ({n_fail} fail) "
                          f"{rate:.2f}/s eta {eta:.0f} min", flush=True)
    print(f"[gen] DONE ok={n_ok} fail={n_fail} "
          f"elapsed={(time.time() - t0) / 60:.1f} min", flush=True)


# ===========================================================================
# v2 explanation -> fused text.
# We include the natural-language content (structural facts, present safety
# checks, risk patterns + why, missing checks, categorical risk_level, summary)
# but deliberately EXCLUDE the verbatim code `evidence` fragments: those are
# normalized VAR/FUN tokens the code encoder already sees, and the 2026-07-04
# evidence_tokens ablation showed such fragments dilute an NL encoder. Keeping
# them out gives the explanation channel its cleanest shot at contributing
# something orthogonal to the code channel.
# ===========================================================================
def v2_text(e) -> str:
    if not isinstance(e, dict):
        return ""
    parts = []
    so = e.get("structural_observations") or []
    if so:
        parts.append(" ".join(str(x) for x in so))
    si = e.get("safety_indicators") or []
    checks = [str(d.get("check", "")).strip() for d in si if isinstance(d, dict)]
    checks = [c for c in checks if c]
    if checks:
        parts.append("Safety checks present: " + "; ".join(checks) + ".")
    ro = e.get("risky_operations") or []
    risks = []
    for d in ro:
        if not isinstance(d, dict):
            continue
        p, w = str(d.get("pattern", "")).strip(), str(d.get("why", "")).strip()
        seg = (p + " — " + w).strip(" —") if (p or w) else ""
        if seg:
            risks.append(seg)
    if risks:
        parts.append("Risky operations: " + "; ".join(risks) + ".")
    mc = [str(x).strip() for x in (e.get("missing_checks") or [])]
    mc = [m for m in mc if m]
    if mc:
        parts.append("Missing checks: " + "; ".join(mc) + ".")
    rl = e.get("risk_level")
    if rl:
        parts.append(f"Overall risk level: {rl}.")
    rs = e.get("risk_summary")
    if rs:
        parts.append(str(rs).strip())
    return " ".join(p for p in parts if p).strip()


def _norm_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _rank_auc(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rs_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rs_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_v2_rows():
    rows = []
    with open(EXPL_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def standalone_signal(rows):
    """Sanity check on the generated subset: does the label-free explanation
    signal separate classes AT ALL here? (v1 ~0.51 = chance; gate ~0.58.)"""
    risk, nrisky, labels = [], [], []
    valid, grounded, total, empty0, empty1, n0, n1 = 0, 0, 0, 0, 0, 0, 0
    for r in rows:
        e = r.get("explanation") or {}
        lab = r.get("label")
        if lab not in (0, 1) or not isinstance(e, dict) or e.get("risk_level") not in RISK_MAP:
            continue
        valid += 1
        ro = e.get("risky_operations") or []
        code_n = _norm_ws(r.get("raw_code", ""))
        for op in ro:
            ev = op.get("evidence", "") if isinstance(op, dict) else ""
            total += 1
            if _norm_ws(ev) and _norm_ws(ev) in code_n:
                grounded += 1
        risk.append(RISK_MAP[e["risk_level"]])
        nrisky.append(len(ro))
        labels.append(lab)
        if lab == 0:
            n0 += 1
            empty0 += 1 if RISK_MAP[e["risk_level"]] == 0 else 0
        else:
            n1 += 1
            empty1 += 1 if RISK_MAP[e["risk_level"]] == 0 else 0
    auc_risk = _rank_auc(risk, labels) if labels else float("nan")
    auc_nrisky = _rank_auc(nrisky, labels) if labels else float("nan")
    print("\n[signal] standalone label-free explanation signal on this subset:")
    print(f"  valid={valid}  grounding={grounded}/{total} "
          f"({100*grounded/max(1,total):.1f}%) verbatim-in-code")
    print(f"  risk_level=none : label0 {100*empty0/max(1,n0):.0f}%  "
          f"label1 {100*empty1/max(1,n1):.0f}%")
    print(f"  SIGNAL AUC: risk_level={auc_risk:.3f}  n_risky={auc_nrisky:.3f}  "
          f"(v1 ~0.51, gate ~0.58)")
    return {"signal_auc_risk_level": auc_risk, "signal_auc_n_risky": auc_nrisky,
            "grounding_rate": grounded / max(1, total), "valid": valid}


# ===========================================================================
# Phase 3/4 — train one rung (adapted from fusevul_ladder/train.py), evaluate
# on the shared internal test split. L1 and L2 receive IDENTICAL indices.
# ===========================================================================
def _tok(tokenizer, texts, max_len):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def _best_thr_f1(prob1, y):
    import numpy as np
    from sklearn.metrics import f1_score
    best, bs = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        s = f1_score(y, (prob1 >= t).astype(int), zero_division=0)
        if s > bs:
            bs, best = s, float(t)
    return best


def _metrics_at(thr, prob1, y):
    import numpy as np
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    yh = (prob1 >= thr).astype(int)
    return dict(threshold=round(float(thr), 3),
                acc=100 * accuracy_score(y, yh),
                f1=100 * f1_score(y, yh, zero_division=0),
                prec=100 * precision_score(y, yh, zero_division=0),
                rec=100 * recall_score(y, yh, zero_division=0))


def train_eval(rung, packed, fit_idx, tune_idx, test_idx, args):
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 f1_score)
    from transformers import AutoModel, AutoTokenizer
    from model import LadderModel                      # reference ladder model

    t0 = time.time()
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"\n[{rung}] device={device} amp={'bf16' if bf16 else 'fp16'} "
          f"fit={len(fit_idx)} tune={len(tune_idx)} test={len(test_idx)}", flush=True)

    code_tok = AutoTokenizer.from_pretrained(CODE_ID)
    text_tok = AutoTokenizer.from_pretrained(TEXT_ID)
    code_enc = AutoModel.from_pretrained(CODE_ID)
    text_enc = AutoModel.from_pretrained(TEXT_ID)
    qual_dim = packed["qual"].shape[1]
    model = LadderModel(code_enc, text_enc, qual_dim=qual_dim,
                        rung=rung, fusion="self").to(device)
    model.enable_grad_checkpointing()

    ci, cm = _tok(code_tok, packed["code"], args.max_code)
    ti, tm = _tok(text_tok, packed["expl"], args.max_text)
    q = torch.from_numpy(packed["qual"])
    y = packed["y"]

    fit_idx = np.asarray(fit_idx)
    tune_idx = np.asarray(tune_idx)
    test_idx = np.asarray(test_idx)
    ytu, yte = y[tune_idx], y[test_idx]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not bf16))

    @torch.no_grad()
    def prob1(idx):
        model.eval()
        idx = torch.as_tensor(np.asarray(idx))
        outs = []
        for i in range(0, len(idx), max(2, args.batch)):
            b = idx[i:i + max(2, args.batch)]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                lo = model(ci[b].to(device), cm[b].to(device), ti[b].to(device),
                           tm[b].to(device), q[b].to(device))
            outs.append(torch.softmax(lo.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(outs)

    best_ap, best = -1.0, None
    wait = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(fit_idx)
        opt.zero_grad()
        losses = []
        for si, i in enumerate(range(0, len(perm), args.batch)):
            b = torch.as_tensor(perm[i:i + args.batch])
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
                logits = model(ci[b].to(device), cm[b].to(device), ti[b].to(device),
                               tm[b].to(device), q[b].to(device))
                yb = torch.as_tensor(y[perm[i:i + args.batch]], dtype=torch.long,
                                     device=device)
                loss = nn.functional.cross_entropy(logits, yb) / args.grad_accum
            scaler.scale(loss).backward()
            if (si + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
            losses.append(loss.item() * args.grad_accum)

        tu_p = prob1(tune_idx)
        te_p = prob1(test_idx)
        ap = average_precision_score(ytu, tu_p) if ytu.sum() > 0 else 0.0
        te_f1 = f1_score(yte, (te_p >= 0.5).astype(int), zero_division=0) * 100
        print(f"[{rung}] ep{ep}/{args.epochs} loss={np.mean(losses):.4f} "
              f"tune_prauc={ap*100:.2f} test_f1@0.5={te_f1:.2f} "
              f"test_roc={roc_auc_score(yte, te_p)*100:.2f}", flush=True)
        if ap > best_ap:
            best_ap, best, wait = ap, (ep, te_p, tu_p), 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[{rung}] early stop @ep{ep}", flush=True)
                break

    ep_best, te_p, tu_p = best
    thr = _best_thr_f1(tu_p, ytu)
    out = {
        "rung": rung, "best_epoch": ep_best,
        "test_roc_auc": 100 * roc_auc_score(yte, te_p),
        "test_pr_auc": 100 * average_precision_score(yte, te_p),
        "argmax": _metrics_at(0.5, te_p, yte),
        "tuned_on_tune": _metrics_at(thr, te_p, yte),
        "seconds": round(time.time() - t0, 1),
    }
    np.savez_compressed(os.path.join(OUT, f"pilot_{rung}_probs.npz"),
                        test_prob=te_p, test_y=yte, tune_prob=tu_p, tune_y=ytu)
    a = out["argmax"]
    print(f"[{rung}] DONE @ep{ep_best} ROC={out['test_roc_auc']:.2f} "
          f"PR={out['test_pr_auc']:.2f} argmax acc={a['acc']:.2f} f1={a['f1']:.2f} "
          f"({out['seconds']/60:.1f} min)", flush=True)
    del model, code_enc, text_enc
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# ===========================================================================
# Build the aligned, packed dataset from generated rows + shared split.
# ===========================================================================
def build_packed(rows):
    import numpy as np
    rows = [r for r in rows if r.get("label") in (0, 1)]
    code = [r.get("raw_code", "") or "" for r in rows]
    expl = [v2_text(r.get("explanation") or {}) for r in rows]
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    qual = np.zeros((len(rows), 22), dtype=np.float32)  # L1/L2 ignore qual
    empties = sum(1 for e in expl if not e)
    lens = [len(e.split()) for e in expl if e]
    med = sorted(lens)[len(lens) // 2] if lens else 0
    print(f"[data] packed={len(rows)} (pos {100*y.mean():.1f}%)  "
          f"expl empty={empties}  median expl words={med}", flush=True)
    return {"code": code, "expl": expl, "y": y, "qual": qual}


def make_split(y, test_frac, tune_frac, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(y)
    test_mask = np.zeros(n, dtype=bool)
    for c in (0, 1):                                   # stratified test
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_frac)))
        test_mask[idx[:k]] = True
    rem = np.where(~test_mask)[0]
    tune_mask = np.zeros(n, dtype=bool)
    for c in (0, 1):                                   # stratified tune from remainder
        idx = rem[y[rem] == c]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * tune_frac)))
        tune_mask[idx[:k]] = True
    fit_idx = np.where(~test_mask & ~tune_mask)[0]
    tune_idx = np.where(tune_mask)[0]
    test_idx = np.where(test_mask)[0]
    return fit_idx, tune_idx, test_idx


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N_SUBSET)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX)
    ap.add_argument("--timeout", type=int, default=GEN_TIMEOUT)
    ap.add_argument("--think", dest="no_think", action="store_false",
                    help="allow model thinking (default sends think:false)")
    ap.set_defaults(no_think=NO_THINK)
    ap.add_argument("--skip-gen", action="store_true",
                    help="skip generation; only run the L1-vs-L2 comparison")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    ap.add_argument("--max-code", type=int, default=MAX_CODE)
    ap.add_argument("--max-text", type=int, default=MAX_TEXT)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    args = ap.parse_args()

    print("=" * 78)
    print("v2-explanation FUSION pilot  (L1 code-only  vs  L2 code + v2-expl fusion)")
    print(f"  subset N={args.n}  model={args.model}  host={args.host}")
    print(f"  code_enc={CODE_ID}  text_enc={TEXT_ID}")
    print(f"  artifacts under: {OUT}   (delete this folder to remove everything)")
    print("=" * 78, flush=True)

    # Phase 0 + 1 — subset + generation
    subset = select_subset(args.n, GEN_SEED)
    if not args.skip_gen:
        generate(subset, args)
    if not os.path.exists(EXPL_JSONL):
        print("[abort] no explanations file; generation produced nothing.", flush=True)
        sys.exit(1)

    rows = load_v2_rows()
    if len(rows) < 100:
        print(f"[abort] only {len(rows)} explanations generated; run generation first.",
              flush=True)
        sys.exit(1)

    # Phase 1b — standalone signal sanity check
    sig = standalone_signal(rows)

    # Phase 2 — pack + shared split (identical for both rungs)
    packed = build_packed(rows)
    fit_idx, tune_idx, test_idx = make_split(packed["y"], TEST_FRAC, TUNE_FRAC,
                                             SPLIT_SEED)

    # Phase 3/4 — L1 then L2 on the SAME data/split/seed
    res_l1 = train_eval("L1", packed, fit_idx, tune_idx, test_idx, args)
    res_l2 = train_eval("L2", packed, fit_idx, tune_idx, test_idx, args)

    # Phase 5 — verdict table
    d_roc = res_l2["test_roc_auc"] - res_l1["test_roc_auc"]
    d_pr = res_l2["test_pr_auc"] - res_l1["test_pr_auc"]
    d_f1 = res_l2["argmax"]["f1"] - res_l1["argmax"]["f1"]
    d_tf1 = res_l2["tuned_on_tune"]["f1"] - res_l1["tuned_on_tune"]["f1"]
    summary = {
        "config": {"n": args.n, "model": args.model, "code_enc": CODE_ID,
                   "text_enc": TEXT_ID, "epochs": args.epochs,
                   "test_frac": TEST_FRAC, "tune_frac": TUNE_FRAC,
                   "n_test": int(len(test_idx)), "n_fit": int(len(fit_idx))},
        "standalone_signal": sig,
        "L1_code_only": res_l1,
        "L2_code_plus_v2expl": res_l2,
        "deltas_L2_minus_L1": {"roc_auc": d_roc, "pr_auc": d_pr,
                               "f1_argmax": d_f1, "f1_tuned": d_tf1},
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 78)
    print("VERDICT  (test set, n_test = %d) — threshold-free ROC/PR is the fair measure"
          % len(test_idx))
    print("-" * 78)
    print(f"{'metric':<16}{'L1 code-only':>16}{'L2 +v2-expl':>16}{'delta (L2-L1)':>16}")
    print(f"{'ROC-AUC':<16}{res_l1['test_roc_auc']:>16.2f}"
          f"{res_l2['test_roc_auc']:>16.2f}{d_roc:>+16.2f}")
    print(f"{'PR-AUC':<16}{res_l1['test_pr_auc']:>16.2f}"
          f"{res_l2['test_pr_auc']:>16.2f}{d_pr:>+16.2f}")
    print(f"{'F1 @0.5':<16}{res_l1['argmax']['f1']:>16.2f}"
          f"{res_l2['argmax']['f1']:>16.2f}{d_f1:>+16.2f}")
    print(f"{'F1 tuned':<16}{res_l1['tuned_on_tune']['f1']:>16.2f}"
          f"{res_l2['tuned_on_tune']['f1']:>16.2f}{d_tf1:>+16.2f}")
    print(f"{'acc @0.5':<16}{res_l1['argmax']['acc']:>16.2f}"
          f"{res_l2['argmax']['acc']:>16.2f}"
          f"{res_l2['argmax']['acc']-res_l1['argmax']['acc']:>+16.2f}")
    print("-" * 78)
    print(f"standalone explanation signal AUC (risk_level) = "
          f"{sig['signal_auc_risk_level']:.3f}  (v1 ~0.51, gate ~0.58)")
    print(f"full results -> {RESULTS_JSON}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
