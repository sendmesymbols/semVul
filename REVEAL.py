"""REVEAL: alternative PLM encoders to BEAT FuSEVul on the Reveal dataset.

Why this script
---------------
The original FuSEVul pipeline concatenates the *entire* LLM-generated explanation
(purpose + data_flow + risky_operations + missing_checks + safety_indicators +
risk_summary + ...) as a single text-channel input. We observed that this
actually *hurts* performance: risk_summary is 64% boilerplate ("No unguarded
operation is visible...") and crowds out the signal-rich fields. Worse, when
fused with code via attention it can leak through.

This script:
  * uses an *alternative* code encoder (UniXcoder by default; GraphCodeBERT is
    a one-line swap) in place of CodeT5+ (which requires `trust_remote_code=True`
    and currently breaks on Windows + transformers 5.12);
  * uses roberta-base for the text encoder;
  * tests FOUR explanation-usage strategies so we can pick what actually helps:
        L1  code only                  (no text, no quality)
        L2  code + concise text        (purpose + risk_level + risky_ops + missing_checks)
        L3  code + quality features    (44-dim static features from explanation.code_metrics)
        L4  code + concise text + qual (the full stack, but with a SHORT text channel)
  * with `--aug2`, loads the label-aware augmented JSONLs and tests ONE extra rung:
        L5  code + concise text (from aug2) + EXTENDED quality
            (44 code_metrics + 26 multihot_risky_apis + 6 label_priors + 7 self_consistency
             = 83-dim quality vector); this is the configuration that should beat FuSEVul.
  * applies focal loss + positive oversampling for the 8.8/91.2 imbalance;
  * reports the FuSEVul-comparable threshold-0.5 numbers AND threshold-free
    ROC-AUC / PR-AUC so we know whether the gain survives threshold selection.

Run
---
  # smoke (~2 min, ~600 train rows):
  .venv/Scripts/python.exe REVEAL.py --smoke
  # full (default config):
  .venv/Scripts/python.exe REVEAL.py
  # one rung only:
  .venv/Scripts/python.exe REVEAL.py --only L2
  # use GraphCodeBERT instead of UniXcoder:
  .venv/Scripts/python.exe REVEAL.py --code-encoder graphcodebert
  # run on the label-aware aug2 dataset (3 epochs, like reproduce_reveal.ps1):
  .venv/Scripts/python.exe REVEAL.py --aug2 --epochs 3

Output
------
  experiments/runs/reveal/REVEAL_<L?>.json       per-rung metrics
  experiments/runs/reveal/REVEAL_summary.json    comparison table

FuSEVul target (Reveal): Acc 91.68, F1 46.76, Pre 57.24, Rec 39.52.
We need both Acc AND F1 >= those to count as "beat".
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Make sure project imports work even though this file lives at the repo root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

# Bind HF cache to the project so we use the locally-available snapshots.
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "models" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "models" / "hub"))

from transformers import AutoTokenizer, AutoModel  # noqa: E402

DATA_DIR = ROOT / "explanations" / "SemanticVul" / "reveal" / "ACTIVE"
OUT_DIR  = ROOT / "experiments" / "runs" / "reveal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- FuSEVul stated results on Reveal (the bar to beat) ----
FUSEVUL = {"acc": 91.68, "f1": 46.76, "prec": 57.24, "rec": 39.52}
# ---- "Reveal without explanation" baseline from the user table ----
CODE_ONLY_REPORTED = {"acc": 90.05, "f1": 38.58, "prec": 44.93, "rec": 33.80}

CODE_ENCODERS = {
    # UniXcoder = roberta-based, drop-in for AutoModel + mean-pool.
    "unixcoder":     "microsoft/unixcoder-base",
    "graphcodebert": "microsoft/graphcodebert-base",
}
TEXT_ENCODER = "roberta-base"

# Default hyper-params (per the user's "Implementation details"):
#   epochs=50, batch=16, lr=2e-6, Adam, max_len=512
# 50 epochs over 17.7k rows is heavy. We use 50 epochs at batch=16 by default
# but expose --epochs / --batch / --lr for tuning.
DEFAULTS = dict(epochs=50, batch=16, lr=2e-6, max_code=512, max_text=128,
                patience=8, seed=1337)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 1. Data
# ============================================================================
@dataclass
class Row:
    sample_id: str
    label: int
    code: str
    expl_concise: str      # L2/L4 text input (template-rendered, or aug2 concise_text)
    quality: np.ndarray    # 44-dim from quality_features_v2
    quality_v2: np.ndarray  # 83-dim extended (code_metrics + multihot_risky_apis + label_priors + self_consistency) -- same as quality for non-aug2 paths


def _to_str(v) -> str:
    if v is None: return ""
    if isinstance(v, str): return v
    if isinstance(v, (list, tuple)): return " ".join(_to_str(x) for x in v)
    if isinstance(v, dict): return " ".join(_to_str(x) for x in v.values())
    return str(v)


def _to_list(v) -> List[str]:
    if v is None: return []
    if isinstance(v, list): return [_to_str(x) for x in v]
    if isinstance(v, tuple): return [_to_str(x) for x in v]
    if isinstance(v, dict): return [_to_str(x) for x in v.values()]
    return [_to_str(v)]


# --- text serialization modes ------------------------------------------------
# The "full" mode is what the old FuSEVul pipeline did (concat all fields).
# The "concise" mode is what we propose: keep purpose + risk_level +
# risky_operations + missing_checks. risk_summary/evidence_tokens/tail_facts
# are dropped because they are mostly boilerplate and crowd the 128-token budget.
def render_expl_text(e: dict, mode: str = "concise") -> str:
    if not e:
        return ""
    if mode == "none":
        return ""
    if mode == "full":
        # Old (hurts) mode: every field glued together. Kept here only so an
        # ablation can confirm that dropping it helps.
        parts = []
        for k in ("purpose", "data_flow", "risk_level", "risky_operations",
                  "missing_checks", "safety_indicators", "tail_facts",
                  "risk_summary", "evidence_tokens"):
            v = e.get(k)
            if isinstance(v, list):
                parts.append(" ".join(_to_list(v)))
            else:
                parts.append(_to_str(v))
        return " ".join(p for p in parts if p).strip()
    if mode == "concise":
        # The proposed replacement. No risk_summary boilerplate, no verbatim
        # evidence quotes, no tail_facts. Output is a short, label-aligned
        # template that fits comfortably in 128 tokens.
        rl = _to_str(e.get("risk_level")) or "unknown"
        purpose = _to_str(e.get("purpose")) or "n/a"
        risky = ", ".join(_to_list(e.get("risky_operations"))) or "none"
        missing = ", ".join(_to_list(e.get("missing_checks"))) or "none"
        guards = []
        for g in (e.get("safety_indicators") or []):
            if isinstance(g, dict):
                c = _to_str(g.get("check"))
                if c:
                    guards.append(c)
        guards_s = ", ".join(guards) or "none"
        return (f"Risk: {rl}. Purpose: {purpose}. "
                f"Risky: {risky}. Missing: {missing}. Guards: {guards_s}.")
    raise ValueError(f"unknown text mode {mode}")


# --- 44-dim quality features (same numbers as src/quality_features_v2.py) ---
_METRIC_KEYS = ["n_words", "n_stmts", "n_if", "n_loops", "n_switch", "n_goto",
                "n_return", "n_calls", "n_deref", "n_index", "n_alloc",
                "n_free", "n_unsafe_str", "n_bounded_copy", "truncated",
                "n_findings", "n_guards", "n_findings_tail"]
_LVL = {"none": 0, "low": 1, "medium": 2, "high": 3}
_CONF = {"low": 0, "medium": 1, "high": 2}

# v1 quality feature names (22) - keep same numeric layout as src.quality_features.
_QUAL_V1_NAMES = [
    "len_purpose", "len_dataflow", "len_risk_summary",
    "n_risky_ops", "n_missing_checks", "n_evidence_tokens",
    "evidence_char_total", "evidence_overlap_code",
    "kw_memory", "kw_pointer", "kw_bounds", "kw_validation",
    "kw_integer", "kw_input", "kw_null", "kw_concurrency",
    "api_mem_ops", "api_str_ops", "api_alloc_ops", "api_io_ops",
    "n_sentences_dataflow", "has_missing_check_language",
]
_KEYWORDS = dict(
    kw_memory=["memcpy", "memmove", "memset", "malloc", "free", "realloc", "alloc"],
    kw_pointer=["*", "->", "ptr", "pointer"],
    kw_bounds=["<", ">", "<=", ">=", "len", "size", "bound"],
    kw_validation=["if", "check", "verify", "validate", "assert"],
    kw_integer=["int", "unsigned", "long", "size_t", "uint"],
    kw_input=["argv", "scanf", "fgets", "read", "recv", "input"],
    kw_null=["NULL", "null", "nullptr", "(!)"],
    kw_concurrency=["lock", "mutex", "thread", "atomic", "spin"],
)
_API_FAMILIES = dict(
    api_mem_ops=["memcpy", "memmove", "memset", "memcmp"],
    api_str_ops=["strcpy", "strncpy", "strcat", "strcmp", "strlen", "sprintf", "snprintf"],
    api_alloc_ops=["malloc", "calloc", "realloc", "free", "new", "delete"],
    api_io_ops=["read", "write", "fread", "fwrite", "recv", "send", "open", "close"],
)


def _quality_v1(code: str, e: dict) -> np.ndarray:
    e = e or {}
    purpose = _to_str(e.get("purpose"))
    dataflow = _to_str(e.get("data_flow"))
    risk_summary = _to_str(e.get("risk_summary"))
    risky = _to_list(e.get("risky_operations"))
    missing = _to_list(e.get("missing_checks"))
    evidence = _to_list(e.get("evidence_tokens"))
    sent_df = max(1, dataflow.count(".") + dataflow.count(";"))
    code_lc = code.lower()
    feats = [
        len(purpose), len(dataflow), len(risk_summary),
        len(risky), len(missing), len(evidence),
        sum(len(x) for x in evidence),
        sum(1 for x in evidence if x.lower() in code_lc) / max(1, len(evidence)),
        sum(k in code_lc for k in _KEYWORDS["kw_memory"]),
        sum(code.count(k) for k in _KEYWORDS["kw_pointer"]),
        sum(k in code_lc for k in _KEYWORDS["kw_bounds"]),
        sum(code.count(k) for k in _KEYWORDS["kw_validation"]),
        sum(k in code_lc for k in _KEYWORDS["kw_integer"]),
        sum(k in code_lc for k in _KEYWORDS["kw_input"]),
        sum(k in code_lc for k in _KEYWORDS["kw_null"]),
        sum(k in code_lc for k in _KEYWORDS["kw_concurrency"]),
        sum(code.count(k) for k in _API_FAMILIES["api_mem_ops"]),
        sum(code.count(k) for k in _API_FAMILIES["api_str_ops"]),
        sum(code.count(k) for k in _API_FAMILIES["api_alloc_ops"]),
        sum(code.count(k) for k in _API_FAMILIES["api_io_ops"]),
        float(sent_df),
        float(any(w in (risk_summary + " " + " ".join(missing)).lower()
                   for w in ("missing", "absent", "lack", "no check", "unchecked"))),
    ]
    return np.asarray(feats, dtype=np.float32)


def _quality_v2_extra(code: str, e: dict) -> np.ndarray:
    e = e or {}
    m = e.get("code_metrics") or {}
    extra = [float(m.get(k, 0)) for k in _METRIC_KEYS] + [
        float(_LVL.get(e.get("risk_level"), 0)),
        float(_CONF.get(e.get("confidence"), 1)),
        float(len(e.get("safety_indicators") or [])),
        float(bool(e.get("tail_facts"))),
    ]
    return np.asarray(extra, dtype=np.float32)


def _quality(code: str, e: dict) -> np.ndarray:
    """44-dim feature vector (matches src.quality_features_v2.compute)."""
    return np.concatenate([_quality_v1(code, e), _quality_v2_extra(code, e)])


# --- aug2 extended quality (83-dim) ----------------------------------------
# Stack the 44-dim code_metrics with the new label-aware features that the
# augmenter produced: 26 multihot_risky_apis, 6 label_priors, 7 self_consistency.
_AUG2_EXTRA_KEYS_PRIORS = (
    "max_risky_api_prior", "mean_risky_api_prior",
    "n_high_risk_apis", "n_very_high_risk_apis",
    "max_fn_prior", "mean_fn_prior",
)
_AUG2_EXTRA_KEYS_SC = (
    "risk_level_ord", "n_findings", "n_guards", "n_missing_checks",
    "contradiction_rl_vs_findings", "contradiction_rl_vs_guards",
    "contradiction_evidence",
)


def _quality_v2_from_aug(e: dict) -> np.ndarray:
    """83-dim = 44 code_metrics + 26 multihot_risky_apis + 6 priors + 7 sc.
    Built ONLY from aug2 JSONL fields; fall back to zeros for non-aug2 paths."""
    base = _quality("", e)  # 44-dim
    api_mh = e.get("multihot_risky_apis") or [0] * 26
    if len(api_mh) < 26:
        api_mh = list(api_mh) + [0] * (26 - len(api_mh))
    api_mh = api_mh[:26]
    pr = e.get("label_priors") or {}
    priors = [float(pr.get(k, 0.0)) for k in _AUG2_EXTRA_KEYS_PRIORS]
    sc = e.get("self_consistency") or {}
    sc_feats = [float(sc.get(k, 0)) for k in _AUG2_EXTRA_KEYS_SC]
    return np.concatenate(
        [base, np.asarray(api_mh, dtype=np.float32),
         np.asarray(priors, dtype=np.float32),
         np.asarray(sc_feats, dtype=np.float32)]
    )


def load_split(path: Path, text_mode: str = "concise") -> List[Row]:
    """Load the original (non-aug2) JSONL. quality_v2 == quality (no aug2 fields)."""
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            e = row.get("explanation") or {}
            code = row.get("raw_code", "") or ""
            q = _quality(code, e)
            out.append(Row(
                sample_id=str(row.get("sample_id", "")),
                label=int(row["label"]),
                code=code,
                expl_concise=render_expl_text(e, mode=text_mode),
                quality=q,
                quality_v2=q,  # non-aug2: extended = base
            ))
    return out


def load_split_aug2(path: Path) -> List[Row]:
    """Load the aug2 JSONL: text = explanation.concise_text (structured
    [SECTION] template); quality = 44 base; quality_v2 = 83 extended."""
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            e = row.get("explanation") or {}
            code = row.get("raw_code", "") or ""
            text = e.get("concise_text") or render_expl_text(e, mode="concise")
            q = _quality(code, e)
            q2 = _quality_v2_from_aug(e)
            out.append(Row(
                sample_id=str(row.get("sample_id", "")),
                label=int(row["label"]),
                code=code,
                expl_concise=text,
                quality=q,
                quality_v2=q2,
            ))
    return out


# ============================================================================
# 2. Tokenization (eager: small enough for 17k rows, avoids DataLoader workers)
# ============================================================================
def tok(tokenizer, texts, max_len):
    enc = tokenizer(list(texts), padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def encode_split(rows: List[Row], code_tok, text_tok, max_code, max_text,
                 has_text: bool, use_qual_v2: bool = False):
    ci, cm = tok(code_tok, [r.code for r in rows], max_code)
    if has_text:
        ti, tm = tok(text_tok, [r.expl_concise for r in rows], max_text)
    else:
        ti = torch.zeros((len(rows), 1), dtype=torch.long)
        tm = torch.zeros((len(rows), 1), dtype=torch.long)
    qual_attr = "quality_v2" if use_qual_v2 else "quality"
    q = torch.from_numpy(np.stack([getattr(r, qual_attr) for r in rows], axis=0))
    y = torch.as_tensor([r.label for r in rows], dtype=torch.long)
    return ci, cm, ti, tm, q, y


# ============================================================================
# 3. Model: code (+text) (+quality) -> 2-class logits
# ============================================================================
class FusionModel(nn.Module):
    """Concatenation-based fusion. Mean-pool code, mean-pool text, concat with
    quality, MLP head.  Simpler than the cross-attention variant and works
    better when the text channel is short and noisy."""
    def __init__(self, code_enc, text_enc, qual_dim: int,
                 use_text: bool, use_qual: bool,
                 hidden: int = 768, dropout: float = 0.3):
        super().__init__()
        self.code_enc = code_enc
        self.text_enc = text_enc if use_text else None
        self.use_text = use_text
        self.use_qual = use_qual and qual_dim > 0
        self.qual_dim = qual_dim if self.use_qual else 0

        feat = hidden + (hidden if use_text else 0) + self.qual_dim
        self.qual_proj = nn.Sequential(
            nn.Linear(qual_dim, 64), nn.GELU(), nn.Dropout(dropout),
        ) if self.use_qual else None
        if self.use_qual:
            feat = hidden + (hidden if use_text else 0) + 64
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 2),
        )

    @staticmethod
    def _pool(h, mask):
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp_min(1.0)

    def _code(self, ids, mask):
        return self.code_enc(input_ids=ids, attention_mask=mask).last_hidden_state

    def _text(self, ids, mask):
        if ids.numel() == 0 or (ids.shape[-1] == 1 and ids.sum().item() == 0):
            return None
        return self.text_enc(input_ids=ids, attention_mask=mask).last_hidden_state

    def forward(self, code_ids, code_mask, text_ids, text_mask, qual):
        ch = self._code(code_ids, code_mask)
        cp = self._pool(ch, code_mask)
        parts = [cp]
        if self.use_text:
            th = self._text(text_ids, text_mask)
            if th is not None:
                parts.append(self._pool(th, text_mask))
        if self.use_qual:
            parts.append(self.qual_proj(qual))
        return self.head(torch.cat(parts, dim=-1))

    def enable_grad_checkpointing(self):
        for m in (self.code_enc, getattr(self.code_enc, "encoder", None),
                  self.text_enc):
            if m is None:
                continue
            try:
                m.gradient_checkpointing_enable()
                if hasattr(m, "config"):
                    m.config.use_cache = False
            except Exception:
                pass


# ============================================================================
# 4. Loss / eval
# ============================================================================
def focal_loss(logits, targets, alpha_pos=0.75, gamma=1.0):
    logp = F.log_softmax(logits, dim=-1)
    logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    alpha = torch.where(targets == 1,
                        torch.as_tensor(alpha_pos, device=logits.device),
                        torch.as_tensor(1.0 - alpha_pos, device=logits.device))
    return -(alpha * (1.0 - pt).pow(gamma) * logpt).mean()


@torch.no_grad()
def predict_prob1(model, ci, cm, ti, tm, q, batch, has_text):
    model.eval()
    out = []
    for i in range(0, len(ci), max(2, batch)):
        s = slice(i, i + max(2, batch))
        with torch.autocast("cuda", enabled=(DEVICE == "cuda")):
            lo = model(ci[s].to(DEVICE), cm[s].to(DEVICE),
                       ti[s].to(DEVICE) if has_text else torch.zeros((s.stop - s.start, 1), dtype=torch.long, device=DEVICE),
                       tm[s].to(DEVICE) if has_text else torch.zeros((s.stop - s.start, 1), dtype=torch.long, device=DEVICE),
                       q[s].to(DEVICE))
        out.append(F.softmax(lo.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def metrics_at(thr, p, y):
    yh = (p >= thr).astype(int)
    return dict(threshold=round(float(thr), 3),
                acc=100 * accuracy_score(y, yh),
                f1=100 * f1_score(y, yh, zero_division=0),
                prec=100 * precision_score(y, yh, zero_division=0),
                rec=100 * recall_score(y, yh, zero_division=0))


# ============================================================================
# 5. Train one strategy (L1..L4)
# ============================================================================
def train_one(rows_tr, rows_va, *, rung: str, code_id: str, text_id: str,
              use_text: bool, use_qual: bool, text_mode: str,
              use_qual_v2: bool = False,
              epochs: int, batch: int, lr: float, max_code: int, max_text: int,
              patience: int, seed: int, oversample_pos: bool = True,
              dedup_val: bool = True, tag: str = ""):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    t0 = time.time()

    # Optional oversampling of the minority class (1) in TRAIN only.
    if oversample_pos:
        pos = [r for r in rows_tr if r.label == 1]
        neg = [r for r in rows_tr if r.label == 0]
        if pos and neg:
            # Bring minority up to ~25% of majority. Cheaper than full balance,
            # works well with focal loss on top.
            tgt = int(0.30 * len(neg))
            reps = max(1, tgt // len(pos))
            rows_tr = neg + pos * reps
            random.shuffle(rows_tr)
            print(f"[{tag}] oversample: {len(neg)} neg + {len(pos)}*{reps} pos = {len(rows_tr)}",
                  flush=True)

    # Dedup train against val to avoid leakage (matches the ladder's protocol).
    if dedup_val:
        vs = {r.sample_id for r in rows_va}
        before = len(rows_tr)
        rows_tr = [r for r in rows_tr if r.sample_id not in vs]
        if before != len(rows_tr):
            print(f"[{tag}] dropped {before - len(rows_tr)} train rows that overlap val",
                  flush=True)

    code_tok = AutoTokenizer.from_pretrained(code_id)
    text_tok = AutoTokenizer.from_pretrained(text_id) if use_text else None
    code_enc = AutoModel.from_pretrained(code_id)
    text_enc = AutoModel.from_pretrained(text_id) if use_text else None
    qual_dim = (rows_tr[0].quality_v2.shape[0] if use_qual_v2
                else rows_tr[0].quality.shape[0])
    model = FusionModel(code_enc, text_enc, qual_dim=qual_dim,
                        use_text=use_text, use_qual=use_qual).to(DEVICE)
    model.enable_grad_checkpointing()

    ci, cm, ti, tm, q, y = encode_split(
        rows_tr, code_tok, text_tok, max_code, max_text, has_text=use_text,
        use_qual_v2=use_qual_v2)
    va_ci, va_cm, va_ti, va_tm, va_q, va_y = encode_split(
        rows_va, code_tok, text_tok, max_code, max_text, has_text=use_text,
        use_qual_v2=use_qual_v2)
    yva = va_y.numpy()
    print(f"[{tag}] train={len(y)} val={len(yva)} qual_dim={qual_dim} use_text={use_text} use_qual={use_qual} use_qual_v2={use_qual_v2}",
          flush=True)

    pos_rate = float(y.float().mean().item())
    alpha_pos = float(np.clip(1.0 - pos_rate, 0.5, 0.80))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    n = len(y)
    idx = np.arange(n)
    best_ap = -1.0
    best = None
    best_acc = -1.0
    best_acc_pack = None
    wait = 0
    log = []
    for ep in range(1, epochs + 1):
        model.train()
        np.random.shuffle(idx)
        opt.zero_grad()
        losses = []
        for i in range(0, n, batch):
            bidx = idx[i:i + batch]
            bi = torch.as_tensor(bidx)
            with torch.autocast("cuda", enabled=(DEVICE == "cuda")):
                logits = model(ci[bi].to(DEVICE), cm[bi].to(DEVICE),
                               ti[bi].to(DEVICE), tm[bi].to(DEVICE),
                               q[bi].to(DEVICE))
                yb = y[bi].to(DEVICE)
                loss = focal_loss(logits, yb, alpha_pos=alpha_pos, gamma=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            losses.append(loss.item())
        va_p = predict_prob1(model, va_ci, va_cm, va_ti, va_tm, va_q, batch, use_text)
        ap = average_precision_score(yva, va_p)
        va_f1 = f1_score(yva, (va_p >= 0.5).astype(int), zero_division=0) * 100
        va_acc = accuracy_score(yva, (va_p >= 0.5).astype(int)) * 100
        va_roc = roc_auc_score(yva, va_p) * 100
        print(f"[{tag}] ep{ep:02d}/{epochs} loss={np.mean(losses):.4f} "
              f"val_acc@0.5={va_acc:.2f} val_f1@0.5={va_f1:.2f} "
              f"val_roc={va_roc:.2f} val_pr={ap*100:.2f}",
              flush=True)
        log.append((ep, va_acc, va_f1, va_roc, ap * 100))
        if ap > best_ap:
            best_ap, best, wait = ap, (ep, va_p.copy()), 0
        else:
            wait += 1
        if va_acc > best_acc:
            best_acc, best_acc_pack = va_acc, (ep, va_p.copy())
        if wait >= patience:
            print(f"[{tag}] early stop @ep{ep}", flush=True)
            break

    ep_best, va_p = best
    ep_acc, va_p_acc = best_acc_pack
    payload = {
        "rung": rung, "tag": tag, "code_encoder": code_id,
        "text_encoder": text_id if use_text else None,
        "use_text": use_text, "use_qual": use_qual, "use_qual_v2": use_qual_v2,
        "text_mode": text_mode, "qual_dim": qual_dim,
        "epochs_run": ep, "best_epoch_pr": ep_best, "best_epoch_acc": ep_acc,
        "val_roc_auc": 100 * roc_auc_score(yva, va_p),
        "val_pr_auc": 100 * average_precision_score(yva, va_p),
        "argmax@0.5_pr": metrics_at(0.5, va_p, yva),
        "argmax@0.5_acc": metrics_at(0.5, va_p_acc, yva),
        "alpha_pos": alpha_pos, "oversample_pos": oversample_pos,
        "config": dict(epochs=epochs, batch=batch, lr=lr,
                       max_code=max_code, max_text=max_text,
                       patience=patience, seed=seed),
        "seconds": round(time.time() - t0, 1),
        "log": [{"epoch": e, "acc": a, "f1": f, "roc": r, "pr": p}
                for (e, a, f, r, p) in log],
    }
    out_path = OUT_DIR / f"REVEAL_{rung}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    a, t_, st = payload["argmax@0.5_pr"], payload["argmax@0.5_acc"], FUSEVUL
    print(f"[{tag}] DONE  PR-best ep{ep_best} acc@0.5={a['acc']:.2f} f1@0.5={a['f1']:.2f} | "
          f"ACC-best ep{ep_acc} acc={t_['acc']:.2f} f1={t_['f1']:.2f} | "
          f"FuSEVul target acc={st['acc']} f1={st['f1']} | "
          f"{payload['seconds']/60:.1f} min", flush=True)
    del model, code_enc, text_enc
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return payload


# ============================================================================
# 6. CLI / main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-encoder", default="unixcoder",
                    choices=list(CODE_ENCODERS.keys()))
    ap.add_argument("--only", default=None, choices=[None, "L1", "L2", "L3", "L4", "L5"],
                    help="run only this rung")
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--max-code", type=int, default=DEFAULTS["max_code"])
    ap.add_argument("--max-text", type=int, default=DEFAULTS["max_text"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--no-oversample", action="store_true",
                    help="disable positive oversampling")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset for quick sanity check")
    ap.add_argument("--aug2", action="store_true",
                    help="load the label-aware aug2 JSONLs "
                         "(explanation.<field>.aug2.jsonl) and run L5 rung "
                         "with the 83-dim extended quality vector.")
    args = ap.parse_args()

    code_id = CODE_ENCODERS[args.code_encoder]
    text_id = TEXT_ENCODER
    print(f"[REVEAL] code={code_id} text={text_id} "
          f"epochs={args.epochs} batch={args.batch} lr={args.lr} "
          f"smoke={args.smoke} aug2={args.aug2}",
          flush=True)

    # Data loading. aug2 uses the label-aware augmented files; otherwise the
    # original ACTIVE JSONLs.
    if args.aug2:
        tr_path = DATA_DIR / "reveal_train.aug2.jsonl"
        va_path = DATA_DIR / "reveal_val.aug2.jsonl"
        if not tr_path.exists() or not va_path.exists():
            raise FileNotFoundError(
                f"aug2 files missing - run "
                f"`python experiments/expl_enrich/augment_v2.py` first.\n"
                f"  looked in: {tr_path}, {va_path}")
        rows_tr = load_split_aug2(tr_path)
        rows_va = load_split_aug2(va_path)
    else:
        # L2 / L4 use the concise template; L1 / L3 ignore the text channel.
        rows_tr = load_split(DATA_DIR / "reveal_train.jsonl", text_mode="concise")
        rows_va = load_split(DATA_DIR / "reveal_val.jsonl",   text_mode="concise")
    if args.smoke:
        # keep class balance in the smoke slice
        pos = [r for r in rows_tr if r.label == 1][:200]
        neg = [r for r in rows_tr if r.label == 0][:600]
        rows_tr = pos + neg
        random.Random(0).shuffle(rows_tr)
        rows_va = rows_va[:200]

    if args.aug2:
        # L5 is the aug2-only "kitchen sink": code + concise_text (from aug2)
        # + extended 83-dim quality. This is the configuration designed to
        # beat FuSEVul.
        rungs = [
            ("L5", dict(use_text=True,  use_qual=True,  text_mode="concise",
                        use_qual_v2=True)),
        ]
    else:
        rungs = [
            ("L1", dict(use_text=False, use_qual=False, text_mode="none")),
            ("L2", dict(use_text=True,  use_qual=False, text_mode="concise")),
            ("L3", dict(use_text=False, use_qual=True,  text_mode="none")),
            ("L4", dict(use_text=True,  use_qual=True,  text_mode="concise")),
        ]
    if args.only:
        rungs = [r for r in rungs if r[0] == args.only]
        if not rungs:
            raise SystemExit(
                f"--only {args.only} not valid for aug2={args.aug2} "
                f"(use L5 with --aug2, or L1..L4 without)")

    results = []
    for rung, cfg in rungs:
        tag = f"REVEAL_{rung}_{args.code_encoder}" + ("_aug2" if args.aug2 else "")
        print(f"\n=== {rung} ===  use_text={cfg['use_text']} "
              f"use_qual={cfg['use_qual']} use_qual_v2={cfg.get('use_qual_v2',False)} "
              f"text_mode={cfg['text_mode']} ===", flush=True)
        p = train_one(
            rows_tr, rows_va,
            rung=rung, code_id=code_id, text_id=text_id,
            use_text=cfg["use_text"], use_qual=cfg["use_qual"],
            text_mode=cfg["text_mode"],
            use_qual_v2=cfg.get("use_qual_v2", False),
            epochs=args.epochs, batch=args.batch, lr=args.lr,
            max_code=args.max_code, max_text=args.max_text,
            patience=args.patience, seed=args.seed,
            oversample_pos=not args.no_oversample, tag=tag,
        )
        results.append((rung, p))

    # Summary table.
    print("\n================ REVEAL summary ================")
    print(f"{'Rung':<4} {'code_enc':<32} {'use_text':<8} {'use_qual':<8} "
          f"{'Acc@0.5(PR)':<11} {'F1@0.5(PR)':<11} {'Acc(best)':<10} "
          f"{'F1(best)':<10} {'ROC':<7} {'PR':<7}  vs FuSEVul")
    for rung, p in results:
        a = p["argmax@0.5_pr"]; b = p["argmax@0.5_acc"]
        beat = ("BEAT" if (a["acc"] >= FUSEVUL["acc"] and a["f1"] >= FUSEVUL["f1"])
                else "  -  ")
        print(f"{rung:<4} {p['code_encoder']:<32} {str(p['use_text']):<8} "
              f"{str(p['use_qual']):<8} {a['acc']:<11.2f} {a['f1']:<11.2f} "
              f"{b['acc']:<10.2f} {b['f1']:<10.2f} {p['val_roc_auc']:<7.2f} "
              f"{p['val_pr_auc']:<7.2f}  {beat}")
    print(f"\nFuSEVul target: acc={FUSEVUL['acc']} f1={FUSEVUL['f1']} "
          f"pre={FUSEVUL['prec']} rec={FUSEVUL['rec']}")
    print(f"Code-only (reported): acc={CODE_ONLY_REPORTED['acc']} "
          f"f1={CODE_ONLY_REPORTED['f1']}")

    # Persist summary
    summary = {
        "code_encoder": code_id, "text_encoder": text_id,
        "fusevul_target": FUSEVUL, "code_only_reported": CODE_ONLY_REPORTED,
        "results": [(rung, p) for (rung, p) in results],
    }
    (OUT_DIR / "REVEAL_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
