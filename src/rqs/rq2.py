"""RQ2: quality-aware ADAPTIVE GATED FUSION over FROZEN, CACHED pooled embeddings.

Answers RQ2 ("how does quality-aware adaptive gated fusion compare with static
fusion, single-modality models, and classical cached-feature baselines?") on the
regime RO2 names: lightweight heads over frozen, cached encoders.

Pipeline
  1) build_cache(ds): run frozen CodeT5+ / RoBERTa ONCE over train+val, mean-pool
     token states to h_c, h_e in R^768; store with label-free quality features
     (set B, 5 dims) + labels. ~150 MB, built once, reused by every variant.
  2) train tiny fusion heads on the cache (SECONDS each):
       code_only     head(h_c)                              single-modality (code)
       text_only     head(h_e)                              single-modality (explanation)
       static_avg    head(0.5*(h_c+h_e))                    static fusion (the gate's g=0.5 twin)
       static_concat head([h_c;h_e])                        static fusion (concat)
       gated         g=sig(MLP([h_c;h_e;qual])); g*h_c+(1-g)*h_e   quality-aware ADAPTIVE
       gated_noqual  g=sig(MLP([h_c;h_e]))                  ABLATION: does qual add anything?
  3) report acc/F1/ROC/PR table + gate g-vs-grounding mechanism (proves quality-awareness).

Parity: gate bias init 0 -> g=0.5 -> gated STARTS exactly at static_avg, so the
gated-vs-static_avg delta is attributable only to the gate learning to move g.

    python -m src.rqs.rq2 --dataset devign
    python -m src.rqs.rq2 --dataset reveal --subset 2000   # fast smoke
"""
import os
import sys
import json
import glob
import time
import argparse
from datetime import datetime, timezone

# env MUST be set before importing project data (data.py binds the qual builder
# at import time based on SEMVUL_QUAL_V2).
os.environ.setdefault("SEMVUL_ACTIVE_DIR", "1")
os.environ["SEMVUL_QUAL_V2"] = "0"                       # v1-only (label-free, no leaky v2 block)
os.environ.setdefault("SEMVUL_QUAL_SET", "B")            # grounding + specificity (5 feats)
os.environ.setdefault("SEMVUL_EXPL_FIELDS",
    "confidence,risky_operations,missing_checks,function_name,"
    "called_functions,risky_apis,risk_summary,purpose")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, os.path.join(ROOT, "experiments", "fusevul_ladder")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import data as data_mod
from train import _load_code_encoder, _tok, TEXT_ID
from model import focal_ce
from transformers import AutoTokenizer, AutoModel

CACHE_DIR = os.path.join(ROOT, "experiments", "runs", "rq2_cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODES = ["code_only", "text_only", "static_avg", "static_concat",
         "gated", "gated_noqual", "refine", "refine_noqual"]
CLASSICAL = ["logreg", "rf", "svm"]        # classical cached-feature baselines (RO2)
ALL_MODES = MODES + CLASSICAL
REFINE_PEN = 1e-3   # do-no-harm prior: penalize the L3 residual delta magnitude
ENC_SHORT = {"codet5p": "c5", "graphcodebert": "gcb",
             "microsoft/unixcoder-base": "uxc", "unixcoder": "uxc",
             "Salesforce/codet5p-220m": "c5b220", "codet5p-220m": "c5b220"}


def _enc_short(e):
    return ENC_SHORT.get(e) or "".join(ch for ch in e if ch.isalnum()).lower()[:8]


POOLS = ("mean", "cls", "max")


def _pool_all(h, mask):
    """Return {mean, cls, max} pooled vectors (all d-dim). 'cls' = token-0, the
    position codet5p-embedding's trained projection actually reads (see its
    modeling: normalize(proj(last_hidden_state[:,0]))) -- so cls tests using the
    encoder's intended representation vs the generic mean-pool."""
    m = mask.unsqueeze(-1).to(h.dtype)
    mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
    cls = h[:, 0, :]
    mx = torch.nan_to_num(h.masked_fill(m == 0, float("-inf")).max(1).values, neginf=0.0)
    return {"mean": mean, "cls": cls, "max": mx}


@torch.no_grad()
def _encode(enc, kind, ids, mask, bs=16):
    """Encode once; return {pool: [N, d] np.array} for every pool in POOLS."""
    enc.eval().to(DEVICE)
    amp = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    acc = {p: [] for p in POOLS}
    for i in range(0, len(ids), bs):
        s = slice(i, i + bs)
        with torch.autocast("cuda", dtype=amp, enabled=DEVICE == "cuda"):
            if kind == "t5":
                h = enc.encoder(input_ids=ids[s].to(DEVICE),
                                attention_mask=mask[s].to(DEVICE)).last_hidden_state
            else:
                h = enc(input_ids=ids[s].to(DEVICE),
                        attention_mask=mask[s].to(DEVICE)).last_hidden_state
        pooled = _pool_all(h.float(), mask[s].to(DEVICE))
        for p in POOLS:
            acc[p].append(pooled[p].cpu())
    return {p: torch.cat(acc[p]).numpy() for p in POOLS}


def _source_mtime(ds):
    """Newest mtime among the ACTIVE jsonl inputs that actually feed this
    dataset's cache. Returns 0.0 if none are found, so the guard degrades safely
    to plain reuse (never a false 'stale')."""
    try:
        from src.data_io import _active_path
    except Exception:
        return 0.0
    mt = 0.0
    for split in ("train", "val"):
        try:
            p = _active_path(ds, split)
            if p.exists():
                mt = max(mt, p.stat().st_mtime)
        except Exception:
            pass
    return mt


def build_cache(ds, subset=None, max_code=320, max_text=512, tag="", code_enc="codet5p"):
    """Frozen pooled-embedding cache WITH a staleness guard.

    Every build is written to a UTC date-stamped file (..__YYYYmmddTHHMMSSZ.npz)
    so nothing is silently overwritten and each cache's provenance is on disk.
    Reuse picks the NEWEST cache for this (dataset, subset, tag) -- but only when
    it is newer than the ACTIVE source jsonl. Regenerate the enrichment (or touch
    the source) and the stale cache is ignored and a fresh one built AUTOMATICALLY.
    The generated path is printed to the console.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    stem = f"{ds}_n{subset or 'full'}_{_enc_short(code_enc)}{sfx}"
    src_mt = _source_mtime(ds)

    # Candidates: stamped files for this stem + any legacy unstamped file, so a
    # pre-existing fresh cache is still reused (no needless recompute on upgrade).
    cands = glob.glob(os.path.join(CACHE_DIR, f"{stem}__*.npz"))
    legacy = os.path.join(CACHE_DIR, f"{stem}.npz")
    if os.path.exists(legacy):
        cands.append(legacy)
    newest = max(cands, key=os.path.getmtime) if cands else None

    if newest and os.path.getmtime(newest) >= src_mt:
        print(f"[cache] reuse (fresh) {newest}", flush=True)
        return newest
    if newest:
        import time as _t
        print(f"[cache] STALE -> rebuild: {os.path.basename(newest)} "
              f"({_t.ctime(os.path.getmtime(newest))}) is older than ACTIVE "
              f"{ds} source ({_t.ctime(src_mt)})", flush=True)

    tr, va = data_mod.load(ds, subset=subset)
    code_enc_m, code_tok, code_kind = _load_code_encoder(code_enc)
    text_tok = AutoTokenizer.from_pretrained(TEXT_ID)
    text_enc = AutoModel.from_pretrained(TEXT_ID)

    def enc_split(d, name):
        print(f"[cache] encoding {ds} {name} (n={len(d['y'])}) enc={code_enc}...", flush=True)
        ci, cm = _tok(code_tok, d["code"], max_code)
        ti, tm = _tok(text_tok, d["expl"], max_text)
        return _encode(code_enc_m, code_kind, ci, cm), _encode(text_enc, "", ti, tm)

    hc_tr, he_tr = enc_split(tr, "train")     # each is {pool: [N, d]}
    hc_va, he_va = enc_split(va, "val")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(CACHE_DIR, f"{stem}__{stamp}.npz")
    save = {"q_tr": tr["qual"], "y_tr": tr["y"], "q_va": va["qual"], "y_va": va["y"],
            "built_at": stamp, "source_mtime": np.float64(src_mt)}
    for p in POOLS:
        save[f"hc_{p}_tr"], save[f"he_{p}_tr"] = hc_tr[p], he_tr[p]
        save[f"hc_{p}_va"], save[f"he_{p}_va"] = hc_va[p], he_va[p]
    np.savez_compressed(path, **save)
    print(f"[cache] GENERATED {path}", flush=True)
    print(f"[cache]   built_at(UTC)={stamp}  pools={list(POOLS)} "
          f"hc{hc_tr['mean'].shape} he{he_tr['mean'].shape} q{tr['qual'].shape}", flush=True)
    return path


class Fusion(nn.Module):
    def __init__(self, mode, d=768, qd=5, hidden=256, dropout=0.3):
        super().__init__()
        self.mode = mode
        self.ln_c, self.ln_e = nn.LayerNorm(d), nn.LayerNorm(d)
        gate_in = {"gated": 2 * d + qd, "gated_noqual": 2 * d}.get(mode)
        if gate_in:
            self.gate = nn.Sequential(nn.Linear(gate_in, 128), nn.GELU(), nn.Linear(128, d))
            # bias 0 -> g=0.5 -> gated starts EXACTLY at static_avg (clean parity).
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, 0.0)
        # L3 residual refinement: refined = static_concat(hc,he) + alpha * delta.
        # delta MLP last layer AND alpha are ZERO-init, so refine STARTS as the L2
        # static-concat representation and learns only a correction (do-no-harm
        # prior, reinforced by REFINE_PEN). Strictly more expressive than the convex
        # gate: delta is unconstrained, not confined to conv(hc, he).
        refine_in = {"refine": 2 * d + qd, "refine_noqual": 2 * d}.get(mode)
        if refine_in:
            self.refine = nn.Sequential(
                nn.Linear(refine_in, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, 2 * d))
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)
            # alpha init 1.0 (NOT 0): the delta MLP's last layer is already zero-init,
            # so delta == 0 at init and refine STILL starts at the L2 base -- but a
            # nonzero alpha keeps gradient flowing to the MLP. alpha AND the MLP both
            # starting at 0 multiplicatively deadlock (zero gradient to both).
            self.alpha = nn.Parameter(torch.ones(1))
        head_in = 2 * d if mode in ("static_concat", "refine", "refine_noqual") else d
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, hidden),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, hc, he, q):
        hc, he = self.ln_c(hc), self.ln_e(he)
        if self.mode == "code_only":
            z = hc
        elif self.mode == "text_only":
            z = he
        elif self.mode == "static_avg":
            z = 0.5 * (hc + he)
        elif self.mode == "static_concat":
            z = torch.cat([hc, he], -1)
        elif self.mode in ("refine", "refine_noqual"):
            fused = torch.cat([hc, he], -1)                  # the L2 static-concat base
            rin = torch.cat([hc, he, q], -1) if self.mode == "refine" \
                else torch.cat([hc, he], -1)
            self._delta = self.alpha * self.refine(rin)      # [B, 2d] residual correction
            self._refine_pen = self._delta.pow(2).mean()     # do-no-harm penalty term
            z = fused + self._delta                          # refined = L2 + delta
        else:  # gated / gated_noqual
            gin = torch.cat([hc, he, q], -1) if self.mode == "gated" else torch.cat([hc, he], -1)
            self._g = torch.sigmoid(self.gate(gin))          # [B, d] per-sample per-dim
            z = self._g * hc + (1.0 - self._g) * he
        return self.head(z)


def _tune_split(y, frac=0.12, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    tu = np.zeros(len(y), bool)
    for c in (0, 1):
        ci = idx[y == c]
        rng.shuffle(ci)
        tu[ci[:max(1, int(frac * len(ci)))]] = True
    return ~tu, tu


def _metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(acc=round(100 * accuracy_score(y, pred), 2),
                f1=round(100 * f1_score(y, pred, zero_division=0), 2),
                prec=round(100 * precision_score(y, pred, zero_division=0), 2),
                rec=round(100 * recall_score(y, pred), 2),
                roc=round(100 * roc_auc_score(y, p), 2),
                pr=round(100 * average_precision_score(y, p), 2))


def train_variant(C, mode, ds, seed=1, epochs=60, patience=8, lr=1e-3, batch=512, pool="mean",
                   return_probs=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    y = C["y_tr"]
    q = torch.tensor(C["q_tr"], dtype=torch.float32)
    mu, sd = q.mean(0, keepdim=True), q.std(0, keepdim=True).clamp_min(1e-6)
    q = (q - mu) / sd
    hc, he = torch.tensor(C[f"hc_{pool}_tr"]), torch.tensor(C[f"he_{pool}_tr"])
    qv = (torch.tensor(C["q_va"], dtype=torch.float32) - mu) / sd
    hcv, hev, yv = torch.tensor(C[f"hc_{pool}_va"]), torch.tensor(C[f"he_{pool}_va"]), C["y_va"]
    trm, tum = _tune_split(y, seed=seed)
    tr_idx, tu_idx = np.where(trm)[0], np.where(tum)[0]

    model = Fusion(mode, d=int(hc.shape[1]), qd=int(q.shape[1])).to(DEVICE)   # d,qd adapt
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    use_focal = (ds == "reveal")
    pos = float(y[tr_idx].mean())
    alpha = float(np.clip(1 - pos, 0.5, 0.85))

    def prob(idx_or_tensors):
        model.eval()
        with torch.no_grad():
            hc_, he_, q_ = idx_or_tensors
            lo = model(hc_.to(DEVICE), he_.to(DEVICE), q_.to(DEVICE))
            return torch.softmax(lo.float(), -1)[:, 1].cpu().numpy()

    best_ap, best_vp, wait = -1.0, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), batch):
            b = torch.as_tensor(perm[i:i + batch])
            lo = model(hc[b].to(DEVICE), he[b].to(DEVICE), q[b].to(DEVICE))
            yb = torch.as_tensor(y[perm[i:i + batch]], dtype=torch.long, device=DEVICE)
            loss = focal_ce(lo, yb, alpha_pos=alpha, gamma=2.0) if use_focal \
                else nn.functional.cross_entropy(lo, yb)
            if mode in ("refine", "refine_noqual"):
                loss = loss + REFINE_PEN * model._refine_pen   # keep delta small unless it earns it
            opt.zero_grad()
            loss.backward()
            opt.step()
        tu_p = prob((hc[torch.as_tensor(tu_idx)], he[torch.as_tensor(tu_idx)],
                     q[torch.as_tensor(tu_idx)]))
        ap = average_precision_score(y[tu_idx], tu_p) if y[tu_idx].sum() else 0.0
        if ap > best_ap:
            best_ap, wait = ap, 0
            best_vp = prob((hcv, hev, qv))
        else:
            wait += 1
            if wait >= patience:
                break

    res = {"mode": mode, **_metrics(yv, best_vp), "sec": round(time.time() - t0, 2)}
    if return_probs:
        res["_val_p"] = best_vp   # per-sample val probs; caller's responsibility to strip before json.dump
    if mode in ("gated", "gated_noqual"):
        model.eval()
        with torch.no_grad():
            model(hcv.to(DEVICE), hev.to(DEVICE), qv.to(DEVICE))
            gd = model._g.float().cpu().numpy()              # [Nval, d]
        g = gd.mean(1)                                        # per-sample mean gate
        res["gate"] = {
            "g_mean": round(float(g.mean()), 4), "g_std": round(float(g.std()), 4),
            "g_perdim_std": round(float(gd.std(0).mean()), 4),
            "g": [round(float(x), 5) for x in g],
            "grounding_overlap": [round(float(x), 5) for x in C["q_va"][:, 0]],
        }
    elif mode in ("refine", "refine_noqual"):
        model.eval()
        with torch.no_grad():
            model(hcv.to(DEVICE), hev.to(DEVICE), qv.to(DEVICE))
            dl = model._delta.float().cpu().numpy()          # [Nval, 2d] residual correction
        dn = np.linalg.norm(dl, axis=1)                       # per-sample delta magnitude
        res["refine"] = {
            "alpha": round(float(model.alpha.detach().cpu().item()), 5),
            "delta_norm_mean": round(float(dn.mean()), 5),
            "delta_norm_std": round(float(dn.std()), 5),
            "grounding_overlap": [round(float(x), 5) for x in C["q_va"][:, 0]],
        }
    return res


def train_classical(C, kind, seed=1, pool="mean"):
    """Classical cached-feature baseline (RO2 evidence): an sklearn classifier on
    the SAME frozen inputs the gated fusion sees -- [h_c ; h_e ; quality] -- so it
    is a fair 'classical vs neural fusion' comparison. class_weight balances the
    rare positive class (the classical analog of the neural focal loss)."""
    t0 = time.time()
    def feats(sp):
        return np.concatenate([C[f"hc_{pool}_{sp}"], C[f"he_{pool}_{sp}"], C[f"q_{sp}"]], 1)
    Xtr, Xva = feats("tr"), feats("va")
    ytr, yva = C["y_tr"], C["y_va"]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=-1)
    else:  # svm (linear; sigmoid of the margin gives a score for ROC/PR)
        clf = LinearSVC(class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(Xva)[:, 1]
    else:
        p = 1.0 / (1.0 + np.exp(-clf.decision_function(Xva)))
    return {"mode": kind, **_metrics(yva, p), "sec": round(time.time() - t0, 2)}


def _aggregate(runs):
    """Mean/std across seeds for one variant. Scalar metric MEANS keep the same
    keys (so the delta prints below keep working) and gain a '<k>_std' sibling."""
    out = {"mode": runs[0]["mode"], "n_seeds": len(runs)}
    for k in ("acc", "f1", "prec", "rec", "roc", "pr", "sec"):
        vals = [float(r[k]) for r in runs if k in r]
        out[k] = round(float(np.mean(vals)), 2)
        out[k + "_std"] = round(float(np.std(vals)), 2)
    if "gate" in runs[0]:
        out["gate"] = {kk: round(float(np.mean([r["gate"][kk] for r in runs])), 4)
                       for kk in ("g_mean", "g_std", "g_perdim_std")}
    if "refine" in runs[0]:
        out["refine"] = {kk: round(float(np.mean([r["refine"][kk] for r in runs])), 4)
                         for kk in ("alpha", "delta_norm_mean")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["devign", "reveal"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", default=None,
                    help="comma list for a multi-seed sweep with mean+/-std "
                         "(e.g. 1,3,7,13,21); overrides --seed. Heads train off the "
                         "one-time cache, so N seeds cost ~N*seconds, not *encode.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--fields", default=None,
                    help="override the explanation TEXT channel (SEMVUL_EXPL_FIELDS); "
                         "auto-tags a fresh cache so it never silently reuses another "
                         "channel's embeddings (e.g. drop 'confidence' for leakage-clean).")
    ap.add_argument("--tag", default="", help="explicit cache/results suffix")
    ap.add_argument("--qual", choices=["b", "rich"], default="b",
                    help="quality feature set fed to the gate/refine: 'b'=current 5-feat "
                         "set B; 'rich'=expanded label-free set (code complexity + "
                         "explanation specificity/hedging/length). Auto-tags a separate cache.")
    ap.add_argument("--code-enc", default="codet5p",
                    help="frozen code encoder (RO2 encoder comparison): codet5p, "
                         "graphcodebert, or any HF id (microsoft/unixcoder-base, "
                         "Salesforce/codet5p-220m). Separate cache per encoder.")
    ap.add_argument("--pool", choices=list(POOLS), default="mean",
                    help="pooling of frozen token states: mean (default), cls (token-0, "
                         "codet5p's native embedding position), or max. All pools are "
                         "cached in one encode; --pool just selects at train time.")
    args = ap.parse_args()

    tag = args.tag
    if args.qual == "rich":
        os.environ["SEMVUL_QUAL_RICH"] = "1"                 # must precede build_cache/load
        if not tag:
            tag = "qv3rich"
    if args.fields:
        os.environ["SEMVUL_EXPL_FIELDS"] = args.fields
        if not tag:
            import hashlib
            tag = "f" + hashlib.md5(args.fields.encode()).hexdigest()[:6]
    print(f"[rq2] text channel = {os.environ['SEMVUL_EXPL_FIELDS']}")
    print(f"[rq2] quality set  = {args.qual}")
    print(f"[rq2] code encoder = {args.code_enc}   pool = {args.pool}")
    print(f"[rq2] cache tag    = {tag or '(default)'}")

    path = build_cache(args.dataset, subset=args.subset, tag=tag, code_enc=args.code_enc)
    C = dict(np.load(path))
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print(f"\n=== RQ2 {args.dataset} | frozen {args.code_enc}+roberta | {args.pool}-pool "
          f"| seeds {seeds} ===")
    print(f"train={len(C['y_tr'])} (pos {100*C['y_tr'].mean():.1f}%)  val={len(C['y_va'])}\n")

    def ms(m, k):                                     # format "62.34+/-0.51"
        return f"{m[k]:.2f}+/-{m.get(k + '_std', 0.0):.2f}"

    cols = ("acc", "f1", "roc", "pr")
    hdr = f"{'variant':15s}" + "".join(f"{h:>14s}" for h in ("acc", "F1", "ROC", "PR"))
    print(hdr)
    print("-" * len(hdr))
    results, per_seed = {}, {}
    for mode in ALL_MODES:
        if mode in CLASSICAL:
            runs = [train_classical(C, mode, seed=s, pool=args.pool) for s in seeds]
        else:
            runs = [train_variant(C, mode, args.dataset, seed=s, epochs=args.epochs, pool=args.pool)
                    for s in seeds]
        per_seed[mode] = runs
        m = _aggregate(runs)
        results[mode] = m
        print(f"{mode:15s}" + "".join(f"{ms(m, k):>14s}" for k in cols))
    print("-" * len(hdr))

    g = results["gated"].get("gate", {})
    print(f"gate g: mean={g.get('g_mean')} per-dim-std={g.get('g_perdim_std')}  "
          f"(per-dim-std~0 => inert)")
    rf = results["refine"].get("refine", {})
    _inert = abs(rf.get("alpha", 0.0)) < 0.05 and rf.get("delta_norm_mean", 0.0) < 0.1
    print(f"[L3 residual] alpha={rf.get('alpha')} delta_norm_mean={rf.get('delta_norm_mean')}"
          f"  ({'reverted to L2 (inert)' if _inert else 'residual ACTIVE'})\n")

    def d(a, b, k):
        return results[a][k] - results[b][k]
    print(f"{'comparison (mean delta over seeds)':48s}{'dAcc':>8s}{'dF1':>8s}{'dROC':>8s}{'dPR':>8s}")
    pairs = [
        ("adaptive vs L2:  gated  - static_concat", "gated", "static_concat"),
        ("adaptive vs L2:  refine - static_concat", "refine", "static_concat"),
        ("classical vs L2: logreg - static_concat", "logreg", "static_concat"),
        ("classical vs L2: rf     - static_concat", "rf", "static_concat"),
        ("classical vs L2: svm    - static_concat", "svm", "static_concat"),
        ("no-qual  vs L2:  gated_noqual - static_concat", "gated_noqual", "static_concat"),
        ("fusion vs code:  static_concat - code_only", "static_concat", "code_only"),
        ("QUALITY cost:    gated  - gated_noqual", "gated", "gated_noqual"),
        ("QUALITY cost:    refine - refine_noqual", "refine", "refine_noqual"),
    ]
    for label, a, b in pairs:
        print(f"{label:48s}{d(a,b,'acc'):>8.2f}{d(a,b,'f1'):>8.2f}"
              f"{d(a,b,'roc'):>8.2f}{d(a,b,'pr'):>8.2f}")
    roc_std = float(np.mean([results[m].get("roc_std", 0.0) for m in MODES]))
    print(f"\nseed noise: mean ROC std across variants = {roc_std:.2f} "
          f"(treat |dROC| below ~{2*roc_std:.2f} as noise)")
    print("train efficiency (mean s/variant): " +
          "  ".join(f"{m}={results[m].get('sec', '?')}" for m in ALL_MODES))

    def _lean(r):                                     # drop per-sample arrays before saving
        r = dict(r)
        for mk in ("gate", "refine"):
            if mk in r:
                r[mk] = {k: v for k, v in r[mk].items() if not isinstance(v, list)}
        return r
    rsfx = f"_{_enc_short(args.code_enc)}_{args.pool}" + (f"_{tag}" if tag else "")
    out = os.path.join(CACHE_DIR, f"{args.dataset}_rq2_results{rsfx}.json")
    results["_meta"] = {
        "cache": os.path.basename(path),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "text_channel": os.environ["SEMVUL_EXPL_FIELDS"],
        "code_enc": args.code_enc, "pool": args.pool,
        "seeds": seeds,
    }
    results["_per_seed"] = {m: [_lean(r) for r in per_seed[m]] for m in ALL_MODES}
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n[files] cache   = {path}")
    print(f"[files] results = {out}")


if __name__ == "__main__":
    main()
