"""ENTRY POINT: train one gated-fusion head from cached features.

CLI:
  python -m src.train --dataset devign --ladder L1 [--tag full]
                      [--no-expl] [--no-qual] [--fusion gated|concat]
                      [--seeds 1337 2024 42]

Ladder -> which cached code embeddings to concatenate:
  L1: [dataset]_[split]_code_graphcodebert_lora.npz
  L2: L1 + [dataset]_[split]_code_codet5p_frozen.npz
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import CACHE_DIR, HEAD_CFG, RUNS_DIR, QUALITY_FEATURE_NAMES
from src.data_io import load_split
from src.encode_code import cache_path as code_cache
from src.encode_text import cache_path as text_cache
from src.eval import Metrics, report
from src.model import GatedFusion, focal_bce
from src.quality_features import compute_batch


LADDER_CODE_CACHES = {
    "L1": [("graphcodebert", "lora")],
    "L2": [("graphcodebert", "lora"), ("unixcoder", "frozen")],
}


@dataclass
class RunPayload:
    run_id: str
    dataset: str
    ladder: str
    tag: str
    config: dict
    per_seed: List[dict]
    ensemble_metrics: dict


def _load_codes(dataset: str, split: str, ladder: str) -> np.ndarray:
    parts = []
    for enc, tag in LADDER_CODE_CACHES[ladder]:
        p = code_cache(dataset, split, enc, tag)
        if not p.exists():
            raise FileNotFoundError(f"Missing embedding cache: {p} (run train.py to build)")
        parts.append(np.load(p, allow_pickle=True)["embeddings"])
    return np.concatenate(parts, axis=1).astype(np.float32)


def _load_text(dataset: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    p = text_cache(dataset, split)
    d = np.load(p, allow_pickle=True)
    return d["embeddings"].astype(np.float32), d["labels"].astype(np.int64)


def _load_qual(dataset: str, split: str) -> np.ndarray:
    p = CACHE_DIR / f"{dataset}_{split}_quality.npz"
    if p.exists():
        return np.load(p)["quality"].astype(np.float32)
    samples = load_split(dataset, split)
    q = compute_batch(samples).astype(np.float32)
    np.savez_compressed(p, quality=q)
    return q


def _load_sids(dataset: str, split: str) -> np.ndarray:
    """sample_id = sha1(normalized_code)[:16], stored in the code cache."""
    p = code_cache(dataset, split, "graphcodebert", "lora")
    return np.load(p, allow_pickle=True)["sample_ids"].astype(str)


def _dedup_disjoint(sids_tr: np.ndarray, sids_va: np.ndarray):
    """Drop within-train duplicate functions (keep first) and remove from val any
    function that is a within-val duplicate OR already appears in train. Yields
    train/val as disjoint sets of UNIQUE functions, eliminating the ~6% train->val
    leakage baked into the raw Devign split (identical functions in both folds)."""
    seen, tr_keep = set(), []
    for i, s in enumerate(sids_tr):
        if s not in seen:
            seen.add(s); tr_keep.append(i)
    va_seen, va_keep = set(), []
    for i, s in enumerate(sids_va):
        if s in seen or s in va_seen:
            continue
        va_seen.add(s); va_keep.append(i)
    return np.asarray(tr_keep, dtype=int), np.asarray(va_keep, dtype=int)


def _honest_split(y: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified TUNE/TRAIN split by class, deterministic per (frac, seed)."""
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    n_pos = int(round(len(idx_pos) * frac))
    n_neg = int(round(len(idx_neg) * frac))
    tune = np.concatenate([idx_pos[:n_pos], idx_neg[:n_neg]])
    train = np.concatenate([idx_pos[n_pos:], idx_neg[n_neg:]])
    rng.shuffle(tune); rng.shuffle(train)
    return train, tune


def _standardize(train_x: np.ndarray, *others: np.ndarray):
    mu = train_x.mean(0, keepdims=True)
    sd = train_x.std(0, keepdims=True) + 1e-6
    return [(x - mu) / sd for x in (train_x, *others)]


def _one_seed(seed: int, X_tr, X_tu, X_va, y_tr, y_tu, y_va, *,
              code_dim, expl_dim, qual_dim, fusion, use_expl, use_qual,
              alpha_pos):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Ctr, Etr, Qtr = X_tr
    Cvu, Evu, Qvu = X_tu
    Cva, Eva, Qva = X_va

    def to_t(a): return torch.from_numpy(a).float().to(device)
    def y_t(a):  return torch.from_numpy(a).float().to(device)

    model = GatedFusion(
        code_dim=code_dim, expl_dim=expl_dim, qual_dim=qual_dim,
        proj_dim=HEAD_CFG["proj_dim"], hidden=HEAD_CFG["hidden"],
        dropout=HEAD_CFG["dropout"], fusion=fusion,
        use_expl=use_expl, use_qual=use_qual,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=HEAD_CFG["lr"],
                            weight_decay=HEAD_CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HEAD_CFG["epochs"])

    ds = TensorDataset(to_t(Ctr), to_t(Etr), to_t(Qtr), y_t(y_tr))
    dl = DataLoader(ds, batch_size=HEAD_CFG["batch_size"], shuffle=True)

    def _probs(C, E, Q):
        model.eval()
        with torch.no_grad():
            logits = model(to_t(C), to_t(E) if use_expl else None,
                           to_t(Q) if use_qual else None)
            return torch.sigmoid(logits).cpu().numpy()

    best_score, best_state = -1.0, None
    patience = 0
    for ep in range(HEAD_CFG["epochs"]):
        model.train()
        for c, e, q, y in dl:
            logits = model(c, e if use_expl else None, q if use_qual else None)
            loss = focal_bce(logits, y, HEAD_CFG["focal_gamma"], alpha_pos)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        tune_p = _probs(Cvu, Evu, Qvu)
        rep_tune = report(tune_p, y_tu)
        score = rep_tune["max_f1"].f1
        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= HEAD_CFG["early_stop_patience"]:
            break

    model.load_state_dict(best_state)
    p_tune = _probs(Cvu, Evu, Qvu)
    p_val  = _probs(Cva, Eva, Qva)
    rep = report(p_val, y_va, p_tune, y_tu)
    return {k: v.as_dict() for k, v in rep.items()}, p_val, p_tune


def run(dataset: str, ladder: str, tag: str = "full",
        fusion: str = "gated", use_expl: bool = True, use_qual: bool = True,
        seeds=None) -> Path:
    seeds = tuple(seeds or HEAD_CFG["seeds"])

    C_tr_all = _load_codes(dataset, "train", ladder)
    C_va     = _load_codes(dataset, "val",   ladder)
    E_tr_all, y_tr_all = _load_text(dataset, "train")
    E_va,     y_va     = _load_text(dataset, "val")
    Q_tr_all = _load_qual(dataset, "train")
    Q_va     = _load_qual(dataset, "val")

    # --- de-leak: train/val must be disjoint sets of unique functions ---
    sids_tr, sids_va = _load_sids(dataset, "train"), _load_sids(dataset, "val")
    keep_tr, keep_va = _dedup_disjoint(sids_tr, sids_va)
    n_tr0, n_va0 = len(sids_tr), len(sids_va)
    C_tr_all, E_tr_all, Q_tr_all, y_tr_all = (a[keep_tr] for a in (C_tr_all, E_tr_all, Q_tr_all, y_tr_all))
    C_va, E_va, Q_va, y_va = (a[keep_va] for a in (C_va, E_va, Q_va, y_va))
    print(f"[dedup] train {n_tr0}->{len(keep_tr)}  val {n_va0}->{len(keep_va)} "
          f"(dropped {n_tr0 - len(keep_tr)} train dups, {n_va0 - len(keep_va)} val dup/leaked)")

    # honest tune split
    idx_tr, idx_tu = _honest_split(y_tr_all, HEAD_CFG["tune_frac"], HEAD_CFG["tune_seed"])
    C_tr, E_tr, Q_tr, y_tr = C_tr_all[idx_tr], E_tr_all[idx_tr], Q_tr_all[idx_tr], y_tr_all[idx_tr]
    C_tu, E_tu, Q_tu, y_tu = C_tr_all[idx_tu], E_tr_all[idx_tu], Q_tr_all[idx_tu], y_tr_all[idx_tu]

    C_tr, C_tu, C_va = _standardize(C_tr, C_tu, C_va)
    E_tr, E_tu, E_va = _standardize(E_tr, E_tu, E_va)
    Q_tr, Q_tu, Q_va = _standardize(Q_tr, Q_tu, Q_va)

    pos = int((y_tr == 1).sum()); tot = len(y_tr)
    alpha_pos = min(HEAD_CFG["alpha_pos_cap"], (tot - pos) / tot)

    per_seed, seed_probs_val, seed_probs_tune = [], [], []
    for s in seeds:
        rep, p_val, p_tune = _one_seed(
            s, (C_tr, E_tr, Q_tr), (C_tu, E_tu, Q_tu), (C_va, E_va, Q_va),
            y_tr, y_tu, y_va,
            code_dim=C_tr.shape[1], expl_dim=E_tr.shape[1], qual_dim=Q_tr.shape[1],
            fusion=fusion, use_expl=use_expl, use_qual=use_qual,
            alpha_pos=alpha_pos,
        )
        per_seed.append({"seed": int(s), **rep})
        seed_probs_val.append(p_val); seed_probs_tune.append(p_tune)

    ens_val = np.mean(seed_probs_val, axis=0)
    ens_tune = np.mean(seed_probs_tune, axis=0)
    ens_rep = report(ens_val, y_va, ens_tune, y_tu)

    run_id = f"{dataset}_{ladder}_{tag}"
    out = RUNS_DIR / f"{run_id}.json"
    payload = RunPayload(
        run_id=run_id, dataset=dataset, ladder=ladder, tag=tag,
        config=dict(fusion=fusion, use_expl=use_expl, use_qual=use_qual,
                    seeds=list(seeds), alpha_pos=alpha_pos,
                    quality_feature_names=QUALITY_FEATURE_NAMES),
        per_seed=per_seed,
        ensemble_metrics={k: v.as_dict() for k, v in ens_rep.items()},
    )
    # also save ensemble probs for L3 reuse
    np.savez_compressed(RUNS_DIR / f"{run_id}_probs.npz",
                        probs_val=ens_val, y_val=y_va,
                        probs_tune=ens_tune, y_tune=y_tu)

    with out.open("w", encoding="utf-8") as fh:
        json.dump(asdict(payload), fh, indent=2)
    print(f"[train] wrote {out.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["devign", "reveal"])
    ap.add_argument("--ladder", required=True, choices=["L1", "L2"])
    ap.add_argument("--tag", default="full",
                    help="config label, e.g. full / no_expl / no_qual / concat")
    ap.add_argument("--no-expl", action="store_true")
    ap.add_argument("--no-qual", action="store_true")
    ap.add_argument("--fusion", default="gated", choices=["gated", "concat"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()

    run(args.dataset, args.ladder, tag=args.tag,
        fusion=args.fusion, use_expl=not args.no_expl, use_qual=not args.no_qual,
        seeds=args.seeds)


if __name__ == "__main__":
    main()
