"""RQ3 evidence: under the audited Devign/ReVeal splits, how does SemanticVul
compare with FuSEVul and representative baselines in predictive performance,
threshold robustness, and low-resource training feasibility?

Menu-driven (reveal | devign | both). Per dataset, five bundles:

  A) PERFORMANCE BENCHMARK -- 5-seed mean+/-std Accuracy/Precision/Recall/F1/
     PR-AUC for each ladder rung present, with the SemanticVul rung tagged, the
     FuSEVul stated paper numbers as the anchor row, and L1 as the code-only
     baseline. Acc/Prec/Rec/F1 use the PAPER-FAITHFUL protocol
     (base_paper_protocol.by_val_acc.argmax -> best val-acc@0.5 epoch) so they
     are FuSEVul-comparable; PR-AUC from val_pr_auc. A Delta row = SemanticVul -
     FuSEVul on the cells FuSEVul reported, plus that delta as a % of FuSEVul's
     value. Cells are mean +/- SD over n seeds (n = "seeds present" above);
     95% CI is in bundle B, not repeated here. Positive-class prevalence is
     printed so the accuracy column's honesty floor is explicit.

  B) SEED STABILITY -- mean/median/min/max/95%CI of Accuracy & F1 across seeds
     (same paper-faithful protocol as bundle A) -- exposes whether the win
     holds on the WORST seed, not just on average. When L1 and the SemanticVul
     rung are both present, also runs a paired significance test (t-test +
     Wilcoxon signed-rank, matched by seed id) per bundle-A metric.

  C) THRESHOLD SENSITIVITY -- SemanticVul acc/f1/prec/rec across four operating
     points (argmax@0.5, calibrated@0.5, tuned-on-tune@t, tuned-on-val@t) so the
     threshold sensitivity of the headline metric is explicit (legend printed
     inline for readers unfamiliar with the pipeline's threshold protocols).

  D) TRAINING COST & REPRODUCIBILITY -- mean+/-std training seconds, sec/epoch
     (normalizes for seeds that stopped early at different best_epoch), and
     best_epoch per rung (each cache JSON logs `seconds`).

  E) DATASET AUDIT -- audited-split leakage (val-in-train / opposite-label /
     within-val dup), correct_val.py exact-code criteria.

Reads the per-seed cache JSONs under experiments/runs/<prefix>_<ds>_l{1,2,3}_cache/
(--cache-prefix, default 'final'; point at a frozen-cached family to substantiate
the low-resource claim). Data quality reads the base benchmark set
explanations/SemanticVul/<ds>/<ds>_{train,val}.jsonl. Read-only; no GPU; safe on
a partial ladder (missing rungs/seeds are reported, not fatal).

Usage:
    python src/rqs/rq3.py                       # interactive menu
    python src/rqs/rq3.py reveal
    python src/rqs/rq3.py both --rung L2
    python src/rqs/rq3.py reveal --rung L3 --cache-prefix frozen
"""
from __future__ import annotations

import os
import sys
import glob
import json
import argparse
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from src.config import EXPL_DIR, RUNS_DIR

# t(0.975, df) for a small-sample 95% CI half-width (df = n-1); normal past df=10.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

DATASETS = ("reveal", "devign")
RUNGS = ("L1", "L2", "L3")

# Accept both the current "semanticvul_" tag and the historical
# "fusevul_ladder_" tag (pre-rename runs on disk), mirroring aggregate_seeds.py.
_CACHE_TAGS = ("semanticvul", "fusevul_ladder")


def _glob_tagged(cache_dir, ds, rung, suffix=".json"):
    files = set()
    for tag in _CACHE_TAGS:
        files.update(glob.glob(os.path.join(cache_dir, "s*", f"{tag}_{ds}_{rung}{suffix}")))
    return sorted(files)


# FuSEVul's stated benchmark numbers (the anchor row). PR-AUC not reported by the
# paper -> that cell stays blank.
FUSEVUL = {"devign": {"acc": 60.39, "f1": 55.91},
           "reveal": {"acc": 91.68, "f1": 46.76, "prec": 57.24, "rec": 39.52}}

# Headline metrics, paper-faithful protocol (best val-acc@0.5 epoch) so acc/prec/
# rec/f1 are FuSEVul-comparable; PR-AUC is threshold-free.
_BP = "base_paper_protocol.by_val_acc.argmax"
BENCH_METRICS = [
    ("Accuracy",  f"{_BP}.acc",  "acc"),
    ("Precision", f"{_BP}.prec", "prec"),
    ("Recall",    f"{_BP}.rec",  "rec"),
    ("F1",        f"{_BP}.f1",   "f1"),
    ("PR-AUC",    "val_pr_auc",  None),   # None -> FuSEVul did not report it
]

# Operating points for the threshold-robustness bundle.
THRESH_POINTS = [
    ("argmax@0.5",  "argmax"),
    ("calib@0.5",   "calibrated_at_05"),
    ("tuned/tune",  "tuned_on_tune"),
    ("tuned/val",   "tuned_on_val"),
]


# --------------------------------------------------------------------------- IO
def _dig(payload, dotted):
    cur = payload
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


def _rung_payloads(ds, rung, cache_prefix):
    """(cache_name, [payloads]) for <prefix>_<ds>_<rung>_cache across seeds."""
    cache = f"{cache_prefix}_{ds}_{rung.lower()}_cache"
    payloads = []
    for f in _glob_tagged(os.path.join(RUNS_DIR, cache), ds, rung):
        try:
            with open(f, encoding="utf-8") as fh:
                payloads.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            pass
    return cache, payloads


def _present_rungs(ds, cache_prefix):
    return [r for r in RUNGS if _rung_payloads(ds, r, cache_prefix)[1]]


def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _base(ds, split):
    return os.path.join(EXPL_DIR, ds, f"{ds}_{split}.jsonl")


def _active(ds, split):
    return os.path.join(EXPL_DIR, "ACTIVE", ds, f"{split}.jsonl")


def _val_prevalence(ds):
    """Positive fraction of the model-fed val (ACTIVE/<ds>/val.jsonl), or None if
    absent -- this is the val the cache metrics were computed on, so its majority
    rate is the honest accuracy floor for bundle A."""
    p = _active(ds, "val")
    if not os.path.exists(p):
        return None
    rows = _load_jsonl(p)
    if not rows:
        return None
    return sum(int(r.get("label", 0)) for r in rows) / len(rows)


# ---------------------------------------------------------------- small helpers
def _mean_std(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else float("nan"))


def _cell(mu, sd, width=13):
    if mu is None:
        return "--".center(width)
    if sd != sd:                       # single seed -> no std
        return f"{mu:.2f}".center(width)
    return f"{mu:.2f}+-{sd:.2f}".center(width)


def _ci95(a):
    """95% CI half-width (t-distribution, ddof=1); NaN under 2 samples."""
    n = a.size
    if n < 2:
        return float("nan")
    return float(_T95.get(n - 1, 1.96) * a.std(ddof=1) / np.sqrt(n))


def _dist_str(vals, width=32):
    """mean/median/min/max/95%CI across seeds -- shows spread, not just mean+/-std."""
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return "--".center(width)
    ci = _ci95(a)
    ci_s = f"{ci:.2f}" if ci == ci else "--"
    s = f"{a.mean():.2f}/{np.median(a):.2f}/{a.min():.2f}/{a.max():.2f}/+-{ci_s}"
    return s.center(width)


def _pval(x, w=9):
    if x is None or x != x:
        return "--".rjust(w)
    if x < 0.001:
        s = "<.001"
    elif x < 1:
        s = f"{x:.3f}"[1:]           # APA style: drop the leading 0
    else:
        s = "1.000"
    return s.rjust(w)


def _seed_of(path):
    parent = os.path.basename(os.path.dirname(path))
    return int(parent[1:]) if parent.startswith("s") and parent[1:].isdigit() else None


def _rung_seed_payloads(ds, rung, cache_prefix):
    """{seed: payload}, seed-keyed so bundle-B's paired test lines a seed's L1
    value up with the SAME seed's SemanticVul-rung value -- list order alone
    isn't a safe pairing key if a seed is missing from one rung but not the other."""
    cache = f"{cache_prefix}_{ds}_{rung.lower()}_cache"
    out = {}
    for f in _glob_tagged(os.path.join(RUNS_DIR, cache), ds, rung):
        s = _seed_of(f)
        if s is None:
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                out[s] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass
    return out


def _paired_test(ds, rung_a, rung_b, path, cache_prefix):
    """Paired significance for rung_b vs rung_a on one metric path, matched by
    seed id. None if fewer than 2 seeds are common to both rungs. Reports both
    a paired t-test and Wilcoxon signed-rank (needs >=1 non-zero difference)."""
    pa, pb = _rung_seed_payloads(ds, rung_a, cache_prefix), _rung_seed_payloads(ds, rung_b, cache_prefix)
    common = sorted(s for s in pa if s in pb)
    pairs = [(_dig(pa[s], path), _dig(pb[s], path)) for s in common]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xa = np.array([p[0] for p in pairs], dtype=float)
    xb = np.array([p[1] for p in pairs], dtype=float)
    diff = xb - xa
    t_p = w_p = float("nan")
    try:
        t_p = float(ttest_rel(xb, xa).pvalue)
    except (ValueError, ZeroDivisionError):
        pass
    if np.any(diff != 0):
        try:
            w_p = float(wilcoxon(xb, xa).pvalue)
        except ValueError:
            pass
    return {"n": len(pairs), "t_p": t_p, "w_p": w_p}


def _rung_tag(rung, top):
    if rung == top:
        return f"{rung} (SemanticVul)"
    return f"{rung} code-only" if rung == "L1" else rung


# ------------------------------------------------------------ A) benchmark table
def benchmark(ds, present, top, cache_prefix):
    print(f"\n  -- A. Performance Benchmark: 5-seed mean+/-std | Acc/Prec/Rec/F1 = "
          f"argmax@0.5 on best-val-acc epoch\n     (paper-faithful), PR-AUC "
          f"threshold-free (SemanticVul = {top}) --")
    print("    (cells are mean +/- SD across seeds, not 95% CI -- see"
          " B. Seed Stability for CI and significance)")
    head = "    " + "model".ljust(20) + "".join(m.center(13) for m, _, _ in BENCH_METRICS)
    print(head)
    print("    " + "-" * (len(head) - 4))

    # FuSEVul anchor row (stated).
    fu = FUSEVUL[ds]
    row = "    " + "FuSEVul (paper)".ljust(20)
    for _, _, stat_key in BENCH_METRICS:
        v = fu.get(stat_key) if stat_key else None
        row += (f"{v:.2f}".center(13) if v is not None else "--".center(13))
    print(row)

    # Majority / no-skill baseline -- contextualizes the accuracy column (on an
    # imbalanced split, predict-all-negative already scores 1-prevalence).
    prev = _val_prevalence(ds)
    if prev is not None:
        print(f"    positive prevalence (val): {100*prev:.2f}%")
    row = "    " + "majority(all-neg)".ljust(20)
    for label_m, _, _ in BENCH_METRICS:
        if prev is None:
            row += "--".center(13)
        elif label_m == "Accuracy":
            row += f"{100*(1-prev):.2f}".center(13)
        elif label_m == "PR-AUC":                  # no-skill PR-AUC = prevalence
            row += f"{100*prev:.2f}".center(13)
        elif label_m in ("Recall", "F1"):
            row += "0.00".center(13)
        else:                                      # Precision undefined (no TP)
            row += "--".center(13)
    print(row)

    # One row per present rung; the SemanticVul rung tagged.
    for rung in present:
        _, payloads = _rung_payloads(ds, rung, cache_prefix)
        tag = _rung_tag(rung, top)
        row = "    " + tag.ljust(20)
        for _, path, _ in BENCH_METRICS:
            mu, sd = _mean_std([_dig(p, path) for p in payloads])
            row += _cell(mu, sd)
        print(row)

    # Delta: SemanticVul - FuSEVul on the cells FuSEVul reported, in absolute
    # points and as a % of FuSEVul's own value (relative improvement).
    _, top_pl = _rung_payloads(ds, top, cache_prefix)
    row = "    " + "Delta SemVul-FuSE".ljust(20)
    pct_row = "    " + "Delta % (rel. FuSE)".ljust(20)
    for _, path, stat_key in BENCH_METRICS:
        fv = fu.get(stat_key) if stat_key else None
        mu, _ = _mean_std([_dig(p, path) for p in top_pl])
        if fv is not None and mu is not None:
            row += f"{mu-fv:+.2f}".center(13)
            pct_row += (f"{(mu-fv)/fv*100:+.1f}%".center(13) if fv else "--".center(13))
        else:
            row += "--".center(13)
            pct_row += "--".center(13)
    print(row)
    print(pct_row)
    if prev is not None:
        print(f"    (majority = predict all-negative: Accuracy = 1-prevalence "
              f"= {100*(1-prev):.2f}; PR-AUC cell = prevalence, the no-skill floor)")


# ------------------------------------------------ B) seed stability
def seed_distribution(ds, present, top, cache_prefix):
    """mean/median/min/max/95%CI of Accuracy & F1 across seeds (same paper-
    faithful protocol as bundle A) -- exposes whether the win holds on the
    WORST seed, not just on average. When L1 and the SemanticVul rung are both
    present, also runs a paired L1-vs-SemanticVul significance test (t-test +
    Wilcoxon signed-rank, matched by seed id) per bundle-A metric."""
    print(f"\n  -- B. Seed Stability: mean/median/min/max/95%CI "
          f"(Accuracy & F1, protocol as A) --")
    print("    " + "rung".ljust(18) + "n".center(4)
          + "Accuracy m/md/min/max/+-CI".center(34)
          + "F1 m/md/min/max/+-CI".center(34))
    for rung in present:
        _, payloads = _rung_payloads(ds, rung, cache_prefix)
        acc = [_dig(p, f"{_BP}.acc") for p in payloads]
        f1 = [_dig(p, f"{_BP}.f1") for p in payloads]
        print("    " + _rung_tag(rung, top).ljust(18)
              + str(len(payloads)).center(4)
              + _dist_str(acc, 34) + _dist_str(f1, 34))

    if "L1" in present and top in present and top != "L1":
        print(f"\n    L1 vs {top} (SemanticVul) -- paired significance,"
              f" matched by seed id")
        print("    " + "metric".ljust(12) + "n".center(4)
              + "p (paired t-test)".center(20) + "p (Wilcoxon)".center(16))
        for label, path, _ in BENCH_METRICS:
            r = _paired_test(ds, "L1", top, path, cache_prefix)
            if r is None:
                print("    " + label.ljust(12) + "--".center(4)
                      + "--".center(20) + "--".center(16))
            else:
                print("    " + label.ljust(12) + str(r["n"]).center(4)
                      + _pval(r["t_p"], 20) + _pval(r["w_p"], 16))


_THRESH_LEGEND = (
    "    argmax@0.5 : predict positive iff P(vuln) >= 0.5, no calibration/tuning\n"
    "    calib@0.5  : temperature/Platt-calibrated probabilities, then threshold 0.5\n"
    "    tuned/tune : decision threshold optimized on the held-out tune split\n"
    "    tuned/val  : decision threshold optimized directly on val (upper bound,"
    " not deployable as-is)"
)


# --------------------------------------------------- C) threshold sensitivity
def threshold_robustness(ds, top, cache_prefix):
    _, payloads = _rung_payloads(ds, top, cache_prefix)
    if not payloads:
        return
    print(f"\n  -- C. Threshold Sensitivity ({top}, mean over "
          f"{len(payloads)} seeds; default-selected epoch -- differs from A's "
          f"best-val-acc epoch) --")
    print(_THRESH_LEGEND)
    print("    " + "operating point".ljust(16) + "thr".center(7)
          + "acc".center(13) + "f1".center(13) + "prec".center(13)
          + "rec".center(13))
    for label, key in THRESH_POINTS:
        thr, _ = _mean_std([_dig(p, f"{key}.threshold") for p in payloads])
        cells = ""
        for m in ("acc", "f1", "prec", "rec"):
            mu, sd = _mean_std([_dig(p, f"{key}.{m}") for p in payloads])
            cells += _cell(mu, sd)
        thr_s = f"{thr:.2f}" if thr is not None else "--"
        print("    " + label.ljust(16) + thr_s.center(7) + cells)


# --------------------------------------------------- D) training cost & reproducibility
# Hardware fallback for caches trained before train.py captured it (2026-07-19).
# New runs carry payload["hardware"] auto-detected; this is only the stated label
# for legacy runs so their wall-time stays hardware-attributable.
STATED_HW = ("NVIDIA GeForce RTX 5060 Laptop GPU (~8 GB VRAM) "
             "[stated; auto-captured on runs from 2026-07-19 on]")


def _hw_line(payloads):
    """One hardware descriptor for the dataset's runs: prefer an auto-captured
    payload["hardware"]; peak VRAM is the max observed across seeds."""
    hws = [p.get("hardware") for p in payloads
           if isinstance(p, dict) and isinstance(p.get("hardware"), dict)]
    if not hws:
        return STATED_HW
    gpu = next((h.get("gpu") for h in hws if h.get("gpu")), None)
    vram = next((h.get("vram_gb") for h in hws if h.get("vram_gb")), None)
    peaks = [h.get("peak_vram_gb") for h in hws
             if isinstance(h.get("peak_vram_gb"), (int, float))]
    cuda = next((h.get("cuda") for h in hws if h.get("cuda")), None)
    torch_v = next((h.get("torch") for h in hws if h.get("torch")), None)
    s = gpu or "GPU"
    if vram is not None:
        s += f" ({vram:.0f} GB VRAM)"
    if peaks:
        s += f" | peak used {max(peaks):.2f} GB"
    if cuda:
        s += f" | CUDA {cuda}"
    if torch_v:
        s += f" | torch {torch_v}"
    return s


def feasibility(ds, present, cache_prefix):
    print(f"\n  -- D. Training Cost & Reproducibility (per rung) --")
    print("    " + "rung".ljust(8) + "seeds".center(7)
          + "train seconds".center(18) + "min sec".center(12)
          + "sec/epoch*".center(16) + "best_epoch".center(16))
    total_sec = 0.0                    # actual GPU-seconds incurred (all seeds)
    ladder_1seed_sec = 0.0             # cost of ONE full-ladder pass (mean/rung)
    all_payloads = []
    for rung in present:
        _, payloads = _rung_payloads(ds, rung, cache_prefix)
        all_payloads.extend(payloads)
        secs = [_dig(p, "seconds") for p in payloads]
        eps = [_dig(p, "best_epoch") for p in payloads]
        sec_mu, sec_sd = _mean_std(secs)
        ep_mu, ep_sd = _mean_std(eps)
        # per-seed sec/epoch ratio, THEN averaged -- dividing the two means
        # instead would hide seed-to-seed variance in training efficiency.
        rate_mu, rate_sd = _mean_std([s / e for s, e in zip(secs, eps)
                                      if s is not None and e])
        secs_ok = [s for s in secs if s is not None]
        total_sec += sum(secs_ok)
        if sec_mu is not None:
            ladder_1seed_sec += sec_mu
        mn = f"{min(secs_ok):.0f}" if secs_ok else "--"
        print("    " + rung.ljust(8) + str(len(payloads)).center(7)
              + _cell(sec_mu, sec_sd, 18) + mn.center(12)
              + _cell(rate_mu, rate_sd, 16) + _cell(ep_mu, ep_sd, 16))
    print("    (*sec/epoch = train seconds / best_epoch, per seed then averaged;"
          " approximate -- `seconds` includes the post-best patience epochs too,"
          " so this slightly overstates true per-epoch cost)")
    # Absolute-feasibility summary: what it costs to reproduce, on named hardware.
    print(f"    hardware: {_hw_line(all_payloads)}")
    print(f"    reproduce cost: {total_sec/3600:.2f} GPU-h total across all cached "
          f"seeds/rungs | {ladder_1seed_sec/3600:.2f} GPU-h for one full-ladder "
          f"single-seed pass\n    (single GPU, single machine -- no cluster/"
          "multi-GPU; bf16 + gradient checkpointing keep it within the card's "
          "VRAM budget)")


# ------------------------------------------------------------ E) dataset audit
def data_quality(ds):
    tp, vp = _base(ds, "train"), _base(ds, "val")
    print(f"\n  -- E. Dataset Audit (audited split, correct_val.py criteria) --")
    if not (os.path.exists(tp) and os.path.exists(vp)):
        print(f"    [skip] base files absent "
              f"({os.path.basename(tp)}/{os.path.basename(vp)})")
        return
    train, val = _load_jsonl(tp), _load_jsonl(vp)
    train_codes = defaultdict(set)
    for r in train:
        train_codes[r["raw_code"]].add(int(r["label"]))
    val_code_labels = defaultdict(set)
    for r in val:
        val_code_labels[r["raw_code"]].add(int(r["label"]))
    reasons = {"leak": 0, "contradicted": 0, "within_val_dup": 0}
    seen = set()
    for r in val:
        c, lab = r["raw_code"], int(r["label"])
        elsewhere = train_codes.get(c, set()) | val_code_labels[c]
        if (1 - lab) in elsewhere:
            reasons["contradicted"] += 1; continue
        if c in train_codes:
            reasons["leak"] += 1; continue
        if c in seen:
            reasons["within_val_dup"] += 1; continue
        seen.add(c)
    n = len(val)
    bad = sum(reasons.values())
    clean = n - bad
    print(f"    val examples (pre-audit):        {n}")
    print(f"    leakage across train/val:         {reasons['leak']}")
    print(f"    conflicting labels removed:       {reasons['contradicted']}")
    print(f"    duplicate validation samples:     {reasons['within_val_dup']}")
    print(f"    clean validation size:            {clean} ({100*clean/max(1,n):.1f}%)")


# ------------------------------------------------------------------------ driver
def run(ds, cache_prefix, pin_rung):
    print("=" * 74)
    print(f"RQ3  |  {ds.upper()}   (FuSEVul stated: {FUSEVUL[ds]})   "
          f"[cache: {cache_prefix}_*]")
    print("=" * 74)
    present = _present_rungs(ds, cache_prefix)
    if not present:
        print(f"  [skip] no {cache_prefix}_* cache runs found for this dataset.")
        print()
        return
    if pin_rung and pin_rung in present:
        top = pin_rung
    elif pin_rung:
        print(f"  [warn] --rung {pin_rung} has no {cache_prefix}_{ds} cache; "
              f"using top present rung {present[-1]}")
        top = present[-1]
    else:
        top = present[-1]              # highest present rung = SemanticVul
    benchmark(ds, present, top, cache_prefix)
    seed_distribution(ds, present, top, cache_prefix)
    threshold_robustness(ds, top, cache_prefix)
    feasibility(ds, present, cache_prefix)
    data_quality(ds)
    print()


def _choose_datasets(args):
    if args.dataset:
        arg = args.dataset.strip().lower()
        if arg in DATASETS:
            return [arg]
        if arg in ("both", "all"):
            return list(DATASETS)
        print(f"unknown dataset '{arg}' (use: reveal | devign | both)")
        sys.exit(2)
    print("RQ3 evidence -- select dataset:")
    print("  1) reveal")
    print("  2) devign")
    print("  3) both")
    try:
        c = input("choice [1/2/3]: ").strip()
    except EOFError:
        print("(no input; defaulting to both)")
        return list(DATASETS)
    return {"1": ["reveal"], "2": ["devign"], "3": list(DATASETS)}.get(
        c, list(DATASETS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=None,
                    help="reveal | devign | both (omit for an interactive menu)")
    ap.add_argument("--rung", choices=list(RUNGS), default=None,
                    help="pin which rung is 'SemanticVul' (default: top present)")
    ap.add_argument("--cache-prefix", default="final",
                    help="cache family -> <prefix>_<ds>_<rung>_cache "
                         "(default 'final'; use a frozen-cached prefix for the "
                         "low-resource story)")
    args = ap.parse_args()
    for ds in _choose_datasets(args):
        run(ds, args.cache_prefix, args.rung)


if __name__ == "__main__":
    main()
