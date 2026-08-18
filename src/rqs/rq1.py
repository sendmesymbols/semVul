"""RQ1 evidence: do locally-generated, verdict-scrubbed, evidence-grounded
explanations improve explanation faithfulness AND downstream detection, vs
FuSEVul-style free-form LLM explanations?

Menu-driven (reveal | devign | both). Prints two bundles per dataset:

  A) EXPLANATION QUALITY (label-free, faithfulness side of RQ1)
     - Verdict-in-explanation leakage: does the model-fed explanation text
       assert a vuln/safe verdict, and does that assertion correlate with the
       true label (leakage strength, phi/AUC). Reported post-scrub (the 7-col
       channel actually fed) AND on the pre-scrub raw fields (risk_level /
       llm_v1) so the scrub's effect is visible.
     - Data leakage: val rows whose exact code appears in train / with the
       opposite label / duplicated within val -- reusing correct_val.py's
       exact-code criteria (split-integrity; really RQ3, reported here on
       request, separately from verdict leakage).
     - Evidence grounding: |evidence_tokens & code_tokens| / |evidence_tokens|,
       tokenized identically to quality_features.evidence_overlap_code.
     - Quality analysis: confidence / risk_level / evidence / missing-check
       distributions, and whether confidence|risk_level leak the label (AUC).

  B) DOWNSTREAM DETECTION LIFT (the "improve detection" side of RQ1)
     - L1 (code-only) vs L2 (code + explanation) mean +/- std over the seeds
       present in experiments/runs/final_<ds>_l{1,2}_cache/. L2 - L1 = the
       explanation channel's contribution.

Quality/grounding/verdict metrics read the MODEL-FED set
(explanations/SemanticVul/ACTIVE/<ds>/{train,val}.jsonl); data leakage reads the
base benchmark set (explanations/SemanticVul/<ds>/<ds>_{train,val}.jsonl), matching
correct_val.py. Read-only; no GPU; safe to run against a partial ladder.

Usage:
    python src/rqs/rq1.py            # interactive menu
    python src/rqs/rq1.py reveal
    python src/rqs/rq1.py devign
    python src/rqs/rq1.py both
"""
from __future__ import annotations

import os
import re
import sys
import glob
import json
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.config import EXPL_DIR, RUNS_DIR
from src.data_io import _to_str, _to_list

DATASETS = ("reveal", "devign")

# The clean Qwen-only channel fed by final_*_l{2,3}; risk_level stays outside
# the detector channel because it is the generator's pseudo-verdict.
FED_FIELDS = ("confidence", "purpose", "data_flow", "risky_operations",
              "missing_checks", "evidence_tokens", "safety_indicators",
              "risk_summary")

# Pre-scrub fields kept OUT of the channel -- audited to show what the scrub removed.
RAW_ONLY_FIELDS = ("risk_level",)

# Explicit vuln/safe verdict assertions. Presence in the FED text = a surviving
# verdict; correlation with the label = whether it actually leaks.
_VERDICT_RE = re.compile(
    r"not\s+vulnerab|no\s+vulnerabilit|vulnerab|exploitab|"
    r"\bis\s+safe\b|appears?\s+safe|\bunsafe\b|\binsecure\b|cwe[-_ ]?\d",
    re.IGNORECASE)

_RISK_ORDINAL = {"none": 0, "low": 1, "medium": 2, "med": 2, "moderate": 2,
                 "high": 3, "critical": 4}

# Detection metrics pulled from the per-seed cache JSONs (subset of aggregate_seeds).
DET_METRICS = [
    ("ROC-AUC",        "val_roc_auc"),
    ("PR-AUC",         "val_pr_auc"),
    ("tuned f1",       "tuned_on_tune.f1"),
    ("tuned prec",     "tuned_on_tune.prec"),
    ("tuned rec",      "tuned_on_tune.rec"),
    ("basepaper acc",  "base_paper_protocol.by_val_acc.argmax.acc"),
    ("basepaper f1",   "base_paper_protocol.by_val_acc.argmax.f1"),
]


# --------------------------------------------------------------------------- IO
def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _active(ds, split):
    return os.path.join(EXPL_DIR, "ACTIVE", ds, f"{split}.jsonl")


def _base(ds, split):
    return os.path.join(EXPL_DIR, ds, f"{ds}_{split}.jsonl")


# ---------------------------------------------------------------- small helpers
def _tok(text):
    # identical tokenizer to quality_features._tok -> grounding matches the model.
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", (text or "").lower()))


def _field_str(expl, name):
    v = expl.get(name)
    if isinstance(v, list):
        return " ".join(_to_str(x) for x in _to_list(v))
    if isinstance(v, dict):
        return " ".join(_to_str(x) for x in v.values())
    return _to_str(v)


def _fed_text(expl, fields):
    return " ".join(_field_str(expl, f) for f in fields)


def _auc(scores, labels):
    """Rank-based ROC-AUC of a score vs a binary label (no sklearn). Returns the
    prob a random positive outranks a random negative; 0.5 = no signal."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _phi(x_bool, y):
    """Phi coefficient between two binary vectors (leakage strength)."""
    x = np.asarray(x_bool, dtype=int); y = np.asarray(y, dtype=int)
    n = len(x)
    if n == 0:
        return float("nan")
    n11 = int(((x == 1) & (y == 1)).sum()); n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum()); n00 = int(((x == 0) & (y == 0)).sum())
    denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if denom == 0:
        return 0.0
    return (n11 * n00 - n10 * n01) / (denom ** 0.5)


def _mean_std(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return None, None
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else float("nan"))


def _fmt(mu, sd):
    if mu is None:
        return "   --   "
    if sd != sd:
        return f"{mu:6.2f}   "
    return f"{mu:6.2f}+-{sd:4.2f}"


# ------------------------------------------------------------ A) quality bundle
def explanation_quality(ds):
    path = _active(ds, "val")
    if not os.path.exists(path):
        print(f"  [skip] model-fed val set absent: {path}")
        return
    rows = _load_jsonl(path)
    y = np.array([int(r.get("label", 0)) for r in rows])
    n = len(rows)
    pos = int(y.sum())
    print(f"\n  -- Explanation quality  (ACTIVE/{ds}/val.jsonl : {n} rows, "
          f"{pos} vuln = {100*pos/max(1,n):.1f}%) --")

    expls = [r.get("explanation") or {} for r in rows]

    # Verdict-in-explanation leakage, post-scrub (fed 7-col) vs pre-scrub (raw).
    fed_hit = np.array([1 if _VERDICT_RE.search(_fed_text(e, FED_FIELDS)) else 0
                        for e in expls])
    raw_hit = np.array([1 if _VERDICT_RE.search(
        _fed_text(e, FED_FIELDS) + " " + _fed_text(e, RAW_ONLY_FIELDS)) else 0
        for e in expls])
    print(f"    verdict leakage  fed({len(FED_FIELDS)}-col post-scrub): "
          f"{100*fed_hit.mean():5.1f}% of rows carry a vuln/safe assertion "
          f"| strength phi={_phi(fed_hit, y):+.3f} (0=no label leak)")
    print(f"    verdict leakage  raw(+risk_level,llm_v1): "
          f"{100*raw_hit.mean():5.1f}%   -> scrub removed "
          f"{100*(raw_hit.mean()-fed_hit.mean()):+5.1f} pts")

    # risk_level (scrubbed field) as a verdict classifier -> justifies removing it.
    rl = [str(e.get("risk_level", "")).strip().lower() for e in expls]
    rl_ord = np.array([_RISK_ORDINAL.get(x, np.nan) for x in rl], dtype=float)
    ok = ~np.isnan(rl_ord)
    if ok.any():
        auc_rl = _auc(rl_ord[ok], y[ok])
        dist = ", ".join(f"{k}:{v}" for k, v in Counter(rl).most_common())
        print(f"    risk_level (scrubbed) -> label AUC={auc_rl:.3f}  "
              f"[{dist}]  (high AUC = why it is scrubbed)")

    # confidence: is the model-fed confidence field label-leaking?
    conf = np.array([_num(e.get("confidence")) for e in expls], dtype=float)
    okc = ~np.isnan(conf)
    if okc.any():
        auc_c = _auc(conf[okc], y[okc])
        print(f"    confidence -> label AUC={auc_c:.3f}  "
              f"(mean vuln={np.nanmean(conf[y == 1]):.1f} / "
              f"non={np.nanmean(conf[y == 0]):.1f})  0.5=label-free")

    # Evidence grounding (faithfulness core).
    ov, n_evid, empty = [], [], 0
    for r, e in zip(rows, expls):
        ev = _to_list(e.get("evidence_tokens"))
        if not ev:
            empty += 1
            n_evid.append(0)
            continue
        et = _tok(" ".join(_to_str(x) for x in ev))
        ct = _tok(r.get("raw_code", ""))
        ov.append(len(et & ct) / max(1, len(et)))
        n_evid.append(len(ev))
    ov = np.array(ov) if ov else np.array([0.0])
    print(f"    grounding: evidence-code overlap mean={ov.mean():.3f} "
          f"median={np.median(ov):.3f} | grounded(>0)={100*(ov > 0).mean():.1f}% "
          f"| fully(=1)={100*(ov == 1).mean():.1f}% | empty-evidence="
          f"{100*empty/max(1,n):.1f}% | mean #evidence={np.mean(n_evid):.1f}")

    # Content presence (are explanations substantive, label-free).
    n_miss = np.array([len(_to_list(e.get("missing_checks"))) for e in expls])
    n_risky = np.array([len(_to_list(e.get("risky_operations"))) for e in expls])
    print(f"    content: has missing_checks={100*(n_miss > 0).mean():.1f}% "
          f"| has risky_operations={100*(n_risky > 0).mean():.1f}% "
          f"| mean #missing={n_miss.mean():.2f} #risky={n_risky.mean():.2f}")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ------------------------------------------------------------ data leakage (RQ3)
def data_leakage(ds):
    tp, vp = _base(ds, "train"), _base(ds, "val")
    if not (os.path.exists(tp) and os.path.exists(vp)):
        print(f"\n  -- Data leakage (val-in-train) -- [skip] base files absent "
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
    print(f"\n  -- Data leakage (val-in-train, correct_val.py criteria) --")
    print(f"    val={n} | leak={reasons['leak']} "
          f"contradicted={reasons['contradicted']} "
          f"within_val_dup={reasons['within_val_dup']} "
          f"| dropped {bad} = {100*bad/max(1,n):.1f}%  (clean val={n-bad})")


# ------------------------------------------------------ B) detection lift bundle
def _dig(payload, dotted):
    cur = payload
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


# Accept both the current "semanticvul_" tag and the historical
# "fusevul_ladder_" tag (pre-rename runs on disk), mirroring aggregate_seeds.py.
_CACHE_TAGS = ("semanticvul", "fusevul_ladder")


def _rung_seed_vals(ds, rung, metric_path):
    cache = os.path.join(RUNS_DIR, f"final_{ds}_{rung.lower()}_cache")
    files = set()
    for tag in _CACHE_TAGS:
        files.update(glob.glob(os.path.join(cache, "s*", f"{tag}_{ds}_{rung}.json")))
    vals = []
    for f in sorted(files):
        try:
            with open(f, encoding="utf-8") as fh:
                v = _dig(json.load(fh), metric_path)
            if v is not None:
                vals.append(v)
        except (OSError, json.JSONDecodeError):
            pass
    return vals


def detection_lift(ds):
    def nseeds(rung):
        return len(_rung_seed_vals(ds, rung, "val_roc_auc"))
    n1, n2 = nseeds("L1"), nseeds("L2")
    print(f"\n  -- Downstream detection lift  (L1={n1} seeds, L2={n2} seeds "
          f"@ final_{ds}_l1_cache / _l2_cache) --")
    if n1 == 0 and n2 == 0:
        print("    [skip] no cache runs found.")
        return
    print("    " + "metric".ljust(15) + "L1".center(14) + "L2".center(14)
          + "L2-L1".center(10))
    print("    " + "-" * 53)
    for label, path in DET_METRICS:
        m1, s1 = _mean_std(_rung_seed_vals(ds, "L1", path))
        m2, s2 = _mean_std(_rung_seed_vals(ds, "L2", path))
        d = f"{m2-m1:+6.2f}" if (m1 is not None and m2 is not None) else "  --  "
        print("    " + label.ljust(15) + _fmt(m1, s1).center(14)
              + _fmt(m2, s2).center(14) + d.center(10))


# ------------------------------------------------------------------------ driver
def run(ds):
    print("=" * 72)
    print(f"RQ1  |  {ds.upper()}")
    print("=" * 72)
    explanation_quality(ds)
    data_leakage(ds)
    detection_lift(ds)
    print()


def _choose():
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip().lower()
        if arg in DATASETS:
            return [arg]
        if arg in ("both", "all"):
            return list(DATASETS)
        print(f"unknown dataset '{arg}' (use: reveal | devign | both)")
        sys.exit(2)
    print("RQ1 evidence -- select dataset:")
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
    for ds in _choose():
        run(ds)


if __name__ == "__main__":
    main()
