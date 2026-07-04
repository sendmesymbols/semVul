"""Evaluate v2 explanation files: does the channel finally carry class signal?

Reports per file: schema validity, evidence-grounding rate, per-class content
stats, and — the headline — ROC-AUC of label-free explanation signals
(risk_level, n_risky_ops) against the ground-truth label. v1 sits at ~0.51
(chance); the gate for full regeneration is >= ~0.58.

Usage:
  .venv/Scripts/python.exe experiments/expl_v2/pilot_eval.py \
      experiments/expl_v2/out/devign_train__*.jsonl
"""
from __future__ import annotations
import glob
import json
import re
import sys

RISK_MAP = {"none": 0, "low": 1, "medium": 2, "high": 3}
CONF_MAP = {"low": 0, "medium": 1, "high": 2}


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def rank_auc(scores, labels) -> float:
    """Mann-Whitney AUC with tie correction; no sklearn needed."""
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
    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def evaluate(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        print(f"\n=== {path}\n  (empty)")
        return

    n = len(rows)
    valid = 0
    grounded_claims, total_claims = 0, 0
    samples_with_ungrounded = 0
    stats = {0: {"n": 0, "risky": 0, "missing": 0, "safety": 0,
                 "risk_none": 0, "risk_hi": 0},
             1: {"n": 0, "risky": 0, "missing": 0, "safety": 0,
                 "risk_none": 0, "risk_hi": 0}}
    risk_scores, nrisky_scores, combo_scores, labels = [], [], [], []
    gen_secs = []

    for r in rows:
        e = r.get("explanation") or {}
        lab = r.get("label")
        code_n = norm_ws(r.get("raw_code", ""))
        ok = isinstance(e, dict) and e.get("risk_level") in RISK_MAP \
            and isinstance(e.get("risky_operations"), list)
        if ok:
            valid += 1
        if lab not in (0, 1) or not ok:
            continue

        ro = e.get("risky_operations") or []
        mc = e.get("missing_checks") or []
        si = e.get("safety_indicators") or []
        any_ungrounded = False
        for op in ro:
            ev = op.get("evidence", "") if isinstance(op, dict) else str(op)
            total_claims += 1
            if norm_ws(ev) and norm_ws(ev) in code_n:
                grounded_claims += 1
            else:
                any_ungrounded = True
        if any_ungrounded:
            samples_with_ungrounded += 1

        st = stats[lab]
        st["n"] += 1
        st["risky"] += len(ro)
        st["missing"] += len(mc)
        st["safety"] += len(si)
        rl = RISK_MAP[e["risk_level"]]
        st["risk_none"] += 1 if rl == 0 else 0
        st["risk_hi"] += 1 if rl >= 2 else 0

        risk_scores.append(rl)
        nrisky_scores.append(len(ro))
        conf = CONF_MAP.get(e.get("confidence"), 1)
        combo_scores.append(rl * 10 + len(ro) - 0.1 * len(si) + 0.01 * conf)
        labels.append(lab)
        if isinstance(r.get("meta"), dict) and r["meta"].get("gen_seconds"):
            gen_secs.append(float(r["meta"]["gen_seconds"]))

    print(f"\n=== {path}")
    print(f"  rows={n}  schema-valid={valid / n * 100:.1f}%")
    if total_claims:
        print(f"  grounding: {grounded_claims}/{total_claims} claims verbatim-in-code "
              f"({grounded_claims / total_claims * 100:.1f}%), "
              f"{samples_with_ungrounded} samples with >=1 ungrounded claim")
    for lab in (0, 1):
        st = stats[lab]
        if st["n"] == 0:
            continue
        print(f"  label={lab} (n={st['n']}): "
              f"risky={st['risky'] / st['n']:.2f} "
              f"missing={st['missing'] / st['n']:.2f} "
              f"safety={st['safety'] / st['n']:.2f} "
              f"risk_level none={st['risk_none'] / st['n'] * 100:.0f}% "
              f"med/high={st['risk_hi'] / st['n'] * 100:.0f}%")
    if labels:
        print(f"  SIGNAL AUC: risk_level={rank_auc(risk_scores, labels):.3f}  "
              f"n_risky={rank_auc(nrisky_scores, labels):.3f}  "
              f"combo={rank_auc(combo_scores, labels):.3f}   (v1 baseline ~0.51, gate ~0.58)")
    if gen_secs:
        med = sorted(gen_secs)[len(gen_secs) // 2]
        full_h = med * 47000 / 3600
        print(f"  throughput: median {med:.1f}s/sample -> full regen (~47k) ~{full_h:.0f} h")


def main():
    paths = []
    for arg in sys.argv[1:]:
        paths.extend(sorted(glob.glob(arg)))
    if not paths:
        print("usage: pilot_eval.py <jsonl-or-glob> [...]")
        sys.exit(1)
    for p in paths:
        evaluate(p)


if __name__ == "__main__":
    main()
