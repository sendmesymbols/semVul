"""Dual-track evaluation: every saved member + the pooled ensemble, scored on
(1) BENCHMARK val (comparable to FuSEVul's stated numbers) and
(2) CORRECTED val (leak/contradiction/dup rows removed via the saved mask —
    honest-evaluation track, NOT comparable to stated numbers).

No retraining: saved val_prob vectors are aligned to the original val order,
so corrected-val metrics are just a boolean mask away.

  .venv/Scripts/python.exe experiments/expl_enrich/dual_eval.py
"""
from __future__ import annotations
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

RUNS = os.path.join(ROOT, "experiments", "runs")
REPORTS = os.path.join(ROOT, "experiments", "reports")
STATED = {"devign": {"acc": 60.39, "f1": 55.91},
          "reveal": {"acc": 91.68, "f1": 46.76}}


def member_files(ds):
    pats = [os.path.join(RUNS, f"fusevul_ladder_{ds}_L*_probs.npz"),
            os.path.join(RUNS, "seeds", "s*", f"fusevul_ladder_{ds}_L*_probs.npz"),
            os.path.join(RUNS, "laptop", f"fusevul_ladder_{ds}_L*_probs.npz"),
            os.path.join(RUNS, "enriched*", "**",
                         f"fusevul_ladder_{ds}_L*_probs.npz")]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(f for f in files if "smoke" not in f and "BACKUP" not in f))


def joint_sweep(p, y, sa, sf):
    best = None
    for t in np.linspace(0.02, 0.98, 385):
        yh = (p >= t).astype(int)
        a = 100 * accuracy_score(y, yh)
        f = 100 * f1_score(y, yh, zero_division=0)
        m = min(a - sa, f - sf)
        if best is None or m > best[0]:
            best = (m, t, a, f,
                    100 * precision_score(y, yh, zero_division=0),
                    100 * recall_score(y, yh, zero_division=0))
    return best


def block(ds, y, probs, names, mask, tag, lines):
    st = STATED[ds]
    ym, Pm = y[mask], [p[mask] for p in probs]
    ens = np.mean(Pm, axis=0)
    lines.append(f"### {tag}  (n={mask.sum()}, pos={ym.mean()*100:.1f}%)")
    lines.append("")
    lines.append("| model | ROC | PR | acc@0.5 | F1@0.5 | joint thr | acc | F1 | worst-margin |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for nm, p in list(zip(names, Pm)) + [("ENSEMBLE (mean)", ens)]:
        roc = 100 * roc_auc_score(ym, p)
        pr = 100 * average_precision_score(ym, p)
        a5 = 100 * accuracy_score(ym, p >= .5)
        f5 = 100 * f1_score(ym, p >= .5, zero_division=0)
        m, t, a, f, prec, rec = joint_sweep(p, ym, st["acc"], st["f1"])
        beat = " **BEAT BOTH**" if m > 0 else ""
        lines.append(f"| {nm} | {roc:.2f} | {pr:.2f} | {a5:.2f} | {f5:.2f} | "
                     f"{t:.3f} | {a:.2f} | {f:.2f} | {m:+.2f}{beat} |")
    lines.append("")


def main():
    lines = ["# Dual-track evaluation — benchmark val vs corrected val", "",
             "Corrected val = benchmark val minus leak/contradicted/dup rows "
             "(experiments/expl_enrich/correct_val.py). Stated-number claims are "
             "only valid on the BENCHMARK track; the corrected track is the "
             "honest-evaluation secondary. Joint columns = best threshold by "
             "worst-side margin vs stated (val-oracle sweep; use tune-picked "
             "thresholds for headline claims).", ""]
    for ds in ("devign", "reveal"):
        mask_f = os.path.join(RUNS, f"val_clean_mask_{ds}.npz")
        keep = np.load(mask_f)["keep"]
        files = member_files(ds)
        names, probs, y = [], [], None
        for f in files:
            d = np.load(f)
            if y is None:
                y = d["val_y"]
            elif not np.array_equal(d["val_y"], y):
                print(f"[skip] val mismatch: {f}")
                continue
            rel = os.path.relpath(f, RUNS).replace("fusevul_ladder_", "").replace("_probs.npz", "")
            names.append(rel)
            probs.append(d["val_prob"])
        assert len(keep) == len(y)
        st = STATED[ds]
        lines.append(f"## {ds}  (stated acc={st['acc']} f1={st['f1']})")
        lines.append("")
        block(ds, y, probs, names, np.ones(len(y), bool), "BENCHMARK val", lines)
        block(ds, y, probs, names, keep, "CORRECTED val", lines)
    out = os.path.join(REPORTS, "dual_track_eval.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
