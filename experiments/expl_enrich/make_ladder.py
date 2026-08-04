"""Gather the per-rung caches for one dataset and present a ladder report.

Reads runs/l1_<ds>_cache, runs/l2_<ds>_cache, runs/l3_<ds>_cache (whatever the
reproduce_<ds>_lN.ps1 wrappers produced; multiple seeds per cache are averaged),
and prints + writes a single ladder report:

  * one row per rung: ROC / PR / acc@0.5 / F1@0.5 / tuned acc,F1 /
    base-paper(by val acc) acc,F1 / best val-ROC epoch (val-oracle, transparency)
  * a pooled ENSEMBLE row (mean of the available rungs' val probs) with the
    joint-threshold beat-both check vs the dataset's stated numbers.

No training and no retraining: purely reads the cached *_probs.npz + JSON that
train.py already wrote. "produce_<ds>.ps1" and "make_ladder_<ds>.ps1" both call
this.

  .venv/Scripts/python.exe experiments/expl_enrich/make_ladder.py --ds devign
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)

RUNS = os.path.join(ROOT, "experiments", "runs")
REPORTS = os.path.join(ROOT, "experiments", "reports")
STATED = {"devign": {"acc": 60.39, "f1": 55.91},
          "reveal": {"acc": 91.68, "f1": 46.76}}
RUNGS = ("L1", "L2", "L3")


def cache_dir(rung, ds, prefix_tmpl):
    return os.path.join(RUNS, prefix_tmpl.format(rung=rung.lower(), ds=ds))


def find_rung(rung, ds, prefix_tmpl):
    """Return (json_paths, npz_paths) for a rung's cache (any seed subdir)."""
    d = cache_dir(rung, ds, prefix_tmpl)
    js = sorted(glob.glob(os.path.join(d, "**", f"fusevul_ladder_{ds}_{rung}.json"),
                          recursive=True))
    npz = sorted(glob.glob(os.path.join(d, "**", f"fusevul_ladder_{ds}_{rung}_probs.npz"),
                           recursive=True))
    return [f for f in js if "smoke" not in f], [f for f in npz if "smoke" not in f]


def joint_sweep(p, y, sa, sf):
    best = None
    for t in np.linspace(0.02, 0.98, 385):
        yh = (p >= t).astype(int)
        a = 100 * accuracy_score(y, yh)
        f = 100 * f1_score(y, yh, zero_division=0)
        m = min(a - sa, f - sf)
        if best is None or m > best[0]:
            best = (m, t, a, f)
    return best  # (worst_margin, thr, acc, f1)


def best_val_roc_epoch(npz):
    """val-oracle: ROC at the epoch with the highest val ROC (transparency only,
    NOT a headline number — selecting the epoch on val is circular)."""
    d = np.load(npz)
    if "val_probs_per_epoch" not in d:
        return None
    y = d["val_y"]
    per = d["val_probs_per_epoch"]
    eps = d.get("ep_index")
    rocs = [100 * roc_auc_score(y, per[i]) for i in range(len(per))]
    j = int(np.argmax(rocs))
    ep = int(eps[j]) if eps is not None and len(eps) == len(per) else j + 1
    return rocs[j], ep


def mean_val_prob(npz_paths):
    """Average val_prob across seeds for one rung (aligned val order)."""
    ps, y = [], None
    for f in npz_paths:
        d = np.load(f)
        if y is None:
            y = d["val_y"]
        elif not np.array_equal(d["val_y"], y):
            print(f"[warn] val mismatch, skipping {f}")
            continue
        ps.append(d["val_prob"])
    if not ps:
        return None, None
    return np.mean(ps, axis=0), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["devign", "reveal"])
    ap.add_argument("--cache-prefix", default="{rung}_{ds}_cache",
                    help="template for the per-rung cache dir name")
    args = ap.parse_args()
    ds, st = args.ds, STATED[args.ds]

    lines = [f"# {ds} ladder report  (stated acc={st['acc']} f1={st['f1']})", ""]
    lines += ["Merged from per-rung caches "
              f"({', '.join(cache_dir(r, ds, args.cache_prefix).replace(RUNS+os.sep,'') for r in RUNGS)}). "
              "Headline = tuned-on-tune (non-circular). base-paper = FuSEVul's "
              "circular rule (best val-acc epoch @0.5). best-ep-ROC = val-oracle "
              "ceiling (transparency, not a claim).", ""]
    lines += ["| rung | seeds | ROC | PR | acc@0.5 | F1@0.5 | tuned acc | tuned F1 "
              "| base acc | base F1 | best-ep ROC |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]

    rung_probs, rung_y, present = {}, None, []
    for rung in RUNGS:
        js, npz = find_rung(rung, ds, args.cache_prefix)
        if not js:
            lines.append(f"| {rung} | - | *(no cache)* | | | | | | | | |")
            continue
        present.append(rung)
        # metrics from the first seed's JSON (per-seed headline); probs averaged
        payloads = [json.load(open(f, encoding="utf-8")) for f in js]
        p0 = payloads[0]
        argmax = p0.get("argmax", {})
        tuned = p0.get("tuned_on_tune", {})
        base = (p0.get("base_paper_protocol", {}).get("by_val_acc", {}).get("argmax", {}))
        roc = np.mean([p.get("val_roc_auc", float("nan")) for p in payloads])
        pr = np.mean([p.get("val_pr_auc", float("nan")) for p in payloads])
        be = best_val_roc_epoch(npz[0]) if npz else None
        be_s = f"{be[0]:.2f}@ep{be[1]}" if be else "-"
        lines.append(
            f"| {rung} | {len(js)} | {roc:.2f} | {pr:.2f} | "
            f"{argmax.get('acc', float('nan')):.2f} | {argmax.get('f1', float('nan')):.2f} | "
            f"{tuned.get('acc', float('nan')):.2f} | {tuned.get('f1', float('nan')):.2f} | "
            f"{base.get('acc', float('nan')):.2f} | {base.get('f1', float('nan')):.2f} | {be_s} |")
        mp, y = mean_val_prob(npz)
        if mp is not None:
            rung_probs[rung] = mp
            if rung_y is None:
                rung_y = y
            elif not np.array_equal(y, rung_y):
                print(f"[warn] {rung} val order differs; excluded from ensemble")
                del rung_probs[rung]
    lines.append("")

    if rung_probs and rung_y is not None:
        ens = np.mean(list(rung_probs.values()), axis=0)
        roc = 100 * roc_auc_score(rung_y, ens)
        pr = 100 * average_precision_score(rung_y, ens)
        a5 = 100 * accuracy_score(rung_y, ens >= .5)
        f5 = 100 * f1_score(rung_y, ens >= .5, zero_division=0)
        m, t, a, f = joint_sweep(ens, rung_y, st["acc"], st["f1"])
        beat = "  **BEATS BOTH**" if m > 0 else ""
        lines += [f"## Pooled ensemble of {'+'.join(sorted(rung_probs))} "
                  f"(n={len(rung_y)}, pos={rung_y.mean()*100:.1f}%)", "",
                  f"- ROC {roc:.2f} | PR {pr:.2f} | acc@0.5 {a5:.2f} | F1@0.5 {f5:.2f}",
                  f"- joint-threshold (val-oracle) thr={t:.3f} -> acc {a:.2f} / "
                  f"F1 {f:.2f} | worst-margin vs stated {m:+.2f}{beat}",
                  f"- stated: acc {st['acc']} / F1 {st['f1']}", ""]
        if m <= 0:
            lines.append(f"> Not beating both yet: {'acc' if (a-st['acc'])<(f-st['f1']) else 'F1'} "
                         f"is the binding side.")
    else:
        lines += ["## Pooled ensemble", "", "*(no rung caches found - run the "
                  f"reproduce_{ds}_lN.ps1 scripts first)*", ""]

    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, f"ladder_{ds}.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[make_ladder] rungs present: {present or 'NONE'}")
    print(f"[make_ladder] wrote {out}")


if __name__ == "__main__":
    main()
