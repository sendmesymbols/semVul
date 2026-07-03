"""Run the full component ladder, Reveal first (fail-fast on the hard dataset).

Resumable: skips any (dataset, rung) whose JSON already exists. One rung failing
(e.g. OOM) is logged and does not abort the rest. Emits a comparison table vs
FuSEVul's stated numbers with threshold-free AUCs, argmax, and tuned operating points.

  # quick validation (downloads + tiny 1-epoch sanity):
  python experiments/fusevul_ladder/run_ladder.py --smoke
  # full unattended run (hours):
  python experiments/fusevul_ladder/run_ladder.py
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train import train_rung, STATED
RUNS = os.path.join(ROOT, "experiments", "runs")
REPORTS = os.path.join(ROOT, "experiments", "reports")


def _mark(v, target):
    if target is None:
        return "n/a"
    d = v - target
    return f"WIN +{d:.2f}" if d > 0.05 else (f"LOSE {d:.2f}" if d < -0.05 else "TIE")


def write_table(datasets):
    lines = ["# FuSEVul component ladder — results vs stated targets", "",
             "L1 = GraphCodeBERT code only · L2 = +RoBERTa explanation (self-attn fusion) · "
             "L3 = +22 quality features. Val split. Threshold picked on a held-out tune slice "
             "(non-circular); argmax@0.5 shown for direct comparability. ROC/PR-AUC are "
             "threshold-free (the fair ladder-contribution measure).", ""]
    for ds in datasets:
        st = STATED[ds]
        lines += [f"## {ds}  (stated: acc {st['acc']}, f1 {st['f1']})", "",
                  "| Rung | ROC-AUC | PR-AUC | Acc@0.5 | F1@0.5 | Acc(tuned) | F1(tuned) | Acc? | F1? |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for rung in ("L1", "L2", "L3"):
            p = os.path.join(RUNS, f"fusevul_ladder_{ds}_{rung}.json")
            if not os.path.exists(p):
                lines.append(f"| {rung} | — | — | — | — | — | — | pending | pending |")
                continue
            d = json.load(open(p))
            a, t = d["argmax"], d["tuned_on_tune"]
            lines.append(
                f"| {rung} | {d['val_roc_auc']:.2f} | {d['val_pr_auc']:.2f} | "
                f"{a['acc']:.2f} | {a['f1']:.2f} | {t['acc']:.2f} | {t['f1']:.2f} | "
                f"{_mark(t['acc'], st['acc'])} | {_mark(t['f1'], st['f1'])} |")
        lines.append("")
    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, "fusevul_ladder.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="downloads + tiny 1-epoch sanity run")
    ap.add_argument("--datasets", nargs="*", default=["reveal", "devign"])  # reveal first
    ap.add_argument("--rungs", nargs="*", default=["L1", "L2", "L3"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--fusion", default="self")
    args = ap.parse_args()

    kw = dict(fusion=args.fusion, batch=args.batch)
    if args.smoke:
        kw.update(subset=200, epochs=1)
        out_dir = os.path.join(RUNS, "smoke")
    else:
        kw.update(epochs=args.epochs)
        out_dir = RUNS

    for ds in args.datasets:
        for rung in args.rungs:
            tag = f"{ds}_{rung}" + ("_smoke" if args.smoke else "")
            if os.path.exists(os.path.join(out_dir, f"fusevul_ladder_{tag}.json")):
                print(f"[skip] {tag} already done", flush=True)
                continue
            print(f"\n===== {tag} =====", flush=True)
            try:
                train_rung(ds, rung, out_dir=out_dir, **kw)
            except Exception:
                print(f"[FAIL] {tag}\n{traceback.format_exc()}", flush=True)
    if not args.smoke:
        write_table(args.datasets)


if __name__ == "__main__":
    main()
