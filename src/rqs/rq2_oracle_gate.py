"""Oracle confidence-gate ablation for RQ2 -- DIAGNOSTIC ONLY, not an RQ2 result.

Question: if we had an ORACLE that perfectly knew, per validation sample,
whether to trust the code embedding or the explanation embedding -- using
ReVeal's raw explanation.confidence field as the switch -- would that beat
L2 (static_concat) / L3 (gated)?

CAVEAT -- read before trusting any number this prints: ReVeal's ACTIVE
explanation.confidence is NOT organic. It is a label-conditioned synthetic
value produced by fill_data.py (confidence->label AUC ~0.87 on val; mean
confidence ~69 for label=1 vs ~38 for label=0). So "oracle beats L2" here is
evidence about how exploitable that synthetic leak is, NOT evidence that
quality-aware gating works. Never report these numbers as an architecture
contribution without this caveat attached. See memory
reveal-confidence-synthetic-injection for the full provenance trace.

Three variants, all off the SAME frozen cache rq2.py already builds:
  A  code_only   (existing rq2.py mode, unchanged)
  B  text_only   (existing rq2.py mode, unchanged)
  C  oracle(T)   hard switch: p = p_code if conf < T else p_text,
                 for T in {20,30,40,50,60} (conf on ReVeal's native 0..100 scale)

    python -m src.rqs.rq2_oracle_gate --dataset reveal --seeds 1,3,7
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.rqs import rq2

data_mod = rq2.data_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["devign", "reveal"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--seeds", default="1,3,7")
    ap.add_argument("--thresholds", default="20,30,40,50,60")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--pool", choices=list(rq2.POOLS), default="mean")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    path = rq2.build_cache(args.dataset, subset=args.subset)
    C = dict(np.load(path))
    yv = C["y_va"]

    # conf is NOT part of the cached npz (rq2's gate deliberately never sees raw
    # confidence -- see src/quality_features.py). Pull it fresh from the same
    # loader build_cache used; val is never shuffled/subsetted differently, so
    # order must match the cache exactly -- verified by the assertion below.
    _, va = data_mod.load(args.dataset, subset=args.subset)
    assert np.array_equal(va["y"], yv), (
        "val split order mismatch between fresh data_mod.load() and the cached "
        "rq2 npz -- cannot safely align confidence to cached probs")
    conf_va = va["conf"]

    print(f"\n=== oracle confidence-gate ablation | {args.dataset} | seeds {seeds} ===")
    print("CAVEAT: explanation.confidence is treated here as a POSSIBLY label-leaky "
          "signal (confirmed synthetic/label-conditioned for reveal). Any 'oracle beats "
          "L2' result below is an upper bound on exploitable leakage, not a gating result.\n")

    code_runs = [rq2.train_variant(C, "code_only", args.dataset, seed=s, epochs=args.epochs,
                                    pool=args.pool, return_probs=True) for s in seeds]
    text_runs = [rq2.train_variant(C, "text_only", args.dataset, seed=s, epochs=args.epochs,
                                    pool=args.pool, return_probs=True) for s in seeds]

    def agg(metrics_list):
        out = {}
        for k in ("acc", "f1", "roc", "pr"):
            vals = [m[k] for m in metrics_list]
            out[k] = round(float(np.mean(vals)), 2)
            out[k + "_std"] = round(float(np.std(vals)), 2)
        return out

    def ms(m, k):
        return f"{m[k]:.2f}+/-{m.get(k + '_std', 0.0):.2f}"

    hdr = f"{'variant':22s}" + "".join(f"{h:>14s}" for h in ("acc", "F1", "ROC", "PR"))
    print(hdr)
    print("-" * len(hdr))

    results = {"_meta": {
        "dataset": args.dataset, "seeds": seeds, "thresholds": thresholds,
        "cache": os.path.basename(path),
        "caveat": "explanation.confidence is a label-conditioned synthetic value for "
                  "reveal (fill_data.py), not organic; oracle-gate numbers upper-bound "
                  "exploitable leakage, they are NOT an architecture result",
    }}

    results["A_code_only"] = agg([{k: r[k] for k in ("acc", "f1", "roc", "pr")} for r in code_runs])
    results["B_text_only"] = agg([{k: r[k] for k in ("acc", "f1", "roc", "pr")} for r in text_runs])
    for key in ("A_code_only", "B_text_only"):
        print(f"{key:22s}" + "".join(f"{ms(results[key], k):>14s}" for k in ("acc", "f1", "roc", "pr")))

    for T in thresholds:
        per_seed = []
        for cr, tr in zip(code_runs, text_runs):
            p_oracle = np.where(conf_va < T, cr["_val_p"], tr["_val_p"])
            per_seed.append(rq2._metrics(yv, p_oracle))
        key = f"C_oracle_T{int(T)}"
        results[key] = agg(per_seed)
        print(f"{key:22s}" + "".join(f"{ms(results[key], k):>14s}" for k in ("acc", "f1", "roc", "pr")))

    print("-" * len(hdr))
    print(f"conf(val): min={conf_va.min():.0f} max={conf_va.max():.0f} "
          f"mean|y=1={conf_va[yv == 1].mean():.1f} mean|y=0={conf_va[yv == 0].mean():.1f}")

    out_path = os.path.join(rq2.CACHE_DIR, f"{args.dataset}_rq2_oracle_gate.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n[files] results = {out_path}")
    print(f"(compare against static_concat/L2 and gated/L3 in {args.dataset}_rq2_results*.json "
          f"-- read the CAVEAT above before drawing conclusions)")


if __name__ == "__main__":
    main()
