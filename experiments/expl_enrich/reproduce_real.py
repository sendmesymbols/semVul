"""Overnight driver for the REAL-enrichment treatment arm (both datasets).

Configs (gates in explanations/SemanticVul/devign_real/ENRICHMENT_RESULTS.md):
  devign  L1/L2/L3  512-token window, train = enriched.clean.real (NO aug),
          val = enriched.real (full benchmark val, 2732 rows, 67% treated),
          text channel = deanon'd fields + evidence_tokens + lexical_digest
          (TF-IDF gate: +7.45 ROC over anon code)     -> runs/enriched512_real/
  reveal  L1/L2/L3  320-token window (matches runs/enriched arm), train =
          enriched.clean.real, val = enriched.real (2273 rows),
          text channel = CORE fields (no prose, no llm_v1) + lexical_digest
          (TF-IDF gate: +1.39 ROC, CI [+0.31,+2.63])  -> runs/enriched_real/

Output dirs match the runs/enriched* glob so ensemble.py / dual_eval.py
auto-scan the new members alongside every earlier arm (val row-aligned:
*.real val files are order-identical to the untreated ones).

Resumable: any (dataset, rung, seed) with a final JSON is skipped.
Invoked by reproduce_reveal.ps1 / reproduce_devign.ps1.
"""
from __future__ import annotations
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LADDER = os.path.join(ROOT, "experiments", "fusevul_ladder")
for _p in (ROOT, LADDER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUNS = os.path.join(ROOT, "experiments", "runs")
SPLIT_SEED = 1337  # shared tune carve -> tune probs averageable across seeds

# --fields prefix (default since 2026-07-15): the text channel is the single
# materialized `explanation.prefix` string carried by the round-3 ACTIVE files
# (built in D:\SementicVul, proxy-gate F1 reveal 0.5055 / devign 0.6324).
# Recipes were selected by a 14-way per-column ablation:
#   reveal: lexical_digest | log2-binned metrics | calls | risky-first
#           evidence | string_literals | tail  (denoised train baked in)
#   devign: digest | missing | calls | identifier subword morphemes |
#           REAL-code head (devign_real join) | real string literals
#           -- risky_apis measured NEGATIVE and is excluded; purpose/data_flow/
#           evidence_tokens measured negative on devign columns as well.
# Any other channel: pass an explicit comma list via --fields (verbatim to
# SEMVUL_EXPL_FIELDS, same list for both datasets); the final_*.ps1 launchers
# hardcode their 8-column $Cols this way.

os.environ["SEMVUL_EXPL_VARIANT"] = "enriched"
os.environ["SEMVUL_VAL_VARIANT"] = "enriched.real"
# Default to the 44-dim v2 quality block for the fine-tuned ladder, but let a
# caller opt out (SEMVUL_QUAL_V2=0) -- the RQ2 gate scripts do this to feed the
# gate ONLY the clean v1 label-free features (v2 adds risk_level/confidence
# ordinals = the LLM's own leaky vuln score, unwanted for a label-free gate).
os.environ.setdefault("SEMVUL_QUAL_V2", "1")
# ACTIVE/ is the canonical run input (single source of truth). When present it is
# read directly; when absent the loaders fall back to the long-named .real files.
# apply_real_enrichment keeps ACTIVE in sync, so copying only ACTIVE/ is enough.
os.environ["SEMVUL_ACTIVE_DIR"] = "1"


def _check_prefix_present():
    """Fail fast if ACTIVE files lack the materialized explanation.prefix
    (e.g. ACTIVE was rebuilt by apply_real_enrichment from pre-round-3
    sources). Guards against a silently EMPTY text channel."""
    import json
    from src.config import EXPL_DIR
    for ds in ("reveal", "devign"):
        for split in ("train", "val"):
            p = EXPL_DIR / "ACTIVE" / ds / f"{split}.jsonl"
            if not p.exists():
                continue
            with p.open(encoding="utf-8") as fh:
                row = json.loads(fh.readline())
            if not (row.get("explanation") or {}).get("prefix"):
                sys.exit(f"[fields=prefix] {p} has no explanation.prefix - "
                         f"restore the *_final_*_3 files into ACTIVE/ (they were "
                         f"overwritten?), or pass an explicit --fields column list")


def _build_jobs(args):
    """Construct the (ds, rungs, sub, suffix, fields, kw) job tuples. ReVeal-only
    knobs (tail_digest, focal alpha/gamma) are attached to the ReVeal job ONLY;
    the Devign job is never given them."""
    ga = max(1, 32 // args.batch512)
    if args.fields == "prefix":
        _check_prefix_present()
        reveal_fields = devign_fields = "prefix"
    else:  # literal comma-separated column list -> SEMVUL_EXPL_FIELDS verbatim
        # (e.g. "risk_level,confidence,called_functions"); serialized by
        # src/data_io.py:_render_expl_field. Same list for both datasets.
        reveal_fields = devign_fields = args.fields
        if args.tail_digest:  # ReVeal-only opt-in: append tail_digest to its channel
            reveal_fields = reveal_fields + ",tail_digest"
    reveal_kw = {}
    if args.focal_alpha is not None:
        reveal_kw["focal_alpha"] = args.focal_alpha
        reveal_kw["focal_gamma"] = args.focal_gamma
    if args.max_text is not None:      # explanation (RoBERTa) window; 512 = FuSEVul's budget
        reveal_kw["max_text"] = args.max_text
    if args.epochs is not None:
        reveal_kw["epochs"] = args.epochs
    devign_kw = dict(max_code=512, batch=args.batch512, grad_accum=ga)
    if args.epochs is not None:
        devign_kw["epochs"] = args.epochs
    if args.max_text is not None:      # devign text (RoBERTa) window; default 256
        devign_kw["max_text"] = args.max_text
    # Code encoder (graphcodebert default | codet5p = FuSEVul's encoder). Applies
    # to both datasets so the whole ladder shares one code channel.
    reveal_kw["code_enc"] = args.code_enc
    devign_kw["code_enc"] = args.code_enc
    # --max-code: unify the code window across datasets (default keeps the
    # historical reveal 320 / devign 512 asymmetry). When set (e.g. 512 to match
    # FuSEVul + devign), reveal also inherits the memory-safe batch512/grad_accum
    # so the wider window still fits 8GB -- effective batch stays 32 either way.
    if args.max_code is not None:
        reveal_kw["max_code"] = args.max_code
        devign_kw["max_code"] = args.max_code
        reveal_kw["batch"] = args.batch512
        reveal_kw["grad_accum"] = ga
    if args.subset is not None:      # fast directional / mechanism-plot runs
        reveal_kw["subset"] = args.subset
        devign_kw["subset"] = args.subset
    jobs = [
        ("reveal", args.rungs or ["L2", "L3", "L1"],
         "enriched_real" + args.out_tag, "clean.real", reveal_fields, reveal_kw),
        ("devign", args.rungs or ["L1", "L2", "L3"],
         "enriched512_real" + args.out_tag, "clean.real", devign_fields,
         devign_kw),
    ]
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]
    if args.cache_name:
        # explicit per-rung cache dir (e.g. l1_devign_cache). Overrides the
        # enriched*_real[out_tag] subdir so the L1/L2/L3 ps1 wrappers each land
        # in their own named folder that make_ladder.py gathers.
        jobs = [(ds, rungs, args.cache_name, suffix, fields, kw)
                for (ds, rungs, _sub, suffix, fields, kw) in jobs]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["devign", "reveal"], default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1337, 2024])
    ap.add_argument("--batch512", type=int, default=2,
                    help="batch for 512-token devign (2 fits 8GB; 4 on >=16GB)")
    ap.add_argument("--rungs", nargs="*", default=None)
    ap.add_argument("--tail-digest", action="store_true",
                    help="ReVeal only: append tail_digest to the text channel")
    ap.add_argument("--focal-alpha", type=float, default=None,
                    help="ReVeal only: focal positive weight (default auto ~0.80)")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="ReVeal only: focal gamma")
    ap.add_argument("--out-tag", type=str, default="",
                    help="suffix on the run subdir, e.g. _tail_a85 (keeps A/B arms separate)")
    ap.add_argument("--cache-name", type=str, default="",
                    help="explicit run subdir, e.g. l1_devign_cache (overrides "
                         "the enriched*_real dir; used by the per-rung ps1 wrappers)")
    ap.add_argument("--max-text", type=int, default=None,
                    help="explanation (RoBERTa) token window; default 256, 512 = FuSEVul budget")
    ap.add_argument("--max-code", type=int, default=None,
                    help="code (CodeT5+) token window for BOTH datasets; default "
                         "reveal 320 (train_rung default) / devign 512. Set 512 to "
                         "match FuSEVul + devign (reveal then also adopts --batch512 "
                         "so the wider window fits VRAM).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="training epochs per rung (default: train_rung's 12)")
    ap.add_argument("--fields", default="prefix",
                    help="text channel: 'prefix' = materialized round-3 "
                         "explanation.prefix (default); or a literal "
                         "comma-separated explanation.* column list "
                         "(e.g. 'risk_level,confidence,called_functions') "
                         "passed verbatim to SEMVUL_EXPL_FIELDS")
    ap.add_argument("--evidence-window", action="store_true",
                    help="L2/L3 code channel: center the code span on the "
                         "explanation's verbatim evidence instead of the head "
                         "(SEMVUL_CODE_WINDOW=evidence; attacks the truncated-"
                         "positives problem; label-blind; off by default)")
    ap.add_argument("--code-enc", default="graphcodebert",
                    help="code encoder for BOTH datasets: 'graphcodebert' "
                         "(default) | 'codet5p' (Salesforce/codet5p-110m-embedding, "
                         "FuSEVul's encoder; loaded via is_decoder shim) | a raw HF id")
    ap.add_argument("--qual-gate", action="store_true",
                    help="L3 quality-gated fusion (SEMVUL_QUAL_GATE=1): a learned "
                         "per-sample gate g=sigmoid(MLP(qual)) scales the explanation "
                         "stream so poor/boilerplate explanations get down-weighted. "
                         "L3-only by design (model.py); no-op for L1/L2.")
    ap.add_argument("--subset", type=int, default=None,
                    help="train on the first N samples only (fast directional / "
                         "mechanism-plot runs); val is scaled to max(50, N//4).")
    args = ap.parse_args()
    if args.evidence_window:
        os.environ["SEMVUL_CODE_WINDOW"] = "evidence"

    from train import train_rung  # after sys.path setup
    # reveal first when both requested: shorter, and it fails fast.
    jobs = _build_jobs(args)

    for seed in args.seeds:
        for ds, rungs, sub, suffix, fields, kw in jobs:
            os.environ["SEMVUL_TRAIN_SUFFIX"] = suffix
            os.environ["SEMVUL_EXPL_FIELDS"] = fields
            os.environ["SEMVUL_QUAL_GATE"] = "1" if args.qual_gate else "0"
            out_dir = os.path.join(RUNS, sub, f"s{seed}")
            for rung in rungs:
                # New runs write the "semanticvul_" tag (train.py TAG_PREFIX); older
                # completed L1/L2 runs on disk still carry the historical
                # "fusevul_ladder_" tag -- accept either so finished seeds are never
                # silently (and expensively) retrained after the rename.
                done = os.path.join(out_dir, f"semanticvul_{ds}_{rung}.json")
                done_legacy = os.path.join(out_dir, f"fusevul_ladder_{ds}_{rung}.json")
                if os.path.exists(done) or os.path.exists(done_legacy):
                    print(f"[skip] {ds} {rung} s{seed} done", flush=True)
                    continue
                print(f"\n===== {ds} {rung} seed={seed} ({sub}) =====", flush=True)
                train_rung(ds, rung, out_dir=out_dir, seed=seed,
                           split_seed=SPLIT_SEED, **kw)


if __name__ == "__main__":
    main()
