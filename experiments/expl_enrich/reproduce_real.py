"""Train the L1--L3 ladder from ACTIVE inputs.

Final launchers validate ACTIVE before invoking this driver. Static enrichment,
recovered identifiers, lexical digests, and alternate dataset variants are not
selected here. Runs are resumable within their explicitly named cache family.
When SEMVUL_LEGACY_CACHE=1, existing cache folders are reused without
creating or checking a clean-Qwen contract; this is the compatibility path for
the original enriched-cache results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
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

# ACTIVE/ is the canonical, validated run input. The final launchers pass an
# explicit Qwen-only field list through SEMVUL_EXPL_FIELDS.
os.environ.setdefault("SEMVUL_QUAL_V2", "0")
os.environ["SEMVUL_ACTIVE_DIR"] = "1"

CACHE_CONTRACT = ".clean_qwen_contract.json"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_contract(dataset, rungs, fields, kwargs):
    active = os.path.join(ROOT, "explanations", "SemanticVul", "ACTIVE", dataset)
    inputs = {}
    for split in ("train", "val"):
        path = os.path.join(active, f"{split}.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing canonical ACTIVE input: {path}")
        inputs[split] = _sha256(path)
    return {
        "contract": "clean-qwen-active-v1",
        "dataset": dataset,
        "rungs": sorted(rungs),
        "fields": fields,
        "active_sha256": inputs,
        "training": kwargs,
        "gate": {
            "quality_gate": os.environ.get("SEMVUL_QUAL_GATE", "0"),
            "quality_v2": os.environ.get("SEMVUL_QUAL_V2", "0"),
            "gate_lr_multiplier": os.environ.get("SEMVUL_GATE_LR_MULT", ""),
            "hard_conf_switch": os.environ.get("SEMVUL_HARD_CONF_SWITCH", "0"),
            "hard_conf_threshold": os.environ.get("SEMVUL_HARD_CONF_THRESH", ""),
        },
    }


def _prepare_cache(dataset, rungs, subdir, fields, kwargs):
    """Bind a cache directory to one clean input/configuration contract.

    Existing unmarked result directories predate the clean-Qwen contract. They
    are rejected instead of being skipped or overwritten; archive them outside
    the canonical directory before starting the clean rerun.
    """
    out_root = os.path.join(RUNS, subdir)
    if os.environ.get("SEMVUL_LEGACY_CACHE") == "1":
        os.makedirs(out_root, exist_ok=True)
        return
    marker = os.path.join(out_root, CACHE_CONTRACT)
    expected = _cache_contract(dataset, rungs, fields, kwargs)
    completed = []
    if os.path.isdir(out_root):
        for base, _dirs, files in os.walk(out_root):
            completed.extend(os.path.join(base, name) for name in files
                             if name.endswith(".json") and "_partial" not in name
                             and name != CACHE_CONTRACT)
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8") as handle:
            actual = json.load(handle)
        if actual != expected:
            raise RuntimeError(
                f"cache contract mismatch in {out_root}; archive the directory "
                "before running this configuration"
            )
        return
    if completed:
        raise RuntimeError(
            f"legacy/unverified results exist in {out_root}; archive its current "
            "contents before creating clean-Qwen results in this canonical cache"
        )
    os.makedirs(out_root, exist_ok=True)
    temporary = marker + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, marker)


def _build_jobs(args):
    """Construct the (dataset, rungs, cache, suffix, fields, kwargs) jobs."""
    ga = max(1, 32 // args.batch512)
    reveal_fields = devign_fields = args.fields
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
         "clean_qwen_reveal" + args.out_tag, "", reveal_fields, reveal_kw),
        ("devign", args.rungs or ["L1", "L2", "L3"],
         "clean_qwen_devign" + args.out_tag, "", devign_fields,
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
    ap.add_argument("--fields", required=True,
                    help="comma-separated generator-produced explanation fields")
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
    os.environ["SEMVUL_QUAL_GATE"] = "1" if args.qual_gate else "0"
    if not any("L3" in rungs for _ds, rungs, _sub, _suffix, _fields, _kw in jobs):
        os.environ.pop("SEMVUL_HARD_CONF_SWITCH", None)
        os.environ.pop("SEMVUL_HARD_CONF_THRESH", None)
        os.environ.pop("SEMVUL_GATE_LR_MULT", None)

    for ds, rungs, sub, _suffix, fields, kw in jobs:
        _prepare_cache(ds, rungs, sub, fields, kw)

    for seed in args.seeds:
        for ds, rungs, sub, suffix, fields, kw in jobs:
            os.environ.pop("SEMVUL_TRAIN_SUFFIX", None)
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
