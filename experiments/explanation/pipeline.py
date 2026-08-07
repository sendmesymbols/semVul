"""THE explanation pipeline -- one entry point, all six stages.

generate_explanations.ps1 and generate_explanations.sh are thin shims over this
module, so both platforms run byte-identical logic. Nothing else needs calling:
organize_explanations.ps1's job (promote the .real files into ACTIVE/, write the
manifest, build explanation.prefix) is stage 6 here.

STAGES -- each writes the explanation.* columns named on the right:

  1 generate   experiments/explanation/generate.py
               purpose, data_flow, risky_operations, missing_checks,
               evidence_tokens, safety_indicators, risk_summary, risk_level,
               confidence (measured from decode-time logprobs)
  2 install    stage-1 output -> <EXPL>/<ds>/<ds>_<split>.jsonl (the name every
               later stage reads)
  3 enrich     expl_enrich/run_enrich.py            -> llm_v1, code_metrics,
                                                       tail_facts, enrich
  4 clean/aug  expl_enrich/correct_val.py (val)
               expl_enrich/augment_train.py (train) -> .clean / .clean.aug
               variants that the canonical ACTIVE sources are built from
  5 real       expl_enrich/apply_real_enrichment.py -> function_name,
               called_functions, risky_apis, string_literals, lexical_digest,
               real_enrich, tail_digest  (+ refreshes ACTIVE/)
  6 prefix     expl_enrich/build_prefix.py          -> prefix, prefix_recipe
               (+ ACTIVE/README.md manifest)

WHERE IT WRITES
By default everything lands in a WORK TREE (--work-dir, default
experiments/explanation/work/) via SEMVUL_EXPL_DIR, so the shipped
explanations/SemanticVul/ is never modified. The work tree is seeded with the
read-only inputs the later stages need (devign_real/ for de-anonymization).
Pass --promote to run against the real tree instead -- that DOES overwrite the
shipped files, so it is opt-in.

  # full regeneration into the work tree (safe)
  python experiments/explanation/pipeline.py --model qwen2.5-coder:14b
  # end-to-end proof on 6 rows/split
  python experiments/explanation/pipeline.py --smoke
  # skip generation, just rebuild the derived columns from existing stage-1 output
  python experiments/explanation/pipeline.py --from-stage 2
  # write to the real tree (overwrites shipped data)
  python experiments/explanation/pipeline.py --promote

Resumable: stage 1 skips sample_ids already present in its output file, so a
re-run continues where an interrupted one stopped.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SHIPPED_EXPL = ROOT / "explanations" / "SemanticVul"
ENRICH = ROOT / "experiments" / "expl_enrich"

DATASETS = ("devign", "reveal")
SPLITS = ("train", "val")
# read-only inputs later stages need that generation does not produce
SEED_DIRS = ("devign_real",)

MANIFEST = """# ACTIVE explanations -- the ONLY files a run consumes

Written by experiments/explanation/pipeline.py (stage 6). Each dataset uses
exactly two files; the training wrappers read these via SEMVUL_ACTIVE_DIR=1.

| dataset | role  | ACTIVE copy        | built from                                 |
|---------|-------|--------------------|--------------------------------------------|
| devign  | train | devign/train.jsonl | devign_train.enriched.clean.aug.real.jsonl |
| devign  | val   | devign/val.jsonl   | devign_val.enriched.real.jsonl             |
| reveal  | train | reveal/train.jsonl | reveal_train.enriched.clean.real.jsonl     |
| reveal  | val   | reveal/val.jsonl   | reveal_val.enriched.real.jsonl             |

explanation.confidence is MEASURED from the generator's decode-time token
logprobs over the risk_level verdict span (experiments/explanation/generate.py),
not self-reported and not derived from the label.

explanation.prefix is rebuilt by experiments/expl_enrich/build_prefix.py. Check
its byte-fidelity against a reference tree at any time with `--verify`.
"""


def run(cmd, env, label) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run([str(c) for c in cmd], env=env, cwd=str(ROOT)).returncode
    print(f"[{label}] exit={rc} ({time.time() - t0:.1f}s)", flush=True)
    return rc


def seed_work_tree(work: Path, datasets) -> None:
    """Copy the read-only inputs the later stages need into the work tree."""
    for ds in datasets:
        (work / ds).mkdir(parents=True, exist_ok=True)
    (work / "ACTIVE").mkdir(parents=True, exist_ok=True)
    for d in SEED_DIRS:
        src, dst = SHIPPED_EXPL / d, work / d
        if src.is_dir() and not dst.exists():
            print(f"[seed] {d}/ -> work tree", flush=True)
            shutil.copytree(src, dst)


def stage1_generate(args, py, env, jobs) -> list:
    failed = []
    gen = HERE / "generate.py"
    for ds, sp in jobs:
        cmd = [py, gen, "--dataset", ds, "--split", sp, "--model", args.model,
               "--host", args.host, "--mode", args.mode,
               "--num-ctx", args.num_ctx, "--timeout", args.timeout,
               "--workers", args.workers, "--out", str(gen_out(args, ds, sp))]
        if args.stratified:
            cmd += ["--stratified", args.stratified]
        if args.no_think:
            cmd += ["--no-think"]
        if run(cmd, env, f"gen {ds}/{sp}") != 0:
            failed.append(f"{ds}/{sp}")
    return failed


def gen_out(args, ds, sp) -> Path:
    tag = f"__{args.tag}" if args.tag else ""
    model = args.model.replace(":", "_").replace("/", "_").replace(".", "-")
    return HERE / "out" / f"{ds}_{sp}__{model}{tag}.jsonl"


def stage2_install(args, expl, jobs) -> list:
    """Put stage-1 output where every later stage looks for it."""
    failed = []
    for ds, sp in jobs:
        src = gen_out(args, ds, sp)
        if not src.exists():
            print(f"[install] MISSING {src} -- stage 1 produced nothing", flush=True)
            failed.append(f"{ds}/{sp}")
            continue
        dst = expl / ds / f"{ds}_{sp}.jsonl"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        n = sum(1 for _ in dst.open(encoding="utf-8"))
        print(f"[install] {src.name} -> {dst.relative_to(expl.parent)} ({n} rows)",
              flush=True)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--dataset", default="both", choices=["devign", "reveal", "both"])
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    ap.add_argument("--model", default="qwen2.5-coder:14b")
    ap.add_argument("--host", default="http://localhost:9999")
    ap.add_argument("--mode", default="auto", choices=["auto", "anon", "real"])
    ap.add_argument("--stratified", default=None)
    ap.add_argument("--workers", default="1")
    ap.add_argument("--num-ctx", default="8192")
    ap.add_argument("--timeout", default="600")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="6 rows per split, tag 'smoke' -- end-to-end proof")
    ap.add_argument("--work-dir", default=None,
                    help="tree to build in (default experiments/explanation/work)")
    ap.add_argument("--promote", action="store_true",
                    help="build in the SHIPPED explanations/SemanticVul tree "
                         "(OVERWRITES shipped data; off by default)")
    ap.add_argument("--from-stage", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--to-stage", type=int, default=6, choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--aug-copies", default="1")
    ap.add_argument("--tail-offset", default="220")
    args = ap.parse_args()

    if args.smoke:
        args.stratified = args.stratified or "6"
        args.tag = args.tag or "smoke"

    datasets = DATASETS if args.dataset == "both" else (args.dataset,)
    splits = SPLITS if args.split == "both" else (args.split,)
    jobs = [(d, s) for d in datasets for s in splits]

    if args.promote:
        expl = SHIPPED_EXPL
        print("!! --promote: building in the SHIPPED tree; existing files will be "
              "overwritten", flush=True)
    else:
        expl = Path(args.work_dir) if args.work_dir else HERE / "work"
        expl.mkdir(parents=True, exist_ok=True)
        seed_work_tree(expl, datasets)

    py = sys.executable
    _pp = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Stages 2-6 build inside `expl`.
    env = dict(os.environ)
    env["SEMVUL_EXPL_DIR"] = str(expl)
    env["PYTHONPATH"] = _pp

    # Stage 1 is different: it READS the source functions (raw_code + labels) via
    # load_split, which resolves against EXPL_DIR. That input only exists in the
    # SHIPPED tree, so generation must not be pointed at the (initially empty)
    # work tree -- it writes to --out, which is outside the tree anyway.
    env_gen = dict(os.environ)
    env_gen.pop("SEMVUL_EXPL_DIR", None)
    env_gen["PYTHONPATH"] = _pp

    print(f"\n{'=' * 74}\nexplanation pipeline\n{'=' * 74}")
    print(f"  tree     : {expl}{'  (SHIPPED)' if args.promote else '  (work)'}")
    print(f"  jobs     : {', '.join(f'{d}/{s}' for d, s in jobs)}")
    print(f"  model    : {args.model}   mode={args.mode}   workers={args.workers}")
    print(f"  stages   : {args.from_stage}..{args.to_stage}")
    print(f"  rows     : {args.stratified or 'ALL'}")

    # Stages 4-6 build the canonical ACTIVE pair, which needs BOTH splits of a
    # dataset: corrected val is produced by checking val for leakage against
    # train, and ACTIVE/train comes from the .clean[.aug] train variants. Say so
    # now rather than dying inside stage 4.
    if args.to_stage >= 4 and len(splits) < 2:
        print(f"\n  !! --split {args.split} with stages >=4: ACTIVE cannot be "
              f"completed.\n     Corrected val is derived by comparing val "
              f"against train, so stages 4-6\n     need both splits present in "
              f"{expl}.\n     Stages 1-3 will still run for {args.split}; use "
              f"--split both for a full build.", flush=True)

    failed: list = []

    def active(stage):
        return args.from_stage <= stage <= args.to_stage

    if active(1):
        print(f"\n### stage 1/6  generate")
        failed += stage1_generate(args, py, env_gen, jobs)
    if active(2):
        print(f"\n### stage 2/6  install")
        failed += stage2_install(args, expl, jobs)
    if active(3):
        print(f"\n### stage 3/6  static enrichment")
        for ds in datasets:
            for sp in splits:
                run([py, ENRICH / "run_enrich.py", "--only", ds, "--split", sp],
                    env, f"enrich {ds}/{sp}")
    if active(4):
        print(f"\n### stage 4/6  clean / augment")
        for ds in datasets:
            run([py, ENRICH / "correct_val.py", "--only", ds], env, f"clean-val {ds}")
            for variant in ("", "enriched"):
                run([py, ENRICH / "augment_train.py", "--only", ds,
                     "--variant", variant, "--aug-copies", args.aug_copies], env,
                    f"augment-train[{variant or 'plain'}] {ds}")
    if active(5):
        print(f"\n### stage 5/6  real-code enrichment")
        for ds in datasets:
            if run([py, ENRICH / "apply_real_enrichment.py", "--only", ds,
                    "--tail-offset", args.tail_offset], env, f"real {ds}") != 0:
                failed.append(f"real/{ds}")
    if active(6):
        print(f"\n### stage 6/6  prefix + manifest")
        for ds in datasets:
            if run([py, ENRICH / "build_prefix.py", "--only", ds], env,
                   f"prefix {ds}") != 0:
                failed.append(f"prefix/{ds}")
        readme = expl / "ACTIVE" / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(MANIFEST, encoding="utf-8")
        print(f"[manifest] wrote {readme}")

    print(f"\n{'=' * 74}")
    act = expl / "ACTIVE"
    for ds in datasets:
        for sp in splits:
            p = act / ds / f"{sp}.jsonl"
            if p.exists():
                n = sum(1 for _ in p.open(encoding="utf-8"))
                print(f"  ACTIVE/{ds}/{sp}.jsonl  {n} rows")
            else:
                print(f"  ACTIVE/{ds}/{sp}.jsonl  MISSING")
                failed.append(f"active/{ds}/{sp}")
    if failed:
        print(f"\nINCOMPLETE: {', '.join(sorted(set(failed)))}")
        print("re-run the same command to resume (stage 1 skips finished rows)")
        return 1
    print(f"\nPIPELINE OK -> {act}")
    if not args.promote:
        print("This is the work tree. To use it for training, either re-run with "
              "--promote,\nor point the run at it: SEMVUL_EXPL_DIR=" + str(expl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
