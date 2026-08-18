"""Generate and activate clean Qwen explanation datasets.

Stages:
  1. Generate schema-constrained Qwen JSONL files.
  2. Install them as <dataset>_<split>.jsonl in the selected explanation tree.
  3. Validate and copy those files to ACTIVE/<dataset>/{train,val}.jsonl.

No deterministic static enrichment, de-anonymisation, lexical digest, recovered
identifier, prefix, or tail feature is produced by this pipeline.
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
DATASETS = ("devign", "reveal")
SPLITS = ("train", "val")

MANIFEST = """# ACTIVE explanations

These are the only explanation files consumed by final ladder runs. They are
direct copies of schema-constrained Qwen generator output; no static enrichment,
de-anonymisation, lexical digest, recovered identifier, prefix, or tail feature
is permitted. `experiments/explanation/validate_clean.py` enforces this contract.

`explanation.confidence` is measured from decode-time token probabilities over
the generated `risk_level` span. It is not derived from the dataset label, but it
is an LLM risk signal and must be reported and ablated as such.
"""


def run(cmd: list[object], env: dict[str, str], label: str) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    started = time.time()
    rc = subprocess.run([str(c) for c in cmd], env=env, cwd=str(ROOT)).returncode
    print(f"[{label}] exit={rc} ({time.time() - started:.1f}s)", flush=True)
    return rc


def gen_out(args, dataset: str, split: str) -> Path:
    tag = f"__{args.tag}" if args.tag else ""
    model = args.model.replace(":", "_").replace("/", "_").replace(".", "-")
    return HERE / "out" / f"{dataset}_{split}__{model}{tag}.jsonl"


def generate(args, py: str, env: dict[str, str], jobs) -> list[str]:
    failed: list[str] = []
    for dataset, split in jobs:
        cmd: list[object] = [
            py, HERE / "generate.py", "--dataset", dataset, "--split", split,
            "--model", args.model, "--host", args.host, "--mode", args.mode,
            "--num-ctx", args.num_ctx, "--timeout", args.timeout,
            "--workers", args.workers, "--out", gen_out(args, dataset, split),
        ]
        if args.stratified:
            cmd += ["--stratified", args.stratified]
        if args.no_think:
            cmd += ["--no-think"]
        if run(cmd, env, f"generate {dataset}/{split}") != 0:
            failed.append(f"generate/{dataset}/{split}")
    return failed


def install(args, tree: Path, jobs) -> list[str]:
    failed: list[str] = []
    for dataset, split in jobs:
        source = gen_out(args, dataset, split)
        if not source.exists():
            print(f"[install] MISSING {source}", flush=True)
            failed.append(f"install/{dataset}/{split}")
            continue
        target = tree / dataset / f"{dataset}_{split}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        print(f"[install] {source.name} -> {target}", flush=True)
    return failed


def activate(tree: Path, datasets) -> list[str]:
    from validate_clean import validate

    failed: list[str] = []
    for dataset in datasets:
        for split in SPLITS:
            source = tree / dataset / f"{dataset}_{split}.jsonl"
            if not source.exists():
                print(f"[activate] MISSING {source}", flush=True)
                failed.append(f"activate/{dataset}/{split}")
                continue
            rows, errors = validate(source)
            if errors:
                print(f"[activate] REJECT {source}", flush=True)
                for error in errors:
                    print(f"  - {error}", flush=True)
                failed.append(f"activate/{dataset}/{split}")
                continue
            target = tree / "ACTIVE" / dataset / f"{split}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            print(f"[activate] {dataset}/{split}: {rows} rows", flush=True)
    if not failed:
        manifest = tree / "ACTIVE" / "README.md"
        manifest.write_text(MANIFEST, encoding="utf-8")
        print(f"[activate] wrote {manifest}", flush=True)
    return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="both", choices=(*DATASETS, "both"))
    parser.add_argument("--split", default="both", choices=(*SPLITS, "both"))
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--host", default="http://localhost:9999")
    parser.add_argument("--mode", default="auto", choices=("auto", "anon", "real"))
    parser.add_argument("--stratified", default=None)
    parser.add_argument("--workers", default="1")
    parser.add_argument("--num-ctx", default="8192")
    parser.add_argument("--timeout", default="600")
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--from-stage", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--to-stage", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()

    if args.from_stage > args.to_stage:
        parser.error("--from-stage cannot exceed --to-stage")
    if args.smoke:
        args.stratified = args.stratified or "6"
        args.tag = args.tag or "smoke"

    datasets = DATASETS if args.dataset == "both" else (args.dataset,)
    splits = SPLITS if args.split == "both" else (args.split,)
    jobs = [(dataset, split) for dataset in datasets for split in splits]
    tree = SHIPPED_EXPL if args.promote else Path(args.work_dir or HERE / "work")
    tree.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.pop("SEMVUL_EXPL_DIR", None)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    failed: list[str] = []

    if args.from_stage <= 1 <= args.to_stage:
        failed += generate(args, sys.executable, env, jobs)
    if args.from_stage <= 2 <= args.to_stage:
        failed += install(args, tree, jobs)
    if args.from_stage <= 3 <= args.to_stage:
        if splits != SPLITS:
            print("[activate] both train and val are required", flush=True)
            failed.append("activate/incomplete-splits")
        else:
            failed += activate(tree, datasets)

    if failed:
        print("INCOMPLETE: " + ", ".join(sorted(set(failed))), flush=True)
        return 1
    print(f"PIPELINE OK -> {tree / 'ACTIVE'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
