"""Regenerate explanations with prompt/schema v2 via local Ollama.

Resumable: already-generated sample_ids in the output file are skipped, so the
same command can be re-run after interruption. The ground-truth label is
written to the OUTPUT row (needed downstream for training) but is NEVER part
of the prompt.

Usage (bake-off, 300 stratified samples, deterministic across models):
  .venv/Scripts/python.exe experiments/expl_v2/generate_v2.py \
      --dataset devign --split train --model qwen2.5-coder:14b --stratified 300

Full split:
  .venv/Scripts/python.exe experiments/expl_v2/generate_v2.py \
      --dataset devign --split train --model <winner>
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_io import load_split          # noqa: E402
from prompt_v2 import JSON_SCHEMA, build_messages  # noqa: E402

MAX_CODE_CHARS = 12000  # ~3k tokens; median function is ~145 tokens


def sanitize(model: str) -> str:
    return model.replace(":", "_").replace("/", "_").replace(".", "-")


def default_out(dataset: str, split: str, model: str, tag: str) -> str:
    name = f"{dataset}_{split}__{sanitize(model)}{('__' + tag) if tag else ''}.jsonl"
    return os.path.join(HERE, "out", name)


def stratified_pick(samples, n: int, seed: int):
    """Deterministic per-class pick so every bake-off model sees the SAME set."""
    by_label = {0: [], 1: []}
    for s in samples:
        if s.label in by_label:
            by_label[s.label].append(s)
    rng = random.Random(seed)
    picked = []
    for lab in (0, 1):
        pool = sorted(by_label[lab], key=lambda s: s.sample_id)
        rng.shuffle(pool)
        picked.extend(pool[: n // 2])
    picked.sort(key=lambda s: s.sample_id)
    return picked


def ollama_chat(host: str, model: str, messages, num_ctx: int, timeout: int,
                no_think: bool):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": JSON_SCHEMA,
        "options": {"temperature": 0.0, "seed": 1234, "num_ctx": num_ctx},
    }
    if no_think:
        payload["think"] = False
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["devign", "reveal"])
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="http://localhost:9999")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="", help="suffix for the default out name")
    ap.add_argument("--stratified", type=int, default=None,
                    help="pick N samples (N/2 per class), deterministic")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--no-think", action="store_true",
                    help='send "think": false (qwen3-family thinking models)')
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests; set OLLAMA_NUM_PARALLEL>=workers "
                         "on the server or requests just queue")
    args = ap.parse_args()

    out_path = args.out or default_out(args.dataset, args.split, args.model, args.tag)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    samples = load_split(args.dataset, args.split)
    if args.stratified:
        samples = stratified_pick(samples, args.stratified, args.seed)
    if args.limit:
        samples = samples[: args.limit]

    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    todo = [s for s in samples if s.sample_id not in done]
    print(f"[gen] {args.dataset}/{args.split} model={args.model} "
          f"total={len(samples)} done={len(done)} todo={len(todo)} -> {out_path}",
          flush=True)

    def gen_one(s):
        code = s.code[:MAX_CODE_CHARS]
        err = None
        for attempt in (1, 2):
            try:
                t1 = time.time()
                resp = ollama_chat(args.host, args.model,
                                   build_messages(code),
                                   args.num_ctx, args.timeout,
                                   args.no_think)
                dur_s = time.time() - t1
                expl = json.loads(resp["message"]["content"])
                return {
                    "sample_id": s.sample_id,
                    "label": s.label,
                    "raw_code": s.code,
                    "explanation": expl,
                    "meta": {"model": args.model, "prompt": "v2",
                             "gen_seconds": round(dur_s, 2)},
                }, None
            except (urllib.error.URLError, urllib.error.HTTPError,
                    json.JSONDecodeError, KeyError, TimeoutError,
                    OSError) as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 * attempt)
        return None, f"{s.sample_id}: {err}"

    n_ok, n_fail, t0 = 0, 0, time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(gen_one, s) for s in todo]
            for i, fut in enumerate(as_completed(futures)):
                row, err = fut.result()
                if row is None:
                    n_fail += 1
                    print(f"[gen] FAIL {err}", flush=True)
                    continue
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n_ok += 1
                if n_ok % 10 == 0 or i == len(todo) - 1:
                    rate = n_ok / max(time.time() - t0, 1e-9)
                    eta_min = (len(todo) - i - 1) / max(rate, 1e-9) / 60
                    print(f"[gen] {n_ok}/{len(todo)} ok ({n_fail} fail) "
                          f"{rate:.2f}/s eta {eta_min:.0f} min", flush=True)

    print(f"[gen] DONE ok={n_ok} fail={n_fail} "
          f"elapsed={(time.time() - t0) / 60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
