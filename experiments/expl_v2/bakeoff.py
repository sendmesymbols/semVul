"""Bake-off driver: same 150 stratified Devign samples through each candidate.

Resumable — rerun and finished models/samples are skipped. coder-14b is polled
for (it is downloading separately) up to 3 h before being skipped.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
GEN = os.path.join(HERE, "generate_v2.py")
EV = os.path.join(HERE, "pilot_eval.py")
N = "150"
HOST = "http://localhost:9999"

MODELS = [
    ("qwen2.5-coder:7b", []),
    ("mannix/deepseek-coder-v2-lite-instruct:latest", []),
    ("phi4:14b", []),
    ("qwen3.5:9b", ["--no-think"]),
]


def run(model: str, extra):
    cmd = [PY, GEN, "--dataset", "devign", "--split", "train",
           "--model", model, "--stratified", N] + extra
    print(f"[bakeoff] >>> {model}", flush=True)
    subprocess.run(cmd, check=False)


def model_available(name: str) -> bool:
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=15) as r:
            tags = [m["name"] for m in json.loads(r.read())["models"]]
        return name in tags
    except OSError:
        return False


def main():
    for model, extra in MODELS:
        run(model, extra)

    deadline = time.time() + 3 * 3600
    while time.time() < deadline:
        if model_available("qwen2.5-coder:14b"):
            run("qwen2.5-coder:14b", [])
            break
        time.sleep(120)
    else:
        print("[bakeoff] qwen2.5-coder:14b never appeared; skipped", flush=True)

    run("qwen3-coder:480b-cloud", [])

    print("\n[bakeoff] ===== FINAL SCOREBOARD =====", flush=True)
    subprocess.run([PY, EV, os.path.join(HERE, "out", "devign_train__*.jsonl")],
                   check=False)


if __name__ == "__main__":
    main()
