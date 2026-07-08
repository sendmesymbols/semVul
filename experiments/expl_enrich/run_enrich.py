"""Write enriched explanation JSONLs next to the originals.

  explanations/SemanticVul/<ds>/<ds>_<split>.jsonl
    -> explanations/SemanticVul/<ds>/<ds>_<split>.enriched.jsonl

Originals are never modified. Enrichment is deterministic and label-blind
(static_enrich.enrich_row never reads the label), so applying it to val is
legitimate: the same transform runs identically on any unseen function.

  .venv/Scripts/python.exe experiments/expl_enrich/run_enrich.py
"""
from __future__ import annotations
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from static_enrich import enrich_row
from src.config import EXPL_DIR


def main():
    for ds in ("devign", "reveal"):
        for split in ("train", "val"):
            src = EXPL_DIR / ds / f"{ds}_{split}.jsonl"
            dst = EXPL_DIR / ds / f"{ds}_{split}.enriched.jsonl"
            t0 = time.time()
            n = n_find = n_tail = 0
            with src.open("r", encoding="utf-8") as fi, \
                    dst.open("w", encoding="utf-8") as fo:
                for line in fi:
                    line = line.strip()
                    if not line:
                        continue
                    row = enrich_row(json.loads(line))
                    e = row["explanation"]
                    n += 1
                    n_find += int(bool(e["code_metrics"]["n_findings"]))
                    n_tail += int(bool(e["tail_facts"]))
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{ds}/{split}] {n} rows -> {dst.name}  "
                  f"rows-with-findings={100*n_find/n:.1f}%  "
                  f"rows-with-tail-facts={100*n_tail/n:.1f}%  "
                  f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
