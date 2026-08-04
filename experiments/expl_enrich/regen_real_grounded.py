"""Label-blind, real-code-grounded explanation regeneration for devign_real.

Runs the deterministic static enrichment on the REAL code (raw_code) with the
prior (anon-hallucinated) explanation blanked, so every finding/guard/evidence
is grounded in the actual function instead of VAR/FUN hallucination. The label
is NEVER read. Output aligns to the benchmark by sample_id (67% of Devign val).

This is a FAITHFULNESS/RQ1 enhancement (grounded explanations), NOT an accuracy
lever: experiments/scratchpad/real_expl_gate.py showed it adds ~0 ROC over the
code channel (the discriminative real-code signal is lexical/identifier-level and
does not survive abstraction into an explanation). Documented so no one mistakes
the artifact for an accuracy win.

  .venv/Scripts/python.exe experiments/expl_enrich/regen_real_grounded.py
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from static_enrich import enrich_row
from src.config import EXPL_DIR

SRC = EXPL_DIR / "devign_real"


def main():
    for split in ("train", "val"):
        src = SRC / f"devign_real_{split}.jsonl"
        dst = EXPL_DIR / "devign" / f"devign_{split}.real_grounded.jsonl"
        n = n_find = 0
        with src.open(encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
            for line in fi:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # blank the anon-derived prior explanation -> pure real-code findings.
                # label is present in the row but NEVER passed to enrich_row.
                er = enrich_row({"sample_id": row["sample_id"],
                                 "raw_code": row["raw_code"],
                                 "explanation": {}})
                out = {"sample_id": row["sample_id"],
                       "label": int(row["label"]),          # kept for training target only
                       "raw_code": row["raw_code"],          # REAL code
                       "explanation": er["explanation"]}
                n += 1
                n_find += int(bool(out["explanation"]["code_metrics"]["n_findings"]))
                fo.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"[devign_real/{split}] {n} rows -> {dst.name}  "
              f"grounded-findings={100*n_find/n:.1f}%", flush=True)


if __name__ == "__main__":
    main()
