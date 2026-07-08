"""Build a de-anonymized Devign dataset (path C).

Pairs the real code in `explanations/SemanticVul/devign/full_code/function.json`
to the existing anonymized split via a comment-stripped, anonymization-invariant
structural fingerprint (every non-keyword identifier -> ID; strings/chars/numbers
class-collapsed; whitespace-independent). Only label-consistent matches are kept,
so the paired set is 100% label-correct by construction.

Output: explanations/SemanticVul/devign_real/devign_real_{train,val}.jsonl
Each row carries the SAME sample_id/label/explanation as the anonymized row, with
  raw_code  = the REAL function (de-anonymized)
  anon_code = the original anonymized text (kept for paired anon-vs-real eval)

Run:  .venv/Scripts/python.exe -m experiments.build_devign_real   (or run directly)
"""
import json
import re
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPL = ROOT / "explanations" / "SemanticVul"
REAL_JSON = EXPL / "devign" / "full_code" / "function.json"
OUT_DIR = EXPL / "devign_real"

KW = set("""auto break case char const continue default do double else enum extern
float for goto if inline int long register return short signed sizeof static
struct switch typedef union unsigned void volatile while _Bool size_t""".split())
TOK = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\d[\w.]*|[A-Za-z_]\w*|[^\s]')
CMT = re.compile(r'/\*.*?\*/|//[^\n]*', re.DOTALL)


def fingerprint(code: str, strip_comments: bool = False) -> str:
    if strip_comments:
        code = CMT.sub(" ", code)
    out = []
    for t in TOK.findall(code):
        c = t[0]
        if c == '"':                      out.append("S")
        elif c == "'":                    out.append("C")
        elif c.isdigit():                 out.append("N")
        elif c.isalpha() or c == "_":     out.append(t if t in KW else "ID")
        else:                             out.append(t)
    return hashlib.sha1(" ".join(out).encode()).hexdigest()[:16]


def build():
    real = json.load(open(REAL_JSON, encoding="utf-8"))
    # comment-stripped fingerprint -> list of (real_func, label)
    rmap = defaultdict(list)
    for r in real:
        rmap[fingerprint(r["func"], strip_comments=True)].append(
            (r["func"], int(r["target"])))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    totals = {}
    for split in ("train", "val"):
        src = EXPL / "devign" / f"devign_{split}.jsonl"
        out = OUT_DIR / f"devign_real_{split}.jsonl"
        n = matched = ambiguous = 0
        with open(src, encoding="utf-8") as fh, open(out, "w", encoding="utf-8") as w:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                n += 1
                # anon side has no comments; do NOT strip
                cands = rmap.get(fingerprint(r["raw_code"]), [])
                pick = next((f for f, t in cands if t == int(r["label"])), None)
                if pick is None:
                    if cands:
                        ambiguous += 1  # matched structure but label disagreed
                    continue
                matched += 1
                w.write(json.dumps({
                    "sample_id": r["sample_id"],
                    "label": r["label"],
                    "raw_code": pick,               # REAL de-anonymized code
                    "anon_code": r["raw_code"],     # original anonymized text
                    "explanation": r.get("explanation", {}),
                }, ensure_ascii=False) + "\n")
        totals[split] = (matched, n, ambiguous)
        print(f"{split}: paired {matched}/{n} ({100*matched/n:.1f}%)  "
              f"label-mismatch-skipped={ambiguous}  -> {out.relative_to(ROOT)}")
    return totals


if __name__ == "__main__":
    build()
