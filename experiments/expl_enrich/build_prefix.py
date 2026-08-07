"""Build explanation.prefix + prefix_recipe -- the materialized text channel.

WHY THIS FILE EXISTS
--------------------
`reproduce_real.py --fields prefix` (its DEFAULT) reads a pre-rendered
`explanation.prefix` string, and `_check_prefix_present()` hard-exits when it is
missing. No script in this repository has ever produced it:

  * `git log -S prefix_recipe --all` -> no commit ever contained the string
  * `git log --diff-filter=D` -> no script was ever deleted
  * the prefix-bearing files (`*_final_*_3.jsonl`, byte-identical to
    ACTIVE/{train,val}.jsonl) first appear in commit fb745e2 "Desktop 3",
    added as DATA
  * the files apply_real_enrichment.py writes (`*.enriched.real.jsonl`) have NO
    prefix column

So the column was produced off-repo on another machine and its builder was never
committed. This module reconstructs it from the enriched columns, so the pipeline
can regenerate the channel end to end.

RECONSTRUCTION FIDELITY (byte-exact vs ACTIVE val; run --verify for live numbers)
--------------------------------------------------------------------------------
  digest    lexical_digest                                     100%
  calls     ' '.join(called_functions[:15])                    100%
  missing   ' ; '.join(missing_checks[:8])          devign     100%
  strings   ' ; '.join(string_literals[:6])         reveal     100%
  mbin      binned code_metrics                     reveal     100%
  tail      tail_digest ' ; ' tail_facts            reveal     100%
  idparts   subword(function_name + called_functions) devign   99.9%
  strings   devign uses REAL literals (+reallit), not the field  ~91%
  evidence  reveal 'evrisk' selection not recovered            APPROXIMATE
  realcode  devign_real head, truncation length not recovered  APPROXIMATE

The first seven are exact. `evidence` and `realcode` are best-effort: their
selection/truncation rule could not be recovered from the data, so a rebuilt
prefix is NOT byte-identical to the legacy ACTIVE strings on those segments.
Training with `--fields prefix` on rebuilt data is therefore comparable but not
bit-reproducible against the historical numbers. The `final_*` wrappers pass an
explicit `--fields` column list and never touch prefix, so they are unaffected.

Usage:
  # rebuild in place for the ACTIVE files (adds prefix/prefix_recipe)
  .venv/Scripts/python.exe experiments/expl_enrich/build_prefix.py --only reveal
  # report byte-exact fidelity against the existing prefix, change nothing
  .venv/Scripts/python.exe experiments/expl_enrich/build_prefix.py --verify
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.config import EXPL_DIR  # noqa: E402

# recipe tags exactly as they appear in the legacy data
RECIPE = {"devign": "+subw,+realhead,+reallit",
          "reveal": "metbin,+lit,evrisk"}
SEG_ORDER = {"devign": ["digest", "missing", "calls", "idparts", "realcode", "strings"],
             "reveal": ["digest", "mbin", "calls", "evidence", "strings", "tail"]}

CALLS_CAP = 15          # derived: 100% on both datasets
MISSING_CAP = 8         # derived: 100% on devign
STRINGS_CAP = 6         # derived: 100% on reveal, 98.8% on devign
EVIDENCE_CAP = 8        # reveal 'evrisk': best recovered rule -> 76.9% exact
EVIDENCE_ITEM_CHARS = 120   # each conditional is truncated to this many chars

# reveal's recipe tag is "evrisk": the evidence segment is NOT explanation.
# evidence_tokens (9.7% exact). It is a fresh scan of raw_code for conditional
# statements that touch a dereference -- '->', '[' or '*' anywhere in the
# statement (not just the condition: "if ( codec ) * codec = -1 ;" qualifies on
# its body). Each item is cut to EVIDENCE_ITEM_CHARS and up to EVIDENCE_CAP are
# joined with ' ; '. That reaches 76.9% byte-exact; the residual selection detail
# could not be recovered, so this segment remains an approximation.
_COND = re.compile(r"(?:else\s+)?if \([^{;]*\)\s*(?:\{|[^;{]*;)")
_DEREF = re.compile(r"->|\[|\*")
REALCODE_WORDS = 60     # derived: 98.3% (sharp peak; 59 and 61 both score ~50%)
MBIN_CAP = 7            # derived: 100% on reveal
IDPART_MINLEN = 3       # derived: 99.9% on devign

# devign's recipe tag is "+reallit": its strings segment comes from literals in
# the REAL code, not from explanation.string_literals (98.8% vs 90.2% exact).
_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')

# metric -> mbin short name, in code_metrics insertion order
MBIN_NAMES = {
    "n_words": "words", "n_stmts": "stmts", "n_if": "if", "n_loops": "loops",
    "n_switch": "switch", "n_goto": "goto", "n_return": "return",
    "n_calls": "calls", "n_deref": "deref", "n_index": "index",
    "n_alloc": "alloc", "n_free": "free", "n_unsafe_str": "unsafe_str",
    "n_bounded_copy": "bounded_copy", "truncated": "truncated",
    "n_findings": "findings", "n_guards": "guards",
    "n_findings_tail": "findings_tail",
}
_WS = re.compile(r"\s+")


def _s(v) -> str:
    return "" if v is None else str(v)


def _list(v) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def subwords(name: str) -> list:
    """Split an identifier into lowercase word/number parts.

    Handles snake_case, camelCase and ACRONYMBoundaries ("updateMMXDither" ->
    update mmx dither), and separates trailing digits ("av_log2" -> av log 2).
    """
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", _s(name))
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return re.findall(r"[A-Za-z]+|[0-9]+", s)


def seg_idparts(e: dict) -> str:
    toks = subwords(e.get("function_name"))
    for c in _list(e.get("called_functions")):
        toks += subwords(c)
    out, seen = [], set()
    for t in (x.lower() for x in toks if len(x) >= IDPART_MINLEN):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out)


def seg_mbin(e: dict) -> str:
    m = e.get("code_metrics") or {}
    parts = []
    for k, v in m.items():
        if k in MBIN_NAMES and isinstance(v, (int, float)) and v > 0:
            b = min(MBIN_CAP, int(math.floor(math.log2(v))) + 1)
            parts.append(f"{MBIN_NAMES[k]}b{b}")
    return " ".join(parts)


def seg_realcode(e: dict, real_code: str) -> str:
    """Whitespace-normalized head of the REAL (de-anonymized) function body.

    Truncated to REALCODE_WORDS whitespace words -- a WORD cap, not a character
    cap: 60 words scores 98.3% exact while every character cap tops out near 46%.
    """
    return " ".join(_WS.sub(" ", _s(real_code)).strip().split()[:REALCODE_WORDS])


def seg_evidence(code: str) -> str:
    """reveal 'evrisk': dereference-touching conditionals scanned from raw_code."""
    out = []
    for m in _COND.finditer(_s(code)):
        s = m.group(0).strip()
        if _DEREF.search(s):
            out.append(s[:EVIDENCE_ITEM_CHARS])
        if len(out) >= EVIDENCE_CAP:
            break
    return " ; ".join(out)


def seg_real_literals(real_code: str) -> str:
    """devign '+reallit': deduped string literals lifted from the real code."""
    out, seen = [], set()
    for lit in _LITERAL.findall(_s(real_code)):
        if lit not in seen:
            seen.add(lit)
            out.append(lit)
    return " ; ".join(out[:STRINGS_CAP])


def build_prefix(e: dict, ds: str, real_code: str = "", code: str = "") -> str:
    """Compose the prefix string. `code` is the row's raw_code (reveal evidence);
    `real_code` is the de-anonymized source (devign realcode/strings)."""
    if ds == "devign":
        segs = {
            "digest": _s(e.get("lexical_digest")),
            "missing": " ; ".join(_list(e.get("missing_checks"))[:MISSING_CAP]),
            "calls": " ".join(_list(e.get("called_functions"))[:CALLS_CAP]),
            "idparts": seg_idparts(e),
            "realcode": seg_realcode(e, real_code),
            "strings": seg_real_literals(real_code),
        }
    else:
        tail = " ; ".join(x for x in (_s(e.get("tail_digest")),
                                     _s(e.get("tail_facts"))) if x)
        segs = {
            "digest": _s(e.get("lexical_digest")),
            "mbin": seg_mbin(e),
            "calls": " ".join(_list(e.get("called_functions"))[:CALLS_CAP]),
            "evidence": seg_evidence(code),
            "strings": " ; ".join(_list(e.get("string_literals"))[:STRINGS_CAP]),
            "tail": tail,
        }
        if not tail:
            segs.pop("tail")          # legacy: tail segment omitted when empty
    return " | ".join(f"{k}: {segs[k]}" for k in SEG_ORDER[ds] if k in segs)


# --- real-code lookup (devign realcode segment) ----------------------------

def load_real_codes(ds: str) -> dict:
    """sample_id -> real raw_code, from explanations/SemanticVul/devign_real/."""
    if ds != "devign":
        return {}
    out = {}
    for split in ("train", "val"):
        p = EXPL_DIR / "devign_real" / f"devign_real_{split}.jsonl"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(d.get("sample_id", ""))
                if sid:
                    out[sid] = d.get("raw_code", "") or ""
    return out


def active_path(ds: str, split: str):
    return EXPL_DIR / "ACTIVE" / ds / f"{split}.jsonl"


def _split_legacy(pr: str, ds: str) -> dict:
    """Split a legacy prefix on ' | <known-seg>: ' only ('|' occurs in content)."""
    pat = "|".join(re.escape(n) for n in SEG_ORDER[ds])
    marks = [(m.start(), m.group(1), m.end())
             for m in re.finditer(rf"(?:^|\s\|\s)({pat}):\s?", pr)]
    out = {}
    for i, (s, name, end_of_key) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(pr)
        out[name] = pr[end_of_key:stop]
    return out


def verify(datasets) -> int:
    print("Byte-exact fidelity of the reconstruction vs the legacy prefix\n")
    worst = 0
    for ds in datasets:
        reals = load_real_codes(ds)
        for split in ("train", "val"):
            p = active_path(ds, split)
            if not p.exists():
                continue
            seg_ok, seg_tot, whole_ok, n = {}, {}, 0, 0
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    e = r.get("explanation") or {}
                    if not e.get("prefix"):
                        continue
                    n += 1
                    got = _split_legacy(e["prefix"], ds)
                    built = build_prefix(e, ds,
                                         reals.get(str(r.get("sample_id", "")), ""),
                                         r.get("raw_code", "") or "")
                    mine = _split_legacy(built, ds)
                    if built == e["prefix"]:
                        whole_ok += 1
                    for k in SEG_ORDER[ds]:
                        if k not in got:
                            continue
                        seg_tot[k] = seg_tot.get(k, 0) + 1
                        if mine.get(k, "\0") == got[k]:
                            seg_ok[k] = seg_ok.get(k, 0) + 1
            if not n:
                continue
            print(f"  {ds}/{split}  rows with legacy prefix = {n}")
            for k in SEG_ORDER[ds]:
                if k in seg_tot:
                    pct = seg_ok.get(k, 0) / seg_tot[k] * 100
                    flag = "EXACT" if pct == 100 else ("close" if pct >= 90 else "APPROX")
                    print(f"     {k:9s} {seg_ok.get(k,0):6d}/{seg_tot[k]:<6d} "
                          f"{pct:6.2f}%  {flag}")
            print(f"     {'WHOLE':9s} {whole_ok:6d}/{n:<6d} {whole_ok/n*100:6.2f}%\n")
            worst = max(worst, 1 if whole_ok < n else 0)
    return worst


def rebuild(datasets, dry_run=False, out_dir=None) -> None:
    """Write prefix+prefix_recipe. In place on ACTIVE, or into out_dir if given."""
    for ds in datasets:
        reals = load_real_codes(ds)
        if ds == "devign":
            print(f"[prefix] devign real-code map: {len(reals)} functions")
        for split in ("train", "val"):
            p = active_path(ds, split)
            if not p.exists():
                print(f"[prefix] SKIP missing {p}")
                continue
            rows, changed = [], 0
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    e = r.get("explanation")
                    if isinstance(e, dict):
                        e["prefix"] = build_prefix(
                            e, ds, reals.get(str(r.get("sample_id", "")), ""),
                            r.get("raw_code", "") or "")
                        e["prefix_recipe"] = RECIPE[ds]
                        changed += 1
                    rows.append(r)
            if dry_run:
                print(f"[prefix] DRY-RUN {ds}/{split}: would rewrite {changed} rows")
                continue
            if out_dir:
                dest = os.path.join(out_dir, ds, f"{split}.jsonl")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
            else:
                dest = str(p)
            tmp = dest + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, dest)
            print(f"[prefix] {ds}/{split}: wrote prefix+prefix_recipe on "
                  f"{changed} rows -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["devign", "reveal"], default=None)
    ap.add_argument("--verify", action="store_true",
                    help="report byte-exact fidelity vs the legacy prefix; write nothing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="write <out-dir>/<ds>/<split>.jsonl instead of editing "
                         "ACTIVE in place (leaves the shipped data untouched)")
    args = ap.parse_args()
    datasets = [args.only] if args.only else ["devign", "reveal"]
    if args.verify:
        sys.exit(verify(datasets))
    rebuild(datasets, dry_run=args.dry_run, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
