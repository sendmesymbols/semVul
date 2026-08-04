"""Apply the label-blind real-code enrichment to explanation JSONLs -> *.real.jsonl.

Two deterministic transforms (see explanations/SemanticVul/devign_real/ENRICHMENT_RESULTS.md):
  1. DE-ANONYMIZATION (devign only): align anon benchmark code <-> devign_real
     raw_code token streams (99.3% lockstep) to recover VARn/FUNn -> real names,
     substitute into every explanation text field. Gate: +7.45 ROC over anon code
     (train->val TF-IDF, ~70% of the real-code upper bound).
  2. LEXICAL DIGEST (both datasets): function_name / called_functions / risky_apis /
     string_literals / lexical_digest extracted from the real code. ReVeal gate:
     CORE fields as separate channel +1.39 ROC (CI [+0.31,+2.63]).

Labels are copied through and NEVER influence any generated text. Rows without
real code pass through byte-identical (devign coverage ~66%; reveal 100%).
Output rows keep ALL original fields (quality_features_v2 needs code_metrics /
risk_level / confidence) — field SELECTION for the text channel happens at train
time via SEMVUL_EXPL_FIELDS, not here.

Usage (idempotent, ~5 min):
  .venv/Scripts/python.exe experiments/expl_enrich/apply_real_enrichment.py
  # only the files missing on disk:
  .venv/Scripts/python.exe experiments/expl_enrich/apply_real_enrichment.py --missing-only
  # verify presence, exit 1 if incomplete (used by reproduce_*.ps1):
  .venv/Scripts/python.exe experiments/expl_enrich/apply_real_enrichment.py --check
"""
from __future__ import annotations
import argparse
import difflib
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.config import EXPL_DIR  # noqa: E402

REAL_DIR = EXPL_DIR / "devign_real"

TOK = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*'
                 r'|0[xX][0-9a-fA-F]+|\d+\.?\d*|->|\+\+|--|<<=?|>>=?|<=|>=|==|!=|&&|\|\||\S')
ANON_ID = re.compile(r'^(VAR|FUN)\d+$')
C_KEYWORDS = set("""auto break case char const continue default do double else enum
extern float for goto if inline int long register restrict return short signed sizeof
static struct switch typedef union unsigned void volatile while _Bool _Complex bool
true false NULL""".split())
RISKY_APIS = set("""memcpy memmove memset strcpy strncpy strcat strncat sprintf
snprintf vsprintf vsnprintf gets scanf sscanf malloc calloc realloc free alloca strdup
strlen strtok atoi atol read write recv send system exec popen""".split())

TAIL_OFFSET_TOKENS = 220  # code-token offset ~= 320 GraphCodeBERT subwords (ReVeal window)

# (dataset, source basename, needs_deanon) -> writes <basename>.real.jsonl
TARGETS = [
    ("devign", "devign_val.enriched",           True),   # FULL benchmark val (ladder val input)
    ("devign", "devign_val.enriched.clean",     True),
    ("devign", "devign_val.clean",              True),
    ("devign", "devign_train.enriched.clean",   True),
    ("devign", "devign_train.enriched.clean.aug", True),
    ("devign", "devign_train.clean",            True),
    ("devign", "devign_train.clean.aug",        True),
    ("reveal", "reveal_val.enriched",           False),  # FULL benchmark val (ladder val input)
    ("reveal", "reveal_val.enriched.clean",     False),
    ("reveal", "reveal_val.clean",              False),
    ("reveal", "reveal_train.enriched.clean",   False),
    ("reveal", "reveal_train.clean",            False),
]


def strip_comments(code):
    code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', code)


def toks(code, norm_strings=True):
    out = []
    for t in TOK.findall(code):
        if norm_strings and t.startswith('"'):
            t = '""'
        elif norm_strings and t.startswith("'"):
            t = "''"
        out.append(t)
    return out


def build_map(anon_code, raw_code, cap=4000):
    a = toks(anon_code)[:cap]
    r = toks(strip_comments(raw_code))[:cap]
    votes = defaultdict(Counter)
    if len(a) == len(r):
        pairs = zip(a, r)
    else:
        pairs = []
        sm = difflib.SequenceMatcher(None, a, r, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'replace' and (i2 - i1) == (j2 - j1):
                pairs.extend(zip(a[i1:i2], r[j1:j2]))
    for at, rt in pairs:
        if ANON_ID.match(at) and re.match(r'^[A-Za-z_]\w*$', rt):
            votes[at][rt] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


def digest_fields(raw_code):
    ts = toks(strip_comments(raw_code), norm_strings=False)
    name = ''
    for i, t in enumerate(ts):
        if t == '(' and i > 0 and re.match(r'^[A-Za-z_]\w*$', ts[i - 1]) \
                and ts[i - 1] not in C_KEYWORDS:
            name = ts[i - 1]
            break
    callees, lits = [], []
    for i, t in enumerate(ts[:-1]):
        if re.match(r'^[A-Za-z_]\w*$', t) and t not in C_KEYWORDS \
                and ts[i + 1] == '(' and t != name:
            callees.append(t)
        if t.startswith('"') and len(t) > 2:
            lits.append(t)
    callee_list = [c for c, _ in Counter(callees).most_common(30)]
    risky = [c for c in callee_list if c in RISKY_APIS]
    lit_list = list(dict.fromkeys(lits))[:10]
    parts = [f"function {name}"]
    if callee_list:
        parts.append("calls " + " ".join(callee_list))
    if risky:
        parts.append("risky_apis " + " ".join(risky))
    if lit_list:
        parts.append("literals " + " ".join(lit_list))
    return name, callee_list, risky, lit_list, ". ".join(parts)


def tail_digest_fields(raw_code, offset=TAIL_OFFSET_TOKENS):
    """Digest of the code BEYOND the encoder window (ReVeal-only, label-blind).
    Same extraction as digest_fields but over tokens past `offset`, so it carries
    lexical signal the head-truncated code channel cannot see. Empty for functions
    at or below the window length."""
    ts = toks(strip_comments(raw_code), norm_strings=False)
    if len(ts) <= offset:
        return ""
    tail = ts[offset:]
    callees, lits = [], []
    for i, t in enumerate(tail[:-1]):
        if re.match(r'^[A-Za-z_]\w*$', t) and t not in C_KEYWORDS \
                and tail[i + 1] == '(':
            callees.append(t)
        if t.startswith('"') and len(t) > 2:
            lits.append(t)
    callee_list = [c for c, _ in Counter(callees).most_common(30)]
    risky = [c for c in callee_list if c in RISKY_APIS]
    lit_list = list(dict.fromkeys(lits))[:10]
    parts = []
    if callee_list:
        parts.append("tail_calls " + " ".join(callee_list))
    if risky:
        parts.append("tail_risky_apis " + " ".join(risky))
    if lit_list:
        parts.append("tail_literals " + " ".join(lit_list))
    return ". ".join(parts)


def load_real_index():
    idx = {}
    for split in ("train", "val"):
        p = REAL_DIR / f"devign_real_{split}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing (devign_real download)")
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    idx[row["sample_id"]] = row["raw_code"]
    return idx


def enrich_row(r, real_idx, deanon, tail_offset=TAIL_OFFSET_TOKENS):
    """Returns True if the row was treated (mutates r)."""
    if deanon:
        sid = r.get("sample_id")
        raw = real_idx.get(sid)
        if raw is None and isinstance(sid, str):  # aug rows may suffix the id
            raw = real_idx.get(re.split(r"[_#]", sid)[0])
        if raw is None:
            return False
        m = build_map(r.get("raw_code") or "", raw)
        expl = r.get("explanation") or {}
        s = json.dumps(expl)
        s = re.sub(r'\b(VAR|FUN)\d+\b', lambda mo: m.get(mo.group(0), mo.group(0)), s)
        expl = json.loads(s)
        tag = "deanon+digest-v1"
        _add_tail = False               # Devign never gets tail_digest
    else:
        raw = r.get("raw_code") or ""   # reveal code is already real
        expl = r.get("explanation") or {}
        tag = "digest-v1"
        _add_tail = True                # ReVeal-only beyond-window channel
    name, callees, risky, lits, dig = digest_fields(raw)
    expl["function_name"] = name
    expl["called_functions"] = callees
    expl["risky_apis"] = risky
    expl["string_literals"] = lits
    expl["lexical_digest"] = dig
    expl["real_enrich"] = tag
    if _add_tail:
        expl["tail_digest"] = tail_digest_fields(raw, tail_offset)
    r["explanation"] = expl
    return True


# The two files per dataset that a run actually reads. apply refreshes copies of
# these into explanations/SemanticVul/ACTIVE/<ds>/{train,val}.jsonl so ACTIVE is
# always the current single source of truth (incl. ReVeal's per-run tail_digest).
CANONICAL = {
    "devign": {"train": "devign_train.enriched.clean.aug.real.jsonl",
               "val":   "devign_val.enriched.real.jsonl"},
    "reveal": {"train": "reveal_train.enriched.clean.real.jsonl",
               "val":   "reveal_val.enriched.real.jsonl"},
}


def refresh_active(only):
    import shutil
    for ds, roles in CANONICAL.items():
        if only and ds != only:
            continue
        for split, fname in roles.items():
            src = EXPL_DIR / ds / fname
            if not src.exists():
                continue
            dst = EXPL_DIR / "ACTIVE" / ds / f"{split}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            print(f"[real-enrich] ACTIVE refreshed: ACTIVE/{ds}/{split}.jsonl", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing-only", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="verify all outputs exist; exit 1 if any is missing")
    ap.add_argument("--only", choices=["reveal", "devign"], default=None,
                    help="restrict to one dataset's TARGETS (Devign-safe)")
    ap.add_argument("--tail-offset", type=int, default=TAIL_OFFSET_TOKENS)
    args = ap.parse_args()

    targets = [t for t in TARGETS if args.only is None or t[0] == args.only]

    if args.check:
        # A dataset is satisfied if EITHER its .real TARGETS exist OR the ACTIVE
        # consolidated pair exists (copying just ACTIVE is a valid setup).
        need_ds = {args.only} if args.only else {"devign", "reveal"}
        active_ok = all((EXPL_DIR / "ACTIVE" / ds / f"{s}.jsonl").exists()
                        for ds in need_ds for s in ("train", "val"))
        missing = [str(EXPL_DIR / ds / f"{base}.real.jsonl")
                   for ds, base, _ in targets
                   if not (EXPL_DIR / ds / f"{base}.real.jsonl").exists()]
        if missing and not active_ok:
            print("[real-enrich] MISSING:\n  " + "\n  ".join(missing))
            print("run: python experiments/expl_enrich/apply_real_enrichment.py")
            print("(or copy explanations/SemanticVul/ACTIVE/ onto this machine)")
            sys.exit(1)
        print(f"[real-enrich] inputs present ({'ACTIVE' if missing else 'full .real set'})")
        return

    real_idx = None
    for ds, base, deanon in targets:
        src = EXPL_DIR / ds / f"{base}.jsonl"
        dst = EXPL_DIR / ds / f"{base}.real.jsonl"
        if not src.exists():
            print(f"[skip] {src.name} missing (source not built on this machine)")
            continue
        if args.missing_only and dst.exists():
            print(f"[skip] {dst.name} exists")
            continue
        if deanon and real_idx is None:
            print("[real-enrich] indexing devign_real raw code...", flush=True)
            real_idx = load_real_index()
            print(f"[real-enrich] real code for {len(real_idx)} sample_ids", flush=True)
        n = hit = 0
        with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                n += 1
                hit += enrich_row(r, real_idx, deanon, args.tail_offset)
                fout.write(json.dumps(r) + "\n")
        print(f"[real-enrich] {base}: {hit}/{n} treated -> {dst.name}", flush=True)
    refresh_active(args.only)
    print("[real-enrich] done")


if __name__ == "__main__":
    main()
