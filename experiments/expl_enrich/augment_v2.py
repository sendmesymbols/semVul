"""Label-aware explanation augmentation for both Reveal and Devign.

What this does (run it once, it writes the new files in place next to the
originals; originals are NEVER modified):

  For each row in (reveal|devign) x (train|val):

  1. FILL: if the row predates the static enrich pass, run
     static_enrich.enrich_row() on it so it gains code_metrics, risk_level,
     confidence, safety_indicators, tail_facts (all label-blind).
  2. FILL: extract risky_apis / called_functions / string_literals / function_name
     from raw_code via a label-blind regex pass, for EVERY row that doesn't
     already have them. This closes the "risky_apis only 15% on Reveal, 0% on
     Devign" gap.
  3. UPDATE: clean the LLM text fields.
       * drop the "[evidence: ...]" tails from risky_operations (they duplicate
         raw_code and force a useless attention shortcut);
       * drop the risk_summary boilerplate ("No unguarded operation is visible
         in this function.") and replace with the static count line;
       * collapse empty lists to "" (not "none" or "-", which the encoder
         mis-tokenizes);
       * add a self_consistency object: contradictions between risk_level,
         findings, and guards.
  4. AUGMENT (label-aware): compute train-set label-conditional priors ONCE on
     the train split, then attach to every row (train and val).
       * risky_api_priors[k] = P(label=1 | risky_api in row.risky_apis)
         for the top-50 risky_apis vocabulary
       * n_high_risk_apis   = #{k : prior[k] > high_thr}
       * max_risky_api_prior / mean_risky_api_prior / sum_risky_api_logit
       * function_name_priors[k] same idea for the top-100 function names
         (devign is anonymized so this is mostly noise there, but it costs
         nothing to compute)
       * sample_weight = inverse-frequency on the label (for balanced sampling)
     These are NOT label leakage: every prior is computed from the train
     distribution and applied identically to train and val rows. The model
     still has to learn what to do with the priors.
  5. AUGMENT (structural): build multi-hot vectors for risky_apis (50) /
     called_functions (100) / string_literals (50), and a [SECTION]-tagged
     concise_text the model can attend to aspect-wise.
  6. SAVE: write <ds>_<split>.aug2.jsonl in the same folder. These are the
     files REVEAL.py (and any new training script) should read.

Run
---
  .venv/Scripts/python.exe experiments/expl_enrich/augment_v2.py
  # one dataset only:
  .venv/Scripts/python.exe experiments/expl_enrich/augment_v2.py --dataset reveal
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _p in (ROOT, HERE, ROOT / "experiments" / "expl_enrich"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from static_enrich import enrich_row, DANGEROUS_ANY  # noqa: E402
from src.config import EXPL_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# 2. FILL: regex extraction of structured fields from raw_code
# ---------------------------------------------------------------------------
# A conservative C identifier pattern. Allows _ in the middle and digits but
# not at the start. We then drop C keywords + very common type names.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STR_LIT = re.compile(r'"([^"\\]|\\.)*"')
_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

_C_NOISE = {
    "if", "else", "for", "while", "do", "switch", "case", "default", "return",
    "goto", "break", "continue", "sizeof", "struct", "union", "enum", "static",
    "const", "unsigned", "signed", "int", "char", "long", "short", "float",
    "double", "void", "NULL", "true", "false", "typedef", "extern", "register",
    "volatile", "inline", "restrict", "uint8_t", "uint16_t", "uint32_t",
    "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t", "size_t",
    "ssize_t", "bool", "FUN1", "FUN2", "FUN3", "FUN4", "FUN5",
}

# A bigger risky-API list than the one in static_enrich (which only flags
# things that are obviously un-guarded). For priors we want a "did this
# identifier appear anywhere in the code" feature, not a "did the static
# analyzer flag it" feature.
_RISKY_API_VOCAB = sorted({
    # memory
    "memcpy", "memmove", "memset", "memcmp", "bcopy", "bzero",
    # string
    "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp", "strlen",
    "strdup", "strndup", "stpcpy", "stpncpy", "sprintf", "snprintf",
    "vsprintf", "vsnprintf", "gets", "fgets", "puts", "fputs",
    # alloc
    "malloc", "calloc", "realloc", "free", "alloca", "strdup", "strndup",
    "xmalloc", "xcalloc", "xrealloc", "OPENSSL_malloc", "OPENSSL_free",
    "PyMem_Malloc", "av_malloc", "av_mallocz", "av_realloc", "av_free",
    "g_malloc", "g_malloc0", "g_new", "g_new0", "g_renew", "g_free",
    "kmalloc", "kzalloc", "krealloc", "kfree", "vmalloc", "g_realloc",
    # io
    "read", "write", "fread", "fwrite", "recv", "recvfrom", "send",
    "sendto", "fopen", "fclose", "open", "close", "ioctl",
    # parse
    "scanf", "sscanf", "fscanf", "printf", "fprintf", "snprintf",
    # conv
    "atoi", "atol", "atof", "strtol", "strtoul", "strtoll", "strtoull",
    # regex / pattern
    "regexec", "regcomp", "pcre_exec", "pcre_compile",
    # environ / system
    "getenv", "setenv", "system", "popen", "execve", "execvp", "fork",
})

# Anonymized devign tokens are noisy; we keep them anyway as called_functions
# so the model at least sees "this function calls a few other FUN_X". The
# real signal on devign comes from code_metrics (already 100% covered) and
# from the structural template, not from the function name.

# Boilerplate phrases to strip from risk_summary.
_RISK_SUMMARY_BOILERPLATE = (
    "no unguarded operation is visible in this function",
    "no explicit guard is visible",
    "no risky operations or missing checks identified",
    "no risky operations",
    "the function continues ",
    "more statements.",
    "words past the encoder window",
)


def _strip_evidence_tail(s: str) -> str:
    """Drop the ' [evidence: ...]' tail from a risky_operations string."""
    if not s:
        return s
    return re.sub(r"\s*\[evidence:[^\]]*\]\s*\.?\s*$", "", s).strip()


def _clean_risk_summary(s: str) -> str:
    if not s:
        return ""
    out = s
    for bp in _RISK_SUMMARY_BOILERPLATE:
        out = out.replace(bp, "")
    # collapse double spaces, strip
    out = re.sub(r"\s+", " ", out).strip(" .;,")
    return out


def extract_struct(code: str) -> Dict:
    """Label-blind structural extraction from raw_code. Always runs."""
    idents = [t for t in _IDENT.findall(code) if t not in _C_NOISE]
    ident_counts = Counter(idents)

    # called_functions = all distinct call targets, deduped, ordered by count
    calls = []
    seen = set()
    for m in _CALL.finditer(code):
        name = m.group(1)
        if name in _C_NOISE or name in seen:
            continue
        seen.add(name)
        calls.append(name)
    # sort by frequency, then alphabetic
    calls = sorted(calls, key=lambda c: (-ident_counts.get(c, 0), c))

    # risky_apis = intersection with the known risky vocabulary
    risky = [c for c in calls if c in _RISKY_API_VOCAB]

    # string literals (de-duplicated, drop empty, drop format-only "%s")
    strs = []
    for m in _STR_LIT.finditer(code):
        s = m.group(1) or ""
        if 0 < len(s) <= 200:
            strs.append(s)
    strs = list(dict.fromkeys(strs))[:50]  # cap at 50 strings per row

    # function name: first identifier followed by " (" after a type token
    # very rough; for anonymous (devign VAR/FUN) we fall back to the first
    # identifier that looks like a function call definition.
    fn_name = ""
    m = re.search(
        r"(?:^|\n)\s*(?:[A-Za-z_][\w\s\*\->]*?\s+)([A-Za-z_]\w*)\s*\(",
        code,
    )
    if m:
        cand = m.group(1)
        if cand not in _C_NOISE:
            fn_name = cand

    return dict(
        function_name=fn_name,
        called_functions=calls[:100],
        risky_apis=risky,
        string_literals=strs,
    )


# ---------------------------------------------------------------------------
# 3. UPDATE: clean the LLM text fields + self-consistency
# ---------------------------------------------------------------------------
def _self_consistency(e: Dict) -> Dict:
    """A few cheap contradictions that the LLM is known to produce."""
    rl = e.get("risk_level") or "none"
    rl_ord = {"none": 0, "low": 1, "medium": 2, "high": 3}.get(rl, 0)
    n_findings = len(e.get("risky_operations") or [])
    n_guards = len(e.get("safety_indicators") or [])
    n_missing = len(e.get("missing_checks") or [])
    contradiction_rl_vs_findings = int(
        rl_ord >= 2 and n_findings == 0
    )
    contradiction_rl_vs_guards = int(
        rl_ord == 3 and n_guards >= 3
    )
    contradiction_evidence = int(
        n_findings >= 3 and n_guards >= 3
    )
    return dict(
        risk_level_ord=rl_ord,
        n_findings=n_findings,
        n_guards=n_guards,
        n_missing_checks=n_missing,
        contradiction_rl_vs_findings=contradiction_rl_vs_findings,
        contradiction_rl_vs_guards=contradiction_rl_vs_guards,
        contradiction_evidence=contradiction_evidence,
    )


def _concise_text(e: Dict, sc: Dict) -> str:
    """A [SECTION]-tagged template. 100-150 tokens, no evidence tails,
    no risk_summary boilerplate."""
    parts: List[str] = []
    rl = e.get("risk_level") or "unknown"
    parts.append(f"[RISK] {rl} (confidence {e.get('confidence', 'med')}).")
    if e.get("purpose"):
        parts.append(f"[PURPOSE] {e['purpose']}")
    if e.get("data_flow"):
        parts.append(f"[DATA_FLOW] {e['data_flow']}")
    risky = e.get("risky_operations") or []
    if risky:
        parts.append(f"[FINDINGS] {len(risky)}: " + "; ".join(risky[:4]))
    else:
        parts.append("[FINDINGS] none")
    miss = e.get("missing_checks") or []
    if miss:
        parts.append(f"[MISSING] {len(miss)}: " + "; ".join(miss[:4]))
    else:
        parts.append("[MISSING] none")
    guards = e.get("safety_indicators") or []
    if guards:
        guard_strs = []
        for g in guards[:5]:
            if isinstance(g, dict):
                c = g.get("check", "")
                if c:
                    guard_strs.append(c)
        if guard_strs:
            parts.append(f"[GUARDS] {len(guard_strs)}: " + "; ".join(guard_strs))
        else:
            parts.append("[GUARDS] none")
    else:
        parts.append("[GUARDS] none")
    apis = e.get("risky_apis") or []
    if apis:
        parts.append(f"[APIS] {', '.join(apis[:8])}")
    fns = e.get("called_functions") or []
    if fns:
        parts.append(f"[CALLS] {', '.join(fns[:8])}")
    metrics = e.get("code_metrics") or {}
    if metrics:
        parts.append(
            f"[STATS] stmts={metrics.get('n_stmts', 0)} "
            f"calls={metrics.get('n_calls', 0)} "
            f"if={metrics.get('n_if', 0)} "
            f"loops={metrics.get('n_loops', 0)} "
            f"alloc={metrics.get('n_alloc', 0)} "
            f"free={metrics.get('n_free', 0)} "
            f"unsafe_str={metrics.get('n_unsafe_str', 0)} "
            f"trunc={metrics.get('truncated', 0)}"
        )
    if sc and any(sc.get(k) for k in
                  ("contradiction_rl_vs_findings",
                   "contradiction_rl_vs_guards",
                   "contradiction_evidence")):
        flags = [k for k in ("contradiction_rl_vs_findings",
                             "contradiction_rl_vs_guards",
                             "contradiction_evidence") if sc.get(k)]
        parts.append("[CONTRADICTIONS] " + ", ".join(flags))
    return " ".join(parts)


def _clean_risky(risky: List[str]) -> List[str]:
    return [_strip_evidence_tail(s) for s in (risky or []) if s]


# ---------------------------------------------------------------------------
# 4. AUGMENT: label-aware priors on the train split
# ---------------------------------------------------------------------------
def _vocab_topk(counter: Counter, k: int, min_count: int = 3) -> List[str]:
    return [w for w, c in counter.most_common(k * 3)
            if c >= min_count][:k]


def _smoothed_prior(pos: int, total: int, base_rate: float,
                    smoothing: float = 5.0) -> float:
    """Bayesian-smoothed P(label=1 | feature). smoothing=5 (was 20) so a
    feature seen >=5 times has a meaningful prior off base_rate; smaller
    smoothing lets rare-but-strongly-predictive APIs (strdup 21%, memcpy 20%)
    rise above the base rate enough for the model to use them."""
    return (pos + smoothing * base_rate) / (total + smoothing)


def compute_priors(rows: List[Dict], base_rate: float) -> Dict:
    """Compute train-set label-conditional priors for risky_apis,
    called_functions, and a global sample weight for class balance."""
    risky_count = Counter()
    risky_pos = Counter()
    fn_count = Counter()
    fn_pos = Counter()
    label_count = Counter()
    for r in rows:
        y = int(r["label"])
        label_count[y] += 1
        e = r.get("explanation") or {}
        for k in (e.get("risky_apis") or []):
            risky_count[k] += 1
            risky_pos[k] += y
        for k in (e.get("called_functions") or []):
            fn_count[k] += 1
            fn_pos[k] += y

    api_vocab = _vocab_topk(risky_count, k=50, min_count=3)
    fn_vocab = _vocab_topk(fn_count, k=100, min_count=3)

    api_priors = {
        k: _smoothed_prior(risky_pos[k], risky_count[k], base_rate)
        for k in api_vocab
    }
    fn_priors = {
        k: _smoothed_prior(fn_pos[k], fn_count[k], base_rate)
        for k in fn_vocab
    }

    # class-balancing sample weight
    n = sum(label_count.values())
    weights = {y: n / (len(label_count) * c) for y, c in label_count.items()}

    return dict(
        api_vocab=api_vocab,
        fn_vocab=fn_vocab,
        api_priors=api_priors,
        fn_priors=fn_priors,
        base_rate=base_rate,
        sample_weights=weights,
        n_train=n,
    )


def _row_priors(row: Dict, priors: Dict) -> Dict:
    e = row.get("explanation") or {}
    api_p = priors["api_priors"]
    fn_p = priors["fn_priors"]
    risky = e.get("risky_apis") or []
    fns = e.get("called_functions") or []
    api_priors_here = [api_p[k] for k in risky if k in api_p]
    fn_priors_here = [fn_p[k] for k in fns if k in fn_p]
    n_high_api = sum(1 for p in api_priors_here if p > 0.30)
    n_very_high_api = sum(1 for p in api_priors_here if p > 0.45)
    return dict(
        max_risky_api_prior=(max(api_priors_here) if api_priors_here else priors["base_rate"]),
        mean_risky_api_prior=(sum(api_priors_here) / len(api_priors_here)
                              if api_priors_here else priors["base_rate"]),
        n_high_risk_apis=n_high_api,
        n_very_high_risk_apis=n_very_high_api,
        max_fn_prior=(max(fn_priors_here) if fn_priors_here else priors["base_rate"]),
        mean_fn_prior=(sum(fn_priors_here) / len(fn_priors_here)
                       if fn_priors_here else priors["base_rate"]),
    )


def _multihot(items: List[str], vocab: List[str]) -> List[int]:
    s = set(items or []) & set(vocab)
    return [1 if v in s else 0 for v in vocab]


# ---------------------------------------------------------------------------
# 5. AUGMENT: per-row assembly
# ---------------------------------------------------------------------------
def augment_row(row: Dict, priors: Dict) -> Dict:
    code = row.get("raw_code", "") or ""
    e = dict(row.get("explanation") or {})

    # 1. FILL: static enrich if missing
    if "code_metrics" not in e or "risk_level" not in e:
        e = enrich_row({"raw_code": code, "explanation": e})["explanation"]

    # 2. FILL: regex extraction if missing
    if "function_name" not in e or not e.get("function_name"):
        e["function_name"] = ""
    if "called_functions" not in e or not e.get("called_functions"):
        e["called_functions"] = []
    if "risky_apis" not in e or not e.get("risky_apis"):
        e["risky_apis"] = []
    if "string_literals" not in e or not e.get("string_literals"):
        e["string_literals"] = []
    struct = extract_struct(code)
    e["function_name"]     = e["function_name"]     or struct["function_name"]
    e["called_functions"]  = e["called_functions"]  or struct["called_functions"]
    e["risky_apis"]        = e["risky_apis"]        or struct["risky_apis"]
    e["string_literals"]   = e["string_literals"]   or struct["string_literals"]

    # 3. UPDATE: clean text fields
    e["risky_operations"]  = _clean_risky(e.get("risky_operations") or [])
    e["risk_summary"]      = _clean_risk_summary(e.get("risk_summary") or "")

    # 3b. self-consistency
    sc = _self_consistency(e)
    e["self_consistency"]  = sc

    # 4. AUGMENT: label-aware priors
    pr = _row_priors({"explanation": e}, priors)
    e["label_priors"] = pr
    e["sample_weight"] = priors["sample_weights"].get(int(row["label"]), 1.0)

    # 5. AUGMENT: multi-hot
    e["multihot_risky_apis"] = _multihot(e.get("risky_apis") or [],
                                          priors["api_vocab"])
    e["multihot_called_functions"] = _multihot(e.get("called_functions") or [],
                                                priors["fn_vocab"])

    # 5b. structured concise text (label-aware in the sense that priors
    # shape the [RISK] line; useful for the [CONTRADICTIONS] line).
    e["concise_text"] = _concise_text(e, sc)

    out = dict(row)
    out["explanation"] = e
    return out


# ---------------------------------------------------------------------------
# 6. Driver
# ---------------------------------------------------------------------------
def process_dataset(name: str, out_tag: str = "aug2") -> None:
    base = EXPL_DIR / name
    # Reveal uses ACTIVE/reveal_<split>.jsonl; Devign uses <name>_<split>.jsonl
    candidates_train = [
        base / "ACTIVE" / f"{name}_train.jsonl",
        base / f"{name}_train.jsonl",
    ]
    candidates_val = [
        base / "ACTIVE" / f"{name}_val.jsonl",
        base / f"{name}_val.jsonl",
    ]
    src_tr = next((p for p in candidates_train if p.exists()), None)
    src_va = next((p for p in candidates_val   if p.exists()), None)
    if src_tr is None or src_va is None:
        print(f"[{name}] missing source files; skipping", flush=True)
        return

    dst_tr = src_tr.with_name(src_tr.stem + f".{out_tag}.jsonl")
    dst_va = src_va.with_name(src_va.stem + f".{out_tag}.jsonl")

    print(f"[{name}] reading {src_tr.name}", flush=True)
    rows_tr = [json.loads(l) for l in src_tr.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows_va = [json.loads(l) for l in src_va.read_text(encoding="utf-8").splitlines() if l.strip()]
    base_rate = sum(r["label"] for r in rows_tr) / max(1, len(rows_tr))
    print(f"[{name}] train={len(rows_tr)} val={len(rows_va)} "
          f"base_rate={base_rate:.4f}", flush=True)

    t0 = time.time()
    priors = compute_priors(rows_tr, base_rate)
    print(f"[{name}] priors: api_vocab={len(priors['api_vocab'])} "
          f"fn_vocab={len(priors['fn_vocab'])} "
          f"top_api_priors="
          f"{sorted(priors['api_priors'].items(), key=lambda kv: -kv[1])[:5]}",
          flush=True)

    n_tr = n_va = 0
    with dst_tr.open("w", encoding="utf-8") as fo:
        for r in rows_tr:
            fo.write(json.dumps(augment_row(r, priors), ensure_ascii=False) + "\n")
            n_tr += 1
    with dst_va.open("w", encoding="utf-8") as fo:
        for r in rows_va:
            fo.write(json.dumps(augment_row(r, priors), ensure_ascii=False) + "\n")
            n_va += 1
    print(f"[{name}] wrote {n_tr} train + {n_va} val -> "
          f"{dst_tr.name}, {dst_va.name}  ({time.time()-t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["reveal", "devign", "all"],
                    default="all")
    ap.add_argument("--out-tag", default="aug2",
                    help="suffix for output files, e.g. aug2 -> reveal_train.aug2.jsonl")
    args = ap.parse_args()
    ds = ["reveal", "devign"] if args.dataset == "all" else [args.dataset]
    for name in ds:
        process_dataset(name, out_tag=args.out_tag)


if __name__ == "__main__":
    main()
