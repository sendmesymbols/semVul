"""Deterministic, label-blind static enrichment of explanation JSON.

Why this exists (diagnosis 2026-07-08, scratchpad gap analysis):
  - Devign explanations carry ~zero label signal (TF-IDF train->val ROC 53.5;
    channel probe expl_only 54.8) because they were generated from anonymized
    VAR/FUN code and hallucinate semantics ("pointer arithmetic on buf").
  - On samples the code-only model gets WRONG, the explanation signal is
    ANTI-correlated (ROC 38 devign / 25 reveal) -- the fusion rungs can only
    learn to ignore the channel (Devign dROC +0.0) or get dragged down by it
    (Reveal dROC -1.9..-4.7, McNemar p<0.001).

What it does: extracts VERBATIM-GROUNDED findings, present guards, structural
metrics and -- crucially -- facts from the code region BEYOND the ~320-token
code-encoder window (33% of devign / 30% of reveal functions are truncated),
so the text channel finally carries information the code encoder cannot see.
Everything is derived from raw_code only; the ground-truth label is never read.

Field policy per row:
  - purpose / data_flow: kept as-is (weak but harmless).
  - risky_operations / missing_checks: static findings FIRST, then original
    items that pass a grounding filter (their identifiers/APIs appear in the
    code). Ungrounded hallucinations are dropped (originals preserved under
    explanation["llm_v1"]).
  - evidence_tokens: original (grounded subset) + verbatim finding evidence.
  - risk_summary: recomposed, calibrated, verdict-word-free.
  - NEW: safety_indicators[{check,evidence}], code_metrics{}, tail_facts str,
    risk_level none|low|medium|high, confidence, enrich="static-v1".
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ~where a 320-token GraphCodeBERT window ends, in whitespace words
ENCODER_WORD_BUDGET = 230

ALLOC_FNS = {"malloc", "calloc", "realloc", "alloca", "strdup", "strndup",
             "av_malloc", "av_mallocz", "av_realloc", "g_malloc", "g_malloc0",
             "g_new", "g_new0", "g_renew", "kmalloc", "kzalloc", "krealloc",
             "xmalloc", "vmalloc", "OPENSSL_malloc", "PyMem_Malloc"}
UNSAFE_STR = {"strcpy", "strcat", "sprintf", "vsprintf", "gets", "stpcpy"}
BOUNDED_COPY = {"memcpy", "memmove", "bcopy", "strncpy", "strncat", "snprintf",
                "memset"}
FALLIBLE = {"read", "recv", "recvfrom", "fread", "fgets", "write", "send",
            "fwrite", "snprintf", "ioctl"}
FREE_FNS = {"free", "av_free", "g_free", "kfree", "av_freep", "OPENSSL_free"}
DANGEROUS_ANY = ALLOC_FNS | UNSAFE_STR | BOUNDED_COPY | FALLIBLE | FREE_FNS

C_KEYWORDS = {"if", "else", "for", "while", "do", "switch", "case", "default",
              "return", "goto", "break", "continue", "sizeof", "struct",
              "union", "enum", "static", "const", "unsigned", "signed", "int",
              "char", "long", "short", "float", "double", "void", "NULL",
              "true", "false", "typedef", "extern", "register", "volatile",
              "inline", "restrict", "uint8_t", "uint16_t", "uint32_t",
              "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t", "size_t",
              "ssize_t", "bool"}

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _statements(code: str) -> List[Tuple[str, int]]:
    """Split token stream into statements; return (stmt_text, start_word_idx)."""
    toks = code.split()
    out, cur, start = [], [], 0
    for i, t in enumerate(toks):
        if not cur:
            start = i
        cur.append(t)
        if t in (";", "{", "}"):
            out.append((" ".join(cur), start))
            cur = []
    if cur:
        out.append((" ".join(cur), start))
    return out


def _quote(stmt: str, limit: int = 100) -> str:
    return stmt if len(stmt) <= limit else stmt[:limit].rsplit(" ", 1)[0] + " ..."


def _called(stmt: str, fns) -> List[str]:
    hits = []
    for f in fns:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(f)}\s*\(", stmt):
            hits.append(f)
    return hits


def _null_checked(var: str, window: List[str]) -> bool:
    pats = (f"if ( ! {var}", f"if ( {var} == NULL", f"if ( NULL == {var}",
            f"if ( {var} != NULL", f"if ( {var} == 0 )", f"assert ( {var}",
            f"if ( ! ( {var}", f"if ( {var} = = NULL")
    joined = " ".join(window)
    return any(p in joined for p in pats)


def analyze(code: str) -> Dict:
    """Label-blind static analysis of one function. Returns findings/guards/metrics."""
    stmts = _statements(code)
    n_words = len(code.split())
    findings: List[Dict] = []   # {pattern, evidence, why, word_idx, weight}
    guards: List[Dict] = []     # {check, evidence, word_idx}

    freed: List[Tuple[str, int]] = []   # (var, stmt_idx)

    for si, (stmt, widx) in enumerate(stmts):
        nxt = [s for s, _ in stmts[si + 1: si + 6]]
        prv = [s for s, _ in stmts[max(0, si - 4): si]]

        # --- allocations ---
        allocs = _called(stmt, ALLOC_FNS)
        if allocs:
            m = re.match(r"^\s*\*?\s*([A-Za-z_][\w\s\.\->\[\]]*?)\s*=\s*", stmt)
            var = None
            if m:
                var = m.group(1).split()[-1] if m.group(1).split() else None
            if var and not _null_checked(var, nxt) and f"if ( {var}" not in " ".join(nxt):
                findings.append(dict(
                    pattern="allocation result used without a null check",
                    evidence=_quote(stmt), why=f"{allocs[0]} result `{var}` is not "
                    "tested before use on the following paths", word_idx=widx, weight=2))
            elif var:
                guards.append(dict(check=f"allocation `{var}` is null-checked after "
                                   f"{allocs[0]}", evidence=_quote(stmt), word_idx=widx))
            # arithmetic inside the size argument
            arg = stmt.split("(", 1)[-1]
            if re.search(r"[A-Za-z0-9_\)]\s*[*+]\s*[A-Za-z0-9_\(]", arg) and "sizeof" not in arg.split("*")[0]:
                if not any("sizeof" in p and "if" in p for p in prv):
                    findings.append(dict(
                        pattern="size computed by unguarded arithmetic",
                        evidence=_quote(stmt), why="a large operand can wrap the "
                        "multiplication/addition and shrink the allocation",
                        word_idx=widx, weight=2))

        # --- inherently length-unchecked string APIs ---
        for f in _called(stmt, UNSAFE_STR):
            findings.append(dict(
                pattern=f"length-unchecked string operation ({f})",
                evidence=_quote(stmt), why=f"{f} writes without a size bound",
                word_idx=widx, weight=3))

        # --- bounded-copy family: is any bound/guard visible nearby? ---
        copies = _called(stmt, {"memcpy", "memmove", "bcopy"})
        if copies:
            ctx = " ".join(prv + [stmt])
            if "sizeof" in ctx or re.search(r"if\s*\([^)]*[<>]", ctx):
                guards.append(dict(check=f"{copies[0]} preceded by a visible size "
                                   "guard/sizeof", evidence=_quote(stmt), word_idx=widx))
            else:
                findings.append(dict(
                    pattern=f"copy without a visible length guard ({copies[0]})",
                    evidence=_quote(stmt), why="no sizeof or comparison guards the "
                    "copy length in the surrounding statements",
                    word_idx=widx, weight=2))

        # --- off-by-one loop bound feeding an index ---
        if stmt.startswith("for (") and "<=" in stmt:
            body = " ".join(s for s, _ in stmts[si + 1: si + 8])
            if "[" in body:
                findings.append(dict(
                    pattern="loop bound `<=` may step one past the buffer end",
                    evidence=_quote(stmt), why="inclusive bound with array indexing "
                    "in the loop body", word_idx=widx, weight=2))

        # --- discarded fallible results ---
        for f in _called(stmt, FALLIBLE):
            if re.match(rf"^\s*{re.escape(f)}\s*\(", stmt):  # statement IS the bare call
                findings.append(dict(
                    pattern=f"return value of fallible call ignored ({f})",
                    evidence=_quote(stmt), why="failure is silently ignored",
                    word_idx=widx, weight=1))

        # --- free tracking for use-after-free ---
        for f in _called(stmt, FREE_FNS):
            m = re.search(rf"{re.escape(f)}\s*\(\s*([A-Za-z_][\w]*)", stmt)
            if m:
                freed.append((m.group(1), si))

        # --- generic guards worth crediting ---
        if re.match(r"^\s*if \( ", stmt):
            if "== NULL" in stmt or stmt.startswith("if ( !"):
                guards.append(dict(check="explicit null check",
                                   evidence=_quote(stmt), word_idx=widx))
            elif "sizeof" in stmt:
                guards.append(dict(check="capacity check against sizeof",
                                   evidence=_quote(stmt), word_idx=widx))

    # use-after-free / double-free (very conservative: same bare identifier)
    for var, si in freed:
        for stmt, widx in stmts[si + 1:]:
            if re.search(rf"(?<![\w]){re.escape(var)}\s*(->|\[)", stmt) and \
               not re.search(rf"(?<![\w]){re.escape(var)}\s*=", stmt):
                findings.append(dict(
                    pattern="pointer used after being freed",
                    evidence=_quote(stmt), why=f"`{var}` was freed earlier and is "
                    "dereferenced without reassignment", word_idx=widx, weight=3))
                break
            if re.search(rf"(?<![\w]){re.escape(var)}\s*=", stmt):
                break

    # ---- metrics ----
    lc = code.lower()
    metrics = dict(
        n_words=n_words,
        n_stmts=len(stmts),
        n_if=len(re.findall(r"(?<![\w])if \(", code)),
        n_loops=len(re.findall(r"(?<![\w])(for|while) \(", code)),
        n_switch=len(re.findall(r"(?<![\w])switch \(", code)),
        n_goto=len(re.findall(r"(?<![\w])goto ", code)),
        n_return=len(re.findall(r"(?<![\w])return[ ;]", code)),
        n_calls=len(re.findall(r"[A-Za-z_][\w]* \(", code)),
        n_deref=code.count("->"),
        n_index=code.count("["),
        n_alloc=sum(lc.count(f + " (") for f in ("malloc", "calloc", "realloc")),
        n_free=sum(lc.count(f + " (") for f in ("free",)),
        n_unsafe_str=sum(lc.count(f + " (") for f in UNSAFE_STR),
        n_bounded_copy=sum(lc.count(f + " (") for f in BOUNDED_COPY),
        truncated=int(n_words > ENCODER_WORD_BUDGET),
        n_findings=len(findings),
        n_guards=len(guards),
        n_findings_tail=sum(1 for f in findings if f["word_idx"] > ENCODER_WORD_BUDGET),
    )

    # ---- tail facts: what the code encoder cannot see ----
    tail_facts = ""
    if n_words > ENCODER_WORD_BUDGET:
        tail_stmts = [(s, w) for s, w in stmts if w > ENCODER_WORD_BUDGET]
        tail_txt = " ".join(s for s, _ in tail_stmts)
        apis = sorted({f for f in DANGEROUS_ANY
                       if re.search(rf"(?<![\w]){re.escape(f)}\s*\(", tail_txt)})
        tail_finds = [f for f in findings if f["word_idx"] > ENCODER_WORD_BUDGET]
        bits = [f"the function continues ~{n_words - ENCODER_WORD_BUDGET} words past "
                f"the encoder window with {len(tail_stmts)} more statements"]
        if apis:
            bits.append("tail calls: " + ", ".join(apis))
        for f in tail_finds[:3]:
            bits.append(f"tail finding: {f['pattern']} [{f['evidence']}]")
        tail_facts = "; ".join(bits) + "."

    return dict(findings=findings, guards=guards, metrics=metrics,
                tail_facts=tail_facts)


def _grounded(item: str, code: str) -> bool:
    """Keep an original LLM claim only if it names something visible in the code."""
    idents = [t for t in _IDENT.findall(item)
              if t not in C_KEYWORDS and len(t) > 2]
    if not idents:
        return True  # purely generic phrasing; harmless
    hits = sum(1 for t in idents if t in code)
    return hits >= max(1, len(idents) // 2)


def _risk_level(findings: List[Dict]) -> str:
    if not findings:
        return "none"
    w = sum(f["weight"] for f in findings)
    if w >= 6:
        return "high"
    if w >= 3:
        return "medium"
    return "low"


def _summary(findings, guards, tail_facts, metrics) -> str:
    if not findings and not guards:
        s = "No unguarded operation and no explicit guard is visible in this function."
    elif not findings:
        s = (f"{len(guards)} explicit guard(s) present "
             f"({'; '.join(g['check'] for g in guards[:3])}) and no unguarded "
             "operation is visible.")
    else:
        pats = "; ".join(f["pattern"] for f in findings[:4])
        s = (f"{len(findings)} unguarded operation(s) visible: {pats}. "
             f"{len(guards)} guard(s) present.")
    if tail_facts:
        s += " " + tail_facts
    return s


def enrich_row(row: dict) -> dict:
    """Enrich one JSONL row in place-compatible copy. Never reads row['label']."""
    code = row.get("raw_code", "") or ""
    e = dict(row.get("explanation") or {})
    ana = analyze(code)
    findings, guards = ana["findings"], ana["guards"]

    orig_risky = e.get("risky_operations") or []
    orig_missing = e.get("missing_checks") or []
    orig_evidence = e.get("evidence_tokens") or []
    if not isinstance(orig_risky, list):
        orig_risky = [str(orig_risky)]
    if not isinstance(orig_missing, list):
        orig_missing = [str(orig_missing)]
    if not isinstance(orig_evidence, list):
        orig_evidence = [str(orig_evidence)]

    keep_risky = [str(x) for x in orig_risky if _grounded(str(x), code)]
    keep_missing = [str(x) for x in orig_missing if _grounded(str(x), code)]
    keep_evidence = [str(x) for x in orig_evidence if _grounded(str(x), code)]

    static_risky = [f"{f['pattern']} [evidence: {f['evidence']}]" for f in findings]
    static_missing = [f["why"] for f in findings if f["weight"] >= 2]
    static_evidence = [f["evidence"] for f in findings] + \
                      [g["evidence"] for g in guards]

    new_e = dict(e)
    new_e["llm_v1"] = dict(risky_operations=orig_risky,
                           missing_checks=orig_missing,
                           risk_summary=e.get("risk_summary", ""))
    new_e["risky_operations"] = static_risky + [x for x in keep_risky
                                                if x not in static_risky][:4]
    new_e["missing_checks"] = static_missing + [x for x in keep_missing
                                                if x not in static_missing][:4]
    seen = set()
    ev = []
    for x in static_evidence + keep_evidence:
        if x not in seen:
            seen.add(x)
            ev.append(x)
    new_e["evidence_tokens"] = ev[:12]
    new_e["safety_indicators"] = [dict(check=g["check"], evidence=g["evidence"])
                                  for g in guards[:6]]
    new_e["code_metrics"] = ana["metrics"]
    new_e["tail_facts"] = ana["tail_facts"]
    new_e["risk_level"] = _risk_level(findings)
    new_e["confidence"] = "high" if (findings or guards) else "medium"
    new_e["risk_summary"] = _summary(findings, guards, ana["tail_facts"],
                                     ana["metrics"])
    new_e["enrich"] = "static-v1"

    out = dict(row)
    out["explanation"] = new_e
    return out


def enriched_text(row_or_expl) -> str:
    """Render the enriched explanation to the text the RoBERTa channel sees."""
    e = row_or_expl.get("explanation", row_or_expl) if isinstance(row_or_expl, dict) else {}
    parts = [
        str(e.get("purpose") or ""),
        str(e.get("data_flow") or ""),
        f"overall risk level: {e.get('risk_level', 'unknown')}.",
        " ".join(str(x) for x in (e.get("risky_operations") or [])),
        " ".join(str(x) for x in (e.get("missing_checks") or [])),
        " ".join(f"guard present: {g.get('check', '')} [{g.get('evidence', '')}]"
                 for g in (e.get("safety_indicators") or [])),
        str(e.get("tail_facts") or ""),
        str(e.get("risk_summary") or ""),
    ]
    return " ".join(p for p in parts if p).strip()
