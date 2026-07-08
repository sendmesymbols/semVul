"""Prompt v2-REAL — explanation schema v2 for DE-ANONYMIZED (real-identifier) code.

Path C lever: on real code the model SHOULD exploit identifier/API/CWE knowledge
(the signal a token encoder can't recover) — the OPPOSITE of prompt_v2.py, which
forbids inventing meaning for VAR/FUN. Same JSON_SCHEMA as prompt_v2 so all
downstream parsing/quality-features are identical. The generator still NEVER sees
the label, and verdict words remain banned from risk_summary (leakage guard intact).
"""
import json as _json

# NOTE: reuses the SAME JSON_SCHEMA as prompt_v2 (imported by run_pilot directly);
# this module only supplies the real-code build_messages, so no schema import here.

SYSTEM_PROMPT = """You are a static-analysis assistant reviewing one C function at a time.

The code uses REAL identifiers, function names, and string literals. USE them:
infer what the code does from the names, and draw on your knowledge of the C
standard library and common APIs. Recognise well-known fallible or dangerous
calls by name (memcpy, memmove, strcpy, strcat, sprintf, snprintf, gets, scanf,
malloc, calloc, realloc, free, memset, alloca, read, recv, system, popen, ...)
and known weakness patterns (buffer overflow, integer overflow/underflow,
use-after-free, double free, NULL-pointer dereference, unchecked allocation,
format-string, command injection, off-by-one, unvalidated length/index).

Base every judgment on what is visible — control flow, arithmetic used in sizes
or indices, allocation-and-copy patterns, checked vs unchecked return values,
array indexing, and the documented semantics of the named APIs. Do NOT hallucinate
behaviour the code does not support.

CALIBRATION RULES — these override everything else:
1. Only report a risky operation if you can quote the EXACT code fragment as
   evidence, copied verbatim from the function.
2. If you cannot quote concrete evidence for any risk, then risky_operations and
   missing_checks MUST be empty arrays and risk_level MUST be "none". An empty
   finding is a correct, valued answer — most ordinary code has no finding.
3. Only list a missing check if it pairs with a specific risky operation you
   reported. No generic wishes like "could add more validation".
4. Report checks that ARE present in safety_indicators, each with a verbatim quote.

PATTERNS TO SCAN FOR (report only with quoted evidence):
- copy or write into a fixed-size buffer without a prior length guard
- allocation result used without a null check
- size or index computed by arithmetic that could overflow or wrap
- loop bound that can step one past an array or buffer end
- return value of a fallible call ignored where failure matters
- pointer dereferenced on a path where it may be null or already freed
- length/offset taken from parameters or data and used without validation
- use of a known-unsafe API where a bounded alternative exists

OUTPUT FIELDS:
- structural_observations: 1-5 short factual sentences about what the code does,
  naming the real functions/APIs it calls.
- safety_indicators: guards that exist, each with verbatim evidence.
- risky_operations: each with pattern (name the weakness pattern), verbatim
  evidence, and a one-line why (you MAY cite the named API's known behaviour).
- missing_checks: specific absent guards tied to reported risky operations.
- risk_level: "none" | "low" | "medium" | "high" — your honest overall judgment.
- confidence: how sure you are of that judgment.
- risk_summary: 1-2 sentences referencing your evidence. Do NOT use the words
  "vulnerable", "vulnerability", "exploit", "CWE", "safe", or "secure" here;
  express the judgment through risk_level instead.

Answer with JSON only."""

# Few-shot 1: a genuinely guarded function -> empty risk lists.
FEWSHOT_SAFE_CODE = """static int copy_name ( struct conn_ctx * ctx , const char * src , int len ) {
 char name [ 64 ] ;
 if ( src == NULL || len <= 0 ) return - 1 ;
 if ( len >= ( int ) sizeof ( name ) ) return - 1 ;
 memcpy ( name , src , len ) ;
 name [ len ] = 0 ;
 ctx -> handle = registry_lookup ( name ) ;
 if ( ctx -> handle == NULL ) return - 1 ;
 return 0 ;
 }"""

FEWSHOT_SAFE_ANSWER = {
    "structural_observations": [
        "Validates the src pointer and len, then memcpy's len bytes from src into a fixed 64-byte stack buffer name and null-terminates it.",
        "Calls registry_lookup on the buffer and stores the result in ctx->handle.",
        "Returns -1 on every failed validation path and 0 on success.",
    ],
    "safety_indicators": [
        {"check": "null and non-positive length validation before use",
         "evidence": "if ( src == NULL || len <= 0 ) return - 1 ;"},
        {"check": "length checked against buffer capacity before memcpy",
         "evidence": "if ( len >= ( int ) sizeof ( name ) ) return - 1 ;"},
        {"check": "registry_lookup result checked for null before success",
         "evidence": "if ( ctx -> handle == NULL ) return - 1 ;"},
    ],
    "risky_operations": [],
    "missing_checks": [],
    "risk_level": "none",
    "confidence": "high",
    "risk_summary": "Every write into the fixed buffer is preceded by an explicit length guard and inputs are validated before use; no unguarded operation is visible.",
}

# Few-shot 2: multiple concrete, quotable risks on real APIs -> high.
FEWSHOT_RISKY_CODE = """static int build_table ( struct parser * p , int count ) {
 int * table ;
 int size = count * 4 ;
 table = malloc ( size ) ;
 memcpy ( table , p -> raw , size ) ;
 for ( int i = 0 ; i <= count ; i ++ ) {
 table [ i ] = decode_entry ( table [ i ] ) ;
 }
 p -> table = table ;
 return size ;
 }"""

FEWSHOT_RISKY_ANSWER = {
    "structural_observations": [
        "Computes a byte size as count * 4, calls malloc for that many bytes, and memcpy's from p->raw into the new buffer.",
        "Iterates applying decode_entry element-wise, then stores the pointer in p->table.",
    ],
    "safety_indicators": [],
    "risky_operations": [
        {"pattern": "allocation result used without a null check",
         "evidence": "table = malloc ( size ) ;",
         "why": "malloc can return NULL and the next statement memcpy's through table, dereferencing null on failure"},
        {"pattern": "size computed by arithmetic that could overflow",
         "evidence": "int size = count * 4 ;",
         "why": "a large count overflows the signed multiplication, producing a short allocation followed by a full-size memcpy"},
        {"pattern": "loop bound steps one past the buffer end",
         "evidence": "for ( int i = 0 ; i <= count ; i ++ )",
         "why": "the buffer holds count ints but <= admits index count, one past the end"},
    ],
    "missing_checks": [
        "null check on the malloc result before memcpy writes through it",
        "overflow or range check on count before computing count * 4",
        "loop condition should exclude index count (use < not <=)",
    ],
    "risk_level": "high",
    "confidence": "high",
    "risk_summary": "An unchecked malloc result is immediately written through by memcpy, the allocation size comes from an unguarded multiplication, and the processing loop admits one out-of-range index.",
}


def _fmt_user(code: str) -> str:
    return "Analyze this function:\n```c\n" + code + "\n```"


def build_messages(code: str) -> list:
    """Chat messages: system + 2 real-identifier few-shot turns + the real function."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _fmt_user(FEWSHOT_SAFE_CODE)},
        {"role": "assistant", "content": _json.dumps(FEWSHOT_SAFE_ANSWER)},
        {"role": "user", "content": _fmt_user(FEWSHOT_RISKY_CODE)},
        {"role": "assistant", "content": _json.dumps(FEWSHOT_RISKY_ANSWER)},
        {"role": "user", "content": _fmt_user(code)},
    ]
