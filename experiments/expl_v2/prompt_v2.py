"""Prompt + schema v2 for explanation regeneration.

Design (2026-07-04, diagnosis in memory/semvul-fusevul-status.md):
v1 explanations were near-constant across classes (signal AUC ~0.51) because
the schema forced risky_operations/missing_checks for every sample and the
generator hallucinated semantics for anonymized VAR/FUN identifiers.

v2 changes:
  - "safe" is a first-class answer: empty risk lists + risk_level "none".
  - every risky_operation must quote verbatim evidence (post-filtered later).
  - safety_indicators: checks that ARE present (contrastive text signal).
  - purpose/data_flow removed (hallucination magnets on anonymized code);
    replaced by structural_observations (visible facts only).
  - risk_level/confidence are categorical, routed to quality features, and
    verdict words are banned from risk_summary (leakage story intact).

The generator NEVER sees the ground-truth label.
"""

RISK_LEVELS = ["none", "low", "medium", "high"]
CONFIDENCE_LEVELS = ["low", "medium", "high"]

# Ollama structured-output schema (passed as "format" in /api/chat).
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "structural_observations": {
            "type": "array", "items": {"type": "string"}, "maxItems": 6,
        },
        "safety_indicators": {
            "type": "array", "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "check": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["check", "evidence"],
            },
        },
        "risky_operations": {
            "type": "array", "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "evidence": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["pattern", "evidence", "why"],
            },
        },
        "missing_checks": {
            "type": "array", "items": {"type": "string"}, "maxItems": 6,
        },
        "risk_level": {"type": "string", "enum": RISK_LEVELS},
        "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
        "risk_summary": {"type": "string"},
    },
    "required": [
        "structural_observations", "safety_indicators", "risky_operations",
        "missing_checks", "risk_level", "confidence", "risk_summary",
    ],
}

SYSTEM_PROMPT = """You are a static-analysis assistant reviewing one C function at a time.

THE CODE IS NORMALIZED: user-defined identifiers appear as VAR1, VAR2, FUN1, FUN2, etc.
You cannot know what these names mean. NEVER invent a purpose or meaning for an
anonymized name. Base every judgment ONLY on what is visible:
control flow, loop bounds, arithmetic used in sizes or indices, allocation-and-copy
patterns, checked vs unchecked return values, array indexing, and any preserved
standard-library calls (memcpy, memset, snprintf, malloc, free, strlen, read, ...).

CALIBRATION RULES — these override everything else:
1. Only report a risky operation if you can quote the EXACT code fragment as evidence,
   copied verbatim from the function.
2. If you cannot quote concrete evidence for any risk, then risky_operations and
   missing_checks MUST be empty arrays and risk_level MUST be "none".
   An empty finding is a correct, valued answer — most ordinary code has no finding.
3. Only list a missing check if it pairs with a specific risky operation you reported
   (e.g. an unchecked malloc that is then written through). Do not list generic
   wishes like "could add more validation".
4. Report checks that ARE present in safety_indicators, each with a verbatim quote.
   Look for these as actively as you look for risks.

PATTERNS TO SCAN FOR (report only with quoted evidence):
- copy or write into a fixed-size buffer without a prior length guard
- allocation result used without a null check
- size or index computed by arithmetic that could overflow or wrap
- loop bound that can step one past an array or buffer end
- return value of a fallible call ignored where failure matters
- pointer dereferenced on a path where it may be null or already freed
- length/offset taken from parameters or data and used without validation

OUTPUT FIELDS:
- structural_observations: 1-5 short factual sentences about what the code visibly
  does (allocates, copies, loops, branches). Facts only, no speculation.
- safety_indicators: guards that exist, each with verbatim evidence.
- risky_operations: each with pattern, verbatim evidence, and one-line why.
- missing_checks: specific absent guards tied to reported risky operations.
- risk_level: "none" | "low" | "medium" | "high" — your honest overall judgment.
- confidence: how sure you are of that judgment given the anonymization.
- risk_summary: 1-2 sentences referencing your evidence. Do NOT use the words
  "vulnerable", "vulnerability", "exploit", "CWE", "safe", or "secure" here;
  express the judgment through risk_level instead.

Answer with JSON only."""

# Few-shot exemplar 1: a genuinely guarded function -> empty risk lists.
FEWSHOT_SAFE_CODE = """static int FUN1 ( VAR1 * VAR2 , const char * VAR3 , int VAR4 ) {
 char VAR5 [ 64 ] ;
 if ( VAR3 == NULL || VAR4 <= 0 ) return - 1 ;
 if ( VAR4 >= ( int ) sizeof ( VAR5 ) ) return - 1 ;
 memcpy ( VAR5 , VAR3 , VAR4 ) ;
 VAR5 [ VAR4 ] = 0 ;
 VAR2 -> VAR6 = FUN2 ( VAR5 ) ;
 if ( VAR2 -> VAR6 == NULL ) return - 1 ;
 return 0 ;
 }"""

FEWSHOT_SAFE_ANSWER = {
    "structural_observations": [
        "Copies VAR4 bytes from parameter VAR3 into a fixed 64-byte stack buffer VAR5, then null-terminates it.",
        "Calls FUN2 on the buffer and stores the result in a struct field.",
        "Returns -1 on every failed validation path and 0 on success.",
    ],
    "safety_indicators": [
        {"check": "null and non-positive length validation of inputs before use",
         "evidence": "if ( VAR3 == NULL || VAR4 <= 0 ) return - 1 ;"},
        {"check": "length checked against buffer capacity before memcpy",
         "evidence": "if ( VAR4 >= ( int ) sizeof ( VAR5 ) ) return - 1 ;"},
        {"check": "result of FUN2 checked for null before success return",
         "evidence": "if ( VAR2 -> VAR6 == NULL ) return - 1 ;"},
    ],
    "risky_operations": [],
    "missing_checks": [],
    "risk_level": "none",
    "confidence": "high",
    "risk_summary": "Every write into the fixed buffer is preceded by an explicit length guard and inputs are validated; no unguarded operation is visible.",
}

# Few-shot exemplar 2: multiple concrete, quotable risks -> high.
FEWSHOT_RISKY_CODE = """static int FUN1 ( VAR1 * VAR2 , int VAR3 ) {
 VAR4 * VAR5 ;
 int VAR6 = VAR3 * 4 ;
 VAR5 = malloc ( VAR6 ) ;
 memcpy ( VAR5 , VAR2 -> VAR7 , VAR6 ) ;
 for ( int VAR8 = 0 ; VAR8 <= VAR3 ; VAR8 ++ ) {
 VAR5 [ VAR8 ] = FUN2 ( VAR5 [ VAR8 ] ) ;
 }
 VAR2 -> VAR9 = VAR5 ;
 return VAR6 ;
 }"""

FEWSHOT_RISKY_ANSWER = {
    "structural_observations": [
        "Computes a byte size as VAR3 * 4, heap-allocates that many bytes, and copies from a struct field into the new buffer.",
        "Iterates over the buffer applying FUN2 element-wise, then stores the pointer in the struct.",
    ],
    "safety_indicators": [],
    "risky_operations": [
        {"pattern": "allocation result used without a null check",
         "evidence": "VAR5 = malloc ( VAR6 ) ;",
         "why": "the very next statement writes through VAR5 via memcpy, so a failed allocation dereferences null"},
        {"pattern": "size computed by arithmetic that could overflow",
         "evidence": "int VAR6 = VAR3 * 4 ;",
         "why": "a large VAR3 wraps the signed multiplication, producing a short allocation followed by a full-size copy"},
        {"pattern": "loop bound steps one past the buffer end",
         "evidence": "for ( int VAR8 = 0 ; VAR8 <= VAR3 ; VAR8 ++ )",
         "why": "the buffer holds VAR3 elements but <= admits index VAR3"},
    ],
    "missing_checks": [
        "null check on the malloc result before memcpy writes through it",
        "overflow or range check on VAR3 before computing VAR3 * 4",
        "loop condition should exclude index VAR3",
    ],
    "risk_level": "high",
    "confidence": "high",
    "risk_summary": "An unchecked allocation is immediately written through, the allocation size comes from unguarded multiplication, and the processing loop admits one out-of-range index.",
}


def _fmt_user(code: str) -> str:
    return "Analyze this function:\n```c\n" + code + "\n```"


def build_messages(code: str) -> list:
    """Chat messages: system + 2 few-shot turns + the real function."""
    import json as _json
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _fmt_user(FEWSHOT_SAFE_CODE)},
        {"role": "assistant", "content": _json.dumps(FEWSHOT_SAFE_ANSWER)},
        {"role": "user", "content": _fmt_user(FEWSHOT_RISKY_CODE)},
        {"role": "assistant", "content": _json.dumps(FEWSHOT_RISKY_ANSWER)},
        {"role": "user", "content": _fmt_user(code)},
    ]
