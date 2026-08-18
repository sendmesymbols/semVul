"""The explanation prompt + JSON schema (single source of truth).

Replaces the old expl_v2/prompt_v2.py (anonymized code) and
expl_v2_pilot/prompt_v2_real.py (real identifiers). Both variants now live here
behind `mode`, because they only ever differed in the identifier policy and the
few-shot pair.

  mode="anon"  VARn/FUNn normalized code (Devign benchmark form). The model must
               NOT invent meaning for an anonymized name.
  mode="real"  real identifiers, API names and string literals (ReVeal). The model SHOULD exploit API
               knowledge -- that is signal a token encoder cannot recover.

SCHEMA <-> DATA COLUMNS
The schema emits exactly the explanation.* columns that the LLM owns in
explanations/SemanticVul/ACTIVE/{devign,reveal}/{train,val}.jsonl:

    purpose            str
    data_flow          str
    risky_operations   list[str]   "<pattern> [evidence: <verbatim>]"
    missing_checks     list[str]
    evidence_tokens    list[str]   verbatim fragments
    safety_indicators  list[{check, evidence}]
    risk_summary       str
    risk_level         str         NONE | LOW | MEDIUM | HIGH  (upper-case, as in data)

`confidence` is deliberately NOT in this schema. It is not a self-report: the
generator derives it from the decode-time token logprobs of the risk_level
verdict (see generate.py:probe_confidence). Asking the model to state a number
would make it a generated opinion, not an internal confidence measurement.

Legacy post-processing fields are outside the generator contract and are
rejected by validate_clean.py before final training.

The generator NEVER sees the ground-truth label.
"""
import json as _json

RISK_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH"]

# Ollama structured-output schema (passed as "format" to /api/chat).
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "purpose": {"type": "string"},
        "data_flow": {"type": "string"},
        "risky_operations": {
            "type": "array", "items": {"type": "string"}, "maxItems": 6,
        },
        "missing_checks": {
            "type": "array", "items": {"type": "string"}, "maxItems": 6,
        },
        "evidence_tokens": {
            "type": "array", "items": {"type": "string"}, "maxItems": 12,
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
        "risk_summary": {"type": "string"},
        "risk_level": {"type": "string", "enum": RISK_LEVELS},
    },
    # risk_level LAST so the verdict tokens are emitted after the evidence the
    # model just wrote -- the logprob probe then measures confidence in a
    # verdict that is conditioned on the whole explanation.
    "required": [
        "purpose", "data_flow", "risky_operations", "missing_checks",
        "evidence_tokens", "safety_indicators", "risk_summary", "risk_level",
    ],
}

# Field order used to normalize the parsed object (also the JSON key order the
# model is steered toward, which keeps the risk_level probe span at the end).
FIELD_ORDER = tuple(JSON_SCHEMA["required"])

_IDENTITY_ANON = """THE CODE IS NORMALIZED: user-defined identifiers appear as VAR1, VAR2, FUN1, FUN2, etc.
You cannot know what these names mean. NEVER invent a purpose or meaning for an
anonymized name. Base every judgment ONLY on what is visible: control flow, loop
bounds, arithmetic used in sizes or indices, allocation-and-copy patterns,
checked vs unchecked return values, array indexing, and any preserved
standard-library calls (memcpy, memset, snprintf, malloc, free, strlen, read, ...).

For purpose and data_flow, describe only the visible mechanics ("copies a
caller-supplied length into a fixed stack buffer"). Do not guess a domain role
from VARn/FUNn."""

_IDENTITY_REAL = """The code uses REAL identifiers, function names, and string literals. USE them:
infer what the code does from the names, and draw on your knowledge of the C
standard library and common APIs. Recognise well-known fallible or dangerous
calls by name (memcpy, memmove, strcpy, strcat, sprintf, snprintf, gets, scanf,
malloc, calloc, realloc, free, memset, alloca, read, recv, system, popen, ...)
and known weakness patterns (buffer overflow, integer overflow/underflow,
use-after-free, double free, NULL-pointer dereference, unchecked allocation,
format-string, command injection, off-by-one, unvalidated length/index).

Base every judgment on what is visible plus the documented semantics of the
named APIs. Do NOT hallucinate behaviour the code does not support."""

_PATTERNS_COMMON = """- copy or write into a fixed-size buffer without a prior length guard
- allocation result used without a null check
- size or index computed by arithmetic that could overflow or wrap
- loop bound that can step one past an array or buffer end
- return value of a fallible call ignored where failure matters
- pointer dereferenced on a path where it may be null or already freed
- length/offset taken from parameters or data and used without validation"""

_PATTERNS_REAL_EXTRA = "\n- use of a known-unsafe API where a bounded alternative exists"


def _system_prompt(mode: str) -> str:
    identity = _IDENTITY_REAL if mode == "real" else _IDENTITY_ANON
    patterns = _PATTERNS_COMMON + (_PATTERNS_REAL_EXTRA if mode == "real" else "")
    names_hint = ("naming the real functions and APIs it calls"
                  if mode == "real" else
                  "in terms of visible operations only")
    return f"""You are a static-analysis assistant reviewing one C function at a time.

{identity}

CALIBRATION RULES -- these override everything else:
1. Only report a risky operation if you can quote the EXACT code fragment as
   evidence, copied verbatim from the function.
2. If you cannot quote concrete evidence for any risk, then risky_operations and
   missing_checks MUST be empty arrays and risk_level MUST be "NONE".
   An empty finding is a correct, valued answer -- most ordinary code has no finding.
3. Only list a missing check if it pairs with a specific risky operation you
   reported. No generic wishes like "could add more validation".
4. Report checks that ARE present in safety_indicators, each with a verbatim
   quote. Look for these as actively as you look for risks.
5. NEVER reuse a quote from the worked examples above. Every fragment you put in
   evidence or evidence_tokens must appear character-for-character in the
   function you are analysing right now. If the pattern you have in mind is not
   present in THIS function, do not report it -- the examples show the output
   FORMAT, not findings to look for.

PATTERNS TO SCAN FOR (report only with quoted evidence):
{patterns}

OUTPUT FIELDS:
- purpose: one sentence on what the function does, {names_hint}.
- data_flow: one or two sentences tracing where the inputs go -- which
  parameters reach allocations, sizes, indices, copies or dereferences.
- risky_operations: each entry is a single string in exactly this form:
      <pattern> [evidence: <verbatim code fragment>]
  Empty array when nothing is quotable.
- missing_checks: specific absent guards, each tied to a reported risky operation.
- evidence_tokens: the verbatim code fragments you relied on, as plain strings.
- safety_indicators: guards that exist, each with check and verbatim evidence.
- risk_summary: 1-2 sentences referencing your evidence. Do NOT use the words
  "vulnerable", "vulnerability", "exploit", "CWE", "safe", or "secure" here;
  express the judgment through risk_level instead.
- risk_level: "NONE" | "LOW" | "MEDIUM" | "HIGH" -- your honest overall judgment,
  decided last, after the evidence above.

Answer with JSON only."""


# --- few-shot exemplars ----------------------------------------------------
# Pair 1 is a genuinely guarded function -> empty risk lists, risk_level NONE.
# Pair 2 has several concrete, quotable risks -> HIGH. Same two functions in
# both modes, once anonymized and once with real identifiers.

_SAFE_CODE_ANON = """static int FUN1 ( VAR1 * VAR2 , const char * VAR3 , int VAR4 ) {
 char VAR5 [ 64 ] ;
 if ( VAR3 == NULL || VAR4 <= 0 ) return - 1 ;
 if ( VAR4 >= ( int ) sizeof ( VAR5 ) ) return - 1 ;
 memcpy ( VAR5 , VAR3 , VAR4 ) ;
 VAR5 [ VAR4 ] = 0 ;
 VAR2 -> VAR6 = FUN2 ( VAR5 ) ;
 if ( VAR2 -> VAR6 == NULL ) return - 1 ;
 return 0 ;
 }"""

_SAFE_ANSWER_ANON = {
    "purpose": "Validates a caller-supplied pointer and length, copies that many bytes into a fixed 64-byte stack buffer, null-terminates it and passes it to FUN2.",
    "data_flow": "Parameter VAR3 and length VAR4 flow into memcpy into the local buffer VAR5 only after both are validated; VAR5 then flows into FUN2 and the result is stored in VAR2->VAR6.",
    "risky_operations": [],
    "missing_checks": [],
    "evidence_tokens": [
        "if ( VAR3 == NULL || VAR4 <= 0 ) return - 1 ;",
        "if ( VAR4 >= ( int ) sizeof ( VAR5 ) ) return - 1 ;",
        "memcpy ( VAR5 , VAR3 , VAR4 ) ;",
        "if ( VAR2 -> VAR6 == NULL ) return - 1 ;",
    ],
    "safety_indicators": [
        {"check": "null and non-positive length validation of inputs before use",
         "evidence": "if ( VAR3 == NULL || VAR4 <= 0 ) return - 1 ;"},
        {"check": "length checked against buffer capacity before memcpy",
         "evidence": "if ( VAR4 >= ( int ) sizeof ( VAR5 ) ) return - 1 ;"},
        {"check": "result of FUN2 checked for null before success return",
         "evidence": "if ( VAR2 -> VAR6 == NULL ) return - 1 ;"},
    ],
    "risk_summary": "Every write into the fixed buffer is preceded by an explicit length guard and both inputs are validated; no unguarded operation is visible.",
    "risk_level": "NONE",
}

_RISKY_CODE_ANON = """static int FUN1 ( VAR1 * VAR2 , int VAR3 ) {
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

_RISKY_ANSWER_ANON = {
    "purpose": "Computes a byte size from a caller-supplied count, heap-allocates that many bytes, copies into the new buffer and rewrites every element in place.",
    "data_flow": "Parameter VAR3 is multiplied by 4 into VAR6, which becomes both the malloc size and the memcpy length; VAR3 also bounds the rewrite loop, and the buffer is stored into VAR2->VAR9.",
    "risky_operations": [
        "allocation result used without a null check [evidence: VAR5 = malloc ( VAR6 ) ;]",
        "size computed by arithmetic that could overflow [evidence: int VAR6 = VAR3 * 4 ;]",
        "loop bound steps one past the buffer end [evidence: for ( int VAR8 = 0 ; VAR8 <= VAR3 ; VAR8 ++ )]",
    ],
    "missing_checks": [
        "null check on the malloc result before memcpy writes through it",
        "overflow or range check on VAR3 before computing VAR3 * 4",
        "loop condition should exclude index VAR3",
    ],
    "evidence_tokens": [
        "int VAR6 = VAR3 * 4 ;",
        "VAR5 = malloc ( VAR6 ) ;",
        "memcpy ( VAR5 , VAR2 -> VAR7 , VAR6 ) ;",
        "for ( int VAR8 = 0 ; VAR8 <= VAR3 ; VAR8 ++ )",
    ],
    "safety_indicators": [],
    "risk_summary": "An unchecked allocation is immediately written through, the allocation size comes from an unguarded multiplication, and the processing loop admits one out-of-range index.",
    "risk_level": "HIGH",
}

_SAFE_CODE_REAL = """static int copy_name ( struct conn_ctx * ctx , const char * src , int len ) {
 char name [ 64 ] ;
 if ( src == NULL || len <= 0 ) return - 1 ;
 if ( len >= ( int ) sizeof ( name ) ) return - 1 ;
 memcpy ( name , src , len ) ;
 name [ len ] = 0 ;
 ctx -> handle = registry_lookup ( name ) ;
 if ( ctx -> handle == NULL ) return - 1 ;
 return 0 ;
 }"""

_SAFE_ANSWER_REAL = {
    "purpose": "Copies a caller-supplied name into a fixed 64-byte stack buffer and resolves it through registry_lookup, storing the handle in the connection context.",
    "data_flow": "Parameters src and len reach memcpy into the local buffer name only after null and capacity validation; name then flows into registry_lookup and the result into ctx->handle.",
    "risky_operations": [],
    "missing_checks": [],
    "evidence_tokens": [
        "if ( src == NULL || len <= 0 ) return - 1 ;",
        "if ( len >= ( int ) sizeof ( name ) ) return - 1 ;",
        "memcpy ( name , src , len ) ;",
        "if ( ctx -> handle == NULL ) return - 1 ;",
    ],
    "safety_indicators": [
        {"check": "null and non-positive length validation before use",
         "evidence": "if ( src == NULL || len <= 0 ) return - 1 ;"},
        {"check": "length checked against buffer capacity before memcpy",
         "evidence": "if ( len >= ( int ) sizeof ( name ) ) return - 1 ;"},
        {"check": "registry_lookup result checked for null before success",
         "evidence": "if ( ctx -> handle == NULL ) return - 1 ;"},
    ],
    "risk_summary": "Every write into the fixed buffer is preceded by an explicit length guard and inputs are validated before use; no unguarded operation is visible.",
    "risk_level": "NONE",
}

_RISKY_CODE_REAL = """static int build_table ( struct parser * p , int count ) {
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

_RISKY_ANSWER_REAL = {
    "purpose": "Allocates an integer table sized from a caller-supplied count, copies the parser's raw bytes into it and decodes every entry in place.",
    "data_flow": "Parameter count is multiplied by 4 into size, which becomes both the malloc size and the memcpy length; count also bounds the decode loop, and the buffer is published into p->table.",
    "risky_operations": [
        "allocation result used without a null check [evidence: table = malloc ( size ) ;]",
        "size computed by arithmetic that could overflow [evidence: int size = count * 4 ;]",
        "loop bound steps one past the buffer end [evidence: for ( int i = 0 ; i <= count ; i ++ )]",
    ],
    "missing_checks": [
        "null check on the malloc result before memcpy writes through it",
        "overflow or range check on count before computing count * 4",
        "loop condition should exclude index count (use < not <=)",
    ],
    "evidence_tokens": [
        "int size = count * 4 ;",
        "table = malloc ( size ) ;",
        "memcpy ( table , p -> raw , size ) ;",
        "for ( int i = 0 ; i <= count ; i ++ )",
    ],
    "safety_indicators": [],
    "risk_summary": "An unchecked malloc result is immediately written through by memcpy, the allocation size comes from an unguarded multiplication, and the processing loop admits one out-of-range index.",
    "risk_level": "HIGH",
}

_FEWSHOTS = {
    "anon": ((_SAFE_CODE_ANON, _SAFE_ANSWER_ANON),
             (_RISKY_CODE_ANON, _RISKY_ANSWER_ANON)),
    "real": ((_SAFE_CODE_REAL, _SAFE_ANSWER_REAL),
             (_RISKY_CODE_REAL, _RISKY_ANSWER_REAL)),
}


def _fmt_user(code: str) -> str:
    return "Analyze this function:\n```c\n" + code + "\n```"


def _ordered(answer: dict) -> str:
    """Serialize a few-shot answer in FIELD_ORDER (risk_level last)."""
    return _json.dumps({k: answer[k] for k in FIELD_ORDER}, ensure_ascii=False)


def build_messages(code: str, mode: str = "anon") -> list:
    """Chat messages: system + 2 few-shot turns + the function under analysis."""
    if mode not in _FEWSHOTS:
        raise ValueError(f"mode must be 'anon' or 'real', got {mode!r}")
    (safe_code, safe_ans), (risky_code, risky_ans) = _FEWSHOTS[mode]
    return [
        {"role": "system", "content": _system_prompt(mode)},
        {"role": "user", "content": _fmt_user(safe_code)},
        {"role": "assistant", "content": _ordered(safe_ans)},
        {"role": "user", "content": _fmt_user(risky_code)},
        {"role": "assistant", "content": _ordered(risky_ans)},
        {"role": "user", "content": _fmt_user(code)},
    ]
