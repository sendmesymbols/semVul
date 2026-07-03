"""22 explanation-quality features. All derived from the JSON explanation and
raw code -- no ground-truth label used. See config.QUALITY_FEATURE_NAMES."""
import re
from typing import List

import numpy as np

from src.config import QUALITY_FEATURE_NAMES
from src.data_io import Sample, _to_list, _to_str

_MEMORY   = {"memory", "buffer", "heap", "stack", "overflow", "underflow", "leak"}
_POINTER  = {"pointer", "->", "dereference", "deref"}
_BOUNDS   = {"bounds", "boundary", "length check", "size check", "range check", "off-by-one"}
_VALID    = {"validation", "sanitiz", "unchecked", "unsanitized", "validated"}
_INTEGER  = {"integer", "signed", "unsigned", "wrap", "truncat"}
_INPUT    = {"input", "user input", "external", "untrusted", "attacker", "user-supplied"}
_NULL     = {"null", "nullptr", "nil"}
_CONCUR   = {"race", "concurrent", "thread", "lock", "mutex", "atomic"}

_API_MEM   = {"memcpy", "memmove", "memset", "bcopy"}
_API_STR   = {"strcpy", "strcat", "sprintf", "gets", "scanf", "strncpy", "strncat"}
_API_ALLOC = {"malloc", "calloc", "realloc", "free", "alloca", "new", "delete"}
_API_IO    = {"read", "write", "recv", "send", "open", "fopen", "system", "exec", "popen"}


def _text_of(s: Sample) -> str:
    e = s.explanation or {}
    return " ".join([
        _to_str(e.get("purpose")),
        _to_str(e.get("data_flow")),
        " ".join(_to_list(e.get("risky_operations"))),
        " ".join(_to_list(e.get("missing_checks"))),
        _to_str(e.get("risk_summary")),
    ]).lower()


def _count(text: str, terms) -> int:
    return sum(text.count(t) for t in terms)


def _tok(text: str) -> set:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text.lower()))


def compute(s: Sample) -> np.ndarray:
    e = s.explanation or {}
    text = _text_of(s)
    code = s.code or ""

    purpose      = _to_str(e.get("purpose"))
    dataflow     = _to_str(e.get("data_flow"))
    risk_summary = _to_str(e.get("risk_summary"))
    risky_ops    = _to_list(e.get("risky_operations"))
    missing      = _to_list(e.get("missing_checks"))
    evidence     = _to_list(e.get("evidence_tokens"))

    evid_tokens = _tok(" ".join(evidence))
    code_tokens = _tok(code)
    overlap = (len(evid_tokens & code_tokens) / max(1, len(evid_tokens))) if evid_tokens else 0.0

    n_sent_dataflow = max(1, dataflow.count(".") + dataflow.count(";")) if dataflow else 0
    has_missing_lang = 1 if (missing or "missing" in text or "unchecked" in text) else 0

    vals = [
        len(purpose.split()),
        len(dataflow.split()),
        len(risk_summary.split()),
        len(risky_ops),
        len(missing),
        len(evidence),
        sum(len(t) for t in evidence),
        overlap,
        _count(text, _MEMORY), _count(text, _POINTER),
        _count(text, _BOUNDS), _count(text, _VALID),
        _count(text, _INTEGER), _count(text, _INPUT),
        _count(text, _NULL),    _count(text, _CONCUR),
        _count(code.lower(), _API_MEM),
        _count(code.lower(), _API_STR),
        _count(code.lower(), _API_ALLOC),
        _count(code.lower(), _API_IO),
        n_sent_dataflow, has_missing_lang,
    ]
    assert len(vals) == len(QUALITY_FEATURE_NAMES), \
        f"{len(vals)} != {len(QUALITY_FEATURE_NAMES)}"
    return np.asarray(vals, dtype=np.float32)


def compute_batch(samples: List[Sample]) -> np.ndarray:
    return np.stack([compute(s) for s in samples], axis=0)
