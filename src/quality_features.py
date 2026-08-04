"""22 explanation-quality features. All derived from the JSON explanation and
raw code -- no ground-truth label used. See config.QUALITY_FEATURE_NAMES."""
import os
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


# RQ2 feature subsets (SEMVUL_QUAL_SET). Indices into the 22-vector built below,
# so the quality GATE reads only meaningful, label-free signals instead of the
# padded 22. Default (env unset) keeps all 22 -> ladder behavior unchanged.
#   A  grounding      : evidence_overlap_code(7), n_evidence_tokens(5)
#   B  + specificity  : n_risky_ops(3), n_missing_checks(4), has_missing_check_language(21)
#   C  + vuln vocab   : kw_memory..kw_concurrency(8-15)
# Dropped everywhere: length/verbosity (0,1,2,6,20) and code-API counts (16-19).
_QUAL_SET_IDX = {
    "A": [7, 5],
    "B": [7, 5, 3, 4, 21],
    "C": [7, 5, 3, 4, 21, 8, 9, 10, 11, 12, 13, 14, 15],
}


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
    arr = np.asarray(vals, dtype=np.float32)
    idx = _QUAL_SET_IDX.get((os.environ.get("SEMVUL_QUAL_SET") or "").upper())
    return arr[idx] if idx is not None else arr


def compute_batch(samples: List[Sample]) -> np.ndarray:
    return np.stack([compute(s) for s in samples], axis=0)


# ---------------------------------------------------------------------------
# RQ2 "rich" label-free quality set (SEMVUL_QUAL_RICH=1). Motivation: set B's
# evidence_overlap SATURATES (evidence_tokens are extracted FROM the code, so
# overlap ~= 1 for most samples -> no variance -> the gate can't condition on it).
# The same saturation kills any grounding built on called_functions/risky_apis
# (also code-derived). So this set deliberately AVOIDS overlap grounding and uses
# signals with genuine cross-sample variance, all derived from the explanation
# JSON + code_metrics (NO ground-truth label, NO self-reported confidence/risk
# ordinals -> stays defensibly label-free):
#   code complexity (code_metrics): loglen, branching, stmts
#   explanation specificity:        n_risky_ops, n_missing_checks, n_safety_ind,
#                                   n_called_funcs, n_risky_apis
#   explanation trust/verbosity:    hedge_density, expl_loglen, expl_code_ratio
_HEDGE = {"may", "might", "could", "possibly", "possible", "appears", "appear",
          "seems", "seem", "likely", "unclear", "potential", "potentially",
          "suggests", "suggest", "uncertain", "probably", "perhaps", "assume",
          "assumed", "unsure", "may be", "not clear"}

RICH_NAMES = ["code_loglen", "code_branching", "code_stmts", "n_risky_ops",
              "n_missing_checks", "n_safety_ind", "n_called_funcs", "n_risky_apis",
              "hedge_density", "expl_loglen", "expl_code_ratio"]


def compute_rich(s: Sample) -> np.ndarray:
    e = s.explanation or {}
    code = s.code or ""
    cm = e.get("code_metrics") or {}
    n_words = float(cm.get("n_words") or len(code.split()))
    n_stmts = float(cm.get("n_stmts") or 0)
    branching = float((cm.get("n_if") or 0) + (cm.get("n_loops") or 0)
                      + (cm.get("n_switch") or 0) + (cm.get("n_goto") or 0))
    ew = " ".join([_to_str(e.get("purpose")), _to_str(e.get("data_flow")),
                   _to_str(e.get("risk_summary"))]).lower()
    n_ew = float(len(ew.split()))
    hedge = float(sum(ew.count(h) for h in _HEDGE))
    vals = [
        float(np.log1p(n_words)),
        branching,
        n_stmts,
        float(len(_to_list(e.get("risky_operations")))),
        float(len(_to_list(e.get("missing_checks")))),
        float(len(_to_list(e.get("safety_indicators")))),
        float(len(_to_list(e.get("called_functions")))),
        float(len(_to_list(e.get("risky_apis")))),
        hedge / max(1.0, n_ew),
        float(np.log1p(n_ew)),
        n_ew / (n_words + 1.0),
    ]
    assert len(vals) == len(RICH_NAMES), f"{len(vals)} != {len(RICH_NAMES)}"
    return np.asarray(vals, dtype=np.float32)


def compute_batch_rich(samples: List[Sample]) -> np.ndarray:
    return np.stack([compute_rich(s) for s in samples], axis=0)
