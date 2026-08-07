"""22 v1 quality features + 22 static-enrichment features (44 total).

The extra block reads explanation.code_metrics / risk_level / confidence /
safety_indicators / tail_facts written by experiments/expl_enrich (static-v1).
Rows without enrichment get zeros for the new block, so mixed data stays
shape-stable. Label-blind by construction.
"""
from typing import List

import numpy as np

from src.data_io import Sample
from src.quality_features import compute as compute_v1

V2_EXTRA_NAMES = [
    "m_n_words", "m_n_stmts", "m_n_if", "m_n_loops", "m_n_switch", "m_n_goto",
    "m_n_return", "m_n_calls", "m_n_deref", "m_n_index", "m_n_alloc",
    "m_n_free", "m_n_unsafe_str", "m_n_bounded_copy", "m_truncated",
    "m_n_findings", "m_n_guards", "m_n_findings_tail",
    "risk_level_ord", "confidence_ord", "n_safety_indicators", "has_tail_facts",
]

_METRIC_KEYS = ["n_words", "n_stmts", "n_if", "n_loops", "n_switch", "n_goto",
                "n_return", "n_calls", "n_deref", "n_index", "n_alloc",
                "n_free", "n_unsafe_str", "n_bounded_copy", "truncated",
                "n_findings", "n_guards", "n_findings_tail"]
_LVL = {"none": 0, "low": 1, "medium": 2, "high": 3}
_CONF = {"low": 0, "medium": 1, "high": 2}


def _risk_level_ord(v) -> float:
    """Case-insensitive risk_level -> 0..3.

    static_enrich writes lower-case ("high"); the ACTIVE JSONLs carry upper-case
    ("HIGH") and generate.py emits upper-case to match them. A case-sensitive
    lookup silently returned 0 for every upper-case row, i.e. the feature was
    dead on exactly the data in use.
    """
    return float(_LVL.get(str(v).strip().lower(), 0))


def _confidence_ord(v) -> float:
    """Support both legacy low|medium|high strings and current numeric 0..100."""
    if isinstance(v, (int, float)):
        return float(np.clip(float(v), 0.0, 100.0) / 50.0)
    try:
        return float(np.clip(float(v), 0.0, 100.0) / 50.0)
    except (TypeError, ValueError):
        return float(_CONF.get(str(v).strip().lower(), 1))


def compute(s: Sample) -> np.ndarray:
    e = s.explanation or {}
    m = e.get("code_metrics") or {}
    extra = [float(m.get(k, 0)) for k in _METRIC_KEYS] + [
        _risk_level_ord(e.get("risk_level")),
        _confidence_ord(e.get("confidence")),
        float(len(e.get("safety_indicators") or [])),
        float(bool(e.get("tail_facts"))),
    ]
    return np.concatenate([compute_v1(s), np.asarray(extra, dtype=np.float32)])


def compute_batch(samples: List[Sample]) -> np.ndarray:
    return np.stack([compute(s) for s in samples], axis=0)
