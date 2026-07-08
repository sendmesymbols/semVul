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


def compute(s: Sample) -> np.ndarray:
    e = s.explanation or {}
    m = e.get("code_metrics") or {}
    extra = [float(m.get(k, 0)) for k in _METRIC_KEYS] + [
        float(_LVL.get(e.get("risk_level"), 0)),
        float(_CONF.get(e.get("confidence"), 1)),
        float(len(e.get("safety_indicators") or [])),
        float(bool(e.get("tail_facts"))),
    ]
    return np.concatenate([compute_v1(s), np.asarray(extra, dtype=np.float32)])


def compute_batch(samples: List[Sample]) -> np.ndarray:
    return np.stack([compute(s) for s in samples], axis=0)
