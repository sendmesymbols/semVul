"""Metrics and threshold policies.

We ALWAYS report at all three policies so the trade-offs are visible:
  - fixed 0.5           : the paper's natural policy (esp. on balanced data)
  - max-balanced-acc    : honest, class-imbalance-aware
  - max-F1              : the FuSEVul-comparable F1 headline
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)


@dataclass
class Metrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float

    def as_dict(self) -> Dict[str, float]:
        return {k: float(round(v, 4)) for k, v in self.__dict__.items()}


def _at(threshold: float, probs: np.ndarray, y: np.ndarray) -> Metrics:
    yhat = (probs >= threshold).astype(np.int64)
    return Metrics(
        threshold=float(threshold),
        accuracy=100 * accuracy_score(y, yhat),
        precision=100 * precision_score(y, yhat, zero_division=0),
        recall=100 * recall_score(y, yhat, zero_division=0),
        f1=100 * f1_score(y, yhat, zero_division=0),
        pr_auc=100 * average_precision_score(y, probs),
        roc_auc=100 * roc_auc_score(y, probs),
    )


def _thresh_max(probs: np.ndarray, y: np.ndarray, objective: str) -> float:
    """objective in {'f1', 'balanced_acc'}"""
    grid = np.linspace(0.05, 0.95, 91)
    best, best_score = 0.5, -1.0
    for t in grid:
        yhat = (probs >= t).astype(np.int64)
        if objective == "f1":
            s = f1_score(y, yhat, zero_division=0)
        else:
            tp = ((yhat == 1) & (y == 1)).sum()
            tn = ((yhat == 0) & (y == 0)).sum()
            fp = ((yhat == 1) & (y == 0)).sum()
            fn = ((yhat == 0) & (y == 1)).sum()
            sens = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            s = 0.5 * (sens + spec)
        if s > best_score:
            best_score, best = s, float(t)
    return best


def report(probs_val: np.ndarray, y_val: np.ndarray,
           probs_tune: np.ndarray = None, y_tune: np.ndarray = None
           ) -> Dict[str, Metrics]:
    """Return metrics at each threshold policy. Thresholds selected on TUNE
    when provided, else on VAL (self-tuned) with an honest note in the caller."""
    src_p = probs_tune if probs_tune is not None else probs_val
    src_y = y_tune     if y_tune     is not None else y_val

    t_f1  = _thresh_max(src_p, src_y, "f1")
    t_bal = _thresh_max(src_p, src_y, "balanced_acc")
    return {
        "fixed_050":     _at(0.50, probs_val, y_val),
        "max_bal_acc":   _at(t_bal, probs_val, y_val),
        "max_f1":        _at(t_f1,  probs_val, y_val),
    }


def compare_to_paper(m: Metrics, paper: Dict[str, float]) -> Dict[str, str]:
    """Symbol per metric: WIN / LOSE / TIE / (na)."""
    def mark(ours, target):
        if target is None:
            return "(n/a)"
        delta = ours - target
        if delta > 0.05:  return f"WIN +{delta:.2f}"
        if delta < -0.05: return f"LOSE {delta:.2f}"
        return "TIE"
    return {
        "accuracy":  mark(m.accuracy,  paper.get("accuracy")),
        "f1":        mark(m.f1,        paper.get("f1")),
        "precision": mark(m.precision, paper.get("precision")),
        "recall":    mark(m.recall,    paper.get("recall")),
    }
