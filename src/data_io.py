"""Canonical data source: SemanticVul JSONL (contains code + label + explanation).

Each JSONL row: {sample_id, label (0/1), raw_code, explanation: {purpose,
data_flow, risky_operations[], missing_checks[], evidence_tokens[], risk_summary}}

Variant selection: set SEMVUL_EXPL_VARIANT=enriched to load
<ds>_<split>.enriched.jsonl (written by experiments/expl_enrich/run_enrich.py)
instead of the original files. Enriched rows carry extra fields
(safety_indicators, tail_facts, risk_level, code_metrics) which
explanation_text folds into the text channel when present.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from src.config import EXPL_DIR


def _to_str(v) -> str:
    """Coerce any explanation field to a flat string. JSONL is not fully clean:
    e.g. one Devign row has risk_summary as a nested dict."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return " ".join(_to_str(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_to_str(x) for x in v.values())
    return str(v)


def _to_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [_to_str(x) for x in v]
    if isinstance(v, tuple):
        return [_to_str(x) for x in v]
    if isinstance(v, dict):
        return [_to_str(x) for x in v.values()]
    return [_to_str(v)]


@dataclass
class Sample:
    sample_id: str
    label: int
    code: str
    explanation: dict

    @property
    def explanation_text(self) -> str:
        e = self.explanation or {}
        parts = [
            _to_str(e.get("purpose")),
            _to_str(e.get("data_flow")),
        ]
        if e.get("risk_level"):  # enriched rows: lead with the calibrated level
            parts.append(f"overall risk level: {_to_str(e.get('risk_level'))}.")
        parts += [
            " ".join(_to_list(e.get("risky_operations"))),
            " ".join(_to_list(e.get("missing_checks"))),
        ]
        for g in (e.get("safety_indicators") or []):
            if isinstance(g, dict):
                parts.append(f"guard present: {_to_str(g.get('check'))} "
                             f"[{_to_str(g.get('evidence'))}]")
        if e.get("tail_facts"):
            parts.append(_to_str(e.get("tail_facts")))
        parts.append(_to_str(e.get("risk_summary")))
        return " ".join(p for p in parts if p).strip()


def _jsonl_path(dataset: str, split: str) -> Path:
    variant = os.environ.get("SEMVUL_EXPL_VARIANT", "").strip()
    suffix = f".{variant}" if variant else ""
    return EXPL_DIR / dataset / f"{dataset}_{split}{suffix}.jsonl"


def iter_samples(dataset: str, split: str) -> Iterator[Sample]:
    path = _jsonl_path(dataset, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing explanations JSONL: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield Sample(
                sample_id=str(row.get("sample_id", "")),
                label=int(row["label"]),
                code=row.get("raw_code", "") or "",
                explanation=row.get("explanation", {}) or {},
            )


def load_split(dataset: str, split: str) -> List[Sample]:
    return list(iter_samples(dataset, split))
