"""Canonical data source: SemanticVul JSONL (contains code + label + explanation).

Each JSONL row: {sample_id, label (0/1), raw_code, explanation: {purpose,
data_flow, risky_operations[], missing_checks[], evidence_tokens[], risk_summary}}

Final runs select validated Qwen-only files from ACTIVE/. Legacy variant
selection remains only for archived exploratory scripts.
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


# --- explanation-text serialization (configurable field set) ---------------
# Measured 2026-07-08 (leave-one-out held-out AUC): risk_summary / risk_level /
# confidence / tail_facts each add ~0 to the text channel yet cost 40-55% of its
# token budget (risk_summary alone is the 64%-coverage "No unguarded operation.."
# boilerplate). The default channel therefore drops them and emits guards as
# check-only (no verbatim quote). risk_level / confidence / code_metrics still
# reach the model as cheap scalars via quality_features_v2. Placeholder purposes
# ("No model-generated explanation available") are blanked. Overrides:
#   SEMVUL_EXPL_FIELDS=full         emit every field (pre-2026-07-08 behavior)
#   SEMVUL_EXPL_FIELDS=a,b,c        emit exactly this comma list
#   SEMVUL_SAFETY_EVIDENCE=1        restore verbatim guard evidence quotes
_FULL_EXPL_FIELDS = ("purpose", "data_flow", "risk_level", "risky_operations",
                     "missing_checks", "safety_indicators", "tail_facts",
                     "risk_summary")
_DEFAULT_EXPL_FIELDS = ("purpose", "data_flow", "risky_operations",
                        "missing_checks", "safety_indicators")
_PLACEHOLDER_PURPOSE = "No model-generated explanation"


def _expl_field_set():
    v = os.environ.get("SEMVUL_EXPL_FIELDS", "").strip().lower()
    if not v or v == "trim":
        return _DEFAULT_EXPL_FIELDS
    if v == "full":
        return _FULL_EXPL_FIELDS
    return tuple(x.strip() for x in v.split(",") if x.strip())


def _render_expl_field(e: dict, name: str) -> str:
    if name == "purpose":
        p = _to_str(e.get("purpose"))
        return "" if _PLACEHOLDER_PURPOSE in p else p
    if name == "data_flow":
        return _to_str(e.get("data_flow"))
    if name == "risk_level":
        rl = _to_str(e.get("risk_level"))
        return f"overall risk level: {rl}." if rl else ""
    if name == "risky_operations":
        return " ".join(_to_list(e.get("risky_operations")))
    if name == "missing_checks":
        return " ".join(_to_list(e.get("missing_checks")))
    if name == "safety_indicators":
        keep_ev = os.environ.get("SEMVUL_SAFETY_EVIDENCE", "0") == "1"
        gs = []
        for g in (e.get("safety_indicators") or []):
            if isinstance(g, dict):
                c = _to_str(g.get("check"))
                gs.append(f"guard present: {c} [{_to_str(g.get('evidence'))}]"
                          if keep_ev else f"guard present: {c}")
        return " ".join(gs)
    if name == "tail_facts":
        return _to_str(e.get("tail_facts"))
    if name == "risk_summary":
        return _to_str(e.get("risk_summary"))
    if name == "evidence_tokens":
        return " ".join(_to_list(e.get("evidence_tokens")))
    # Any explicitly selected future generator field: emit verbatim.
    return _to_str(e.get(name))


@dataclass
class Sample:
    sample_id: str
    label: int
    code: str
    explanation: dict

    @property
    def explanation_text(self) -> str:
        e = self.explanation or {}
        parts = [_render_expl_field(e, f) for f in _expl_field_set()]
        return " ".join(p for p in parts if p).strip()


def _active_path(dataset: str, split: str) -> Path:
    """Consolidated single-source input: explanations/SemanticVul/ACTIVE/<ds>/<split>.jsonl.
    Copying just this folder onto another machine is enough to run."""
    return EXPL_DIR / "ACTIVE" / dataset / f"{split}.jsonl"


def _jsonl_path(dataset: str, split: str) -> Path:
    ap = _active_path(dataset, split)
    # SEMVUL_ACTIVE_DIR set -> ACTIVE is the canonical input when present.
    if os.environ.get("SEMVUL_ACTIVE_DIR", "").strip() and ap.exists():
        return ap
    variant = os.environ.get("SEMVUL_EXPL_VARIANT", "").strip()
    if split == "val":
        # Legacy exploratory variants; final launchers select ACTIVE instead.
        variant = os.environ.get("SEMVUL_VAL_VARIANT", "").strip() or variant
    suffix = f".{variant}" if variant else ""
    p = EXPL_DIR / dataset / f"{dataset}_{split}{suffix}.jsonl"
    # Fallback: if the long-named file isn't on this machine, use ACTIVE.
    if not p.exists() and ap.exists():
        return ap
    return p


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
