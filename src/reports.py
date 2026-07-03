"""Report generators.

Two reports per dataset:

  1. Ladder progression -- L1 vs L2 vs L3 at all threshold policies + WIN/LOSE
     vs FuSEVul.
  2. Component ablation -- for each ladder, delta of dropping {explanation, QF,
     gate}. Answers 'is each component pulling its weight?'.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.config import FUSEVUL_TARGETS, REPORTS_DIR, RUNS_DIR
from src.eval import compare_to_paper


def _load(run_id: str) -> dict:
    p = RUNS_DIR / f"{run_id}.json"
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _row(name: str, m: dict, target: dict) -> str:
    marks = compare_to_paper(_M(m), target)
    def cell(k):
        v = m[k]
        mk = marks.get(k, "")
        return f"{v:.2f}  ({mk})"
    return f"| {name} | {m['threshold']:.3f} | {cell('accuracy')} | {cell('f1')} | {cell('precision')} | {cell('recall')} |"


class _M:
    def __init__(self, d):
        self.accuracy = d["accuracy"]
        self.f1 = d["f1"]
        self.precision = d["precision"]
        self.recall = d["recall"]


def ladder_progression(dataset: str, ladders: List[str] = None) -> Path:
    ladders = ladders or ["L1", "L2", "L3"]
    target = FUSEVUL_TARGETS[dataset]

    lines = [
        f"# Ladder progression -- {dataset}",
        "",
        f"FuSEVul targets: Acc={target['accuracy']}, F1={target['f1']}, "
        f"P={target['precision']}, R={target['recall']}",
        "",
    ]
    for lad in ladders:
        run_id = f"{dataset}_{lad}_full"
        p = RUNS_DIR / f"{run_id}.json"
        if not p.exists():
            lines.append(f"## {lad} (not trained yet)\n")
            continue
        d = _load(run_id)
        ens = d["ensemble_metrics"]
        lines += [
            f"## {lad} -- {run_id}",
            "",
            "| policy | thr | Acc | F1 | Precision | Recall |",
            "|---|---|---|---|---|---|",
            _row("fixed 0.5",           ens["fixed_050"],   target),
            _row("max balanced-acc",    ens["max_bal_acc"], target),
            _row("max F1",              ens["max_f1"],      target),
            "",
        ]
    out = REPORTS_DIR / f"{dataset}_ladder_progression.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {out.name}")
    return out


def component_ablation(dataset: str, ladder: str) -> Path:
    """Compare full vs {no_expl, no_qual, concat} at the SAME ladder."""
    target = FUSEVUL_TARGETS[dataset]
    tags = ["full", "no_expl", "no_qual", "concat"]
    rows = []
    for tag in tags:
        run_id = f"{dataset}_{ladder}_{tag}"
        p = RUNS_DIR / f"{run_id}.json"
        if not p.exists():
            rows.append((tag, None))
        else:
            rows.append((tag, _load(run_id)["ensemble_metrics"]))

    full_outer = next((m for t, m in rows if t == "full"), None)

    def _cell(m, k): return f"{m[k]:.2f}" if m else "-"
    def _delta(m, base, k):
        if m is None or base is None: return ""
        d = m[k] - base[k]
        return f"  ({'+' if d >= 0 else ''}{d:.2f})"

    def _section(policy_key: str, title: str) -> list:
        base = full_outer[policy_key] if full_outer else None
        out = [
            f"## {title}",
            "",
            "| Config | Acc | F1 | Precision | Recall |",
            "|---|---|---|---|---|",
        ]
        for tag, m in rows:
            if m is None:
                out.append(f"| {tag} | - | - | - | - |")
                continue
            cur = m[policy_key]
            out.append(
                f"| {tag} | {_cell(cur,'accuracy')}{_delta(cur,base,'accuracy')} | "
                f"{_cell(cur,'f1')}{_delta(cur,base,'f1')} | "
                f"{_cell(cur,'precision')}{_delta(cur,base,'precision')} | "
                f"{_cell(cur,'recall')}{_delta(cur,base,'recall')} |"
            )
        out.append("")
        return out

    lines = [
        f"# Component ablation -- {dataset} at {ladder}",
        "",
        "Each row uses the SAME cached embeddings; only the head varies.",
        "'delta' columns = tag - full (positive = the tag helps).",
        "",
    ]
    lines += _section("fixed_050",   "fixed 0.5 threshold")
    lines += _section("max_bal_acc", "max balanced-accuracy threshold (honest headline)")
    lines += _section("max_f1",      "max-F1 threshold")

    lines += [
        "",
        f"FuSEVul target for reference: Acc={target['accuracy']}, F1={target['f1']}",
        "",
    ]
    out = REPORTS_DIR / f"{dataset}_{ladder}_component_ablation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {out.name}")
    return out


def all_reports(dataset: str, ladders: List[str] = None) -> List[Path]:
    ladders = ladders or ["L1", "L2", "L3"]
    outs = [ladder_progression(dataset, ladders)]
    # Component ablation only makes sense for ladders that actually train a head
    # with ablatable components. L3 is a probability ensemble -- no such knobs.
    for lad in ladders:
        if lad == "L3":
            continue
        outs.append(component_ablation(dataset, lad))
    return outs
