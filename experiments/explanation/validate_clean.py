"""Validate that detector inputs contain only generator-produced explanations.

This guard prevents legacy static/real-enriched ACTIVE files from being used by
the final ladder. It deliberately validates every row before an expensive run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "explanations" / "SemanticVul" / "ACTIVE"

REQUIRED = {
    "purpose", "data_flow", "risky_operations", "missing_checks",
    "evidence_tokens", "safety_indicators", "risk_summary", "risk_level",
    "confidence",
}
FORBIDDEN = {
    "llm_v1", "code_metrics", "tail_facts", "enrich", "real_enrich",
    "function_name", "called_functions", "risky_apis", "string_literals",
    "lexical_digest", "tail_digest", "prefix", "prefix_recipe",
}


def validate(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    rows = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            expl = row.get("explanation")
            if not isinstance(expl, dict):
                errors.append(f"line {lineno}: explanation is not an object")
                continue
            missing = sorted(REQUIRED - set(expl))
            forbidden = sorted(FORBIDDEN & set(expl))
            if missing:
                errors.append(f"line {lineno}: missing {', '.join(missing)}")
            if forbidden:
                errors.append(f"line {lineno}: legacy fields {', '.join(forbidden)}")
            confidence = expl.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                errors.append(f"line {lineno}: confidence is not a measured number")
            if len(errors) >= 20:
                break
    if rows == 0:
        errors.append("file contains no rows")
    return rows, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("devign", "reveal"))
    args = parser.parse_args()

    failed = False
    for split in ("train", "val"):
        path = ACTIVE / args.dataset / f"{split}.jsonl"
        if not path.exists():
            print(f"[clean-input] MISSING {path}")
            failed = True
            continue
        rows, errors = validate(path)
        if errors:
            failed = True
            print(f"[clean-input] REJECT {path} ({rows} rows scanned)")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[clean-input] OK {path} ({rows} rows)")

    if failed:
        print("Regenerate and promote clean Qwen outputs with generate_explanations.*; "
              "legacy enriched ACTIVE files cannot be used for final runs.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
