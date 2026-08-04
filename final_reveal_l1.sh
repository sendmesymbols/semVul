#!/usr/bin/env bash
# FINAL_REVEAL run - Train ReVeal L1 (code-only baseline)
#   -> experiments/runs/final_reveal_l1_cache/
# Code channel = CodeT5+ (Salesforce/codet5p-110m-embedding), FuSEVul's encoder
# (loaded via the is_decoder shim in train.py). Text channel = RoBERTa. L1 is
# code-only (no explanation, no quality features), so the text/focal knobs below
# do NOT bind here -- kept for a uniform launcher and a matched config across the
# final_reveal_l{1,2,3} ladder. L2 - L1 = the explanation contribution.
# 5 seeds, 12 epochs, max-code 512 + max-text 512 (matches FuSEVul + devign).
#
#   ./final_reveal_l1.sh                  # batch 2 (8GB); 512-token code window
#   ./final_reveal_l1.sh --batch512 4     # >=16GB GPU
# Resumable: a finished rung JSON is skipped -- to RETRAIN at 512, first clear/
# rename experiments/runs/final_reveal_l1_cache (else the old 320 runs are kept).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

BATCH512=2  # 2 fits 8GB at 512-tok code
EW=()       # --evidence-window (opt-in A/B)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch512) BATCH512="$2"; shift 2 ;;
        --evidence-window) EW=(--evidence-window); shift ;;
        *) echo "Unknown argument: $1 (supported: --batch512 N, --evidence-window)" >&2; exit 1 ;;
    esac
done

# Prefer the already-activated venv; fall back to .venv/ or venv/ in the repo.
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
elif [[ -x "$PWD/.venv/bin/python" ]]; then
    PY="$PWD/.venv/bin/python"
elif [[ -x "$PWD/venv/bin/python" ]]; then
    PY="$PWD/venv/bin/python"
else
    echo "ERROR: no venv python found -- activate your venv or create one (python3 -m venv .venv)" >&2
    exit 1
fi

# Seeds HARDCODED (final_reveal: 5 seeds, matching final_reveal_l2/l3).
SEEDS=(1 2 3 4 5)
# The 8-column text channel (7-col decisive set PLUS purpose at the end; no spaces).
# L1 ignores the text channel; kept identical to L2/L3 for a uniform launcher.
COLS="confidence,risky_operations,missing_checks,function_name,called_functions,risky_apis,risk_summary,purpose"

# ---- ReVeal treatment knobs (HARDCODED; no env vars) ----
TAIL_OFFSET=220
FOCAL_ALPHA=0.85
FOCAL_GAMMA=2.0
# ---------------------------------------------------------

# Self-contained: if ACTIVE/reveal/{train,val}.jsonl exist we do NOT touch the
# enriched source files (ACTIVE already carries tail_digest). Only build when
# ACTIVE is absent. To change TAIL_OFFSET, delete explanations/SemanticVul/
# ACTIVE/reveal/ first so this rebuilds with the new offset.
if ! "$PY" experiments/expl_enrich/apply_real_enrichment.py --check --only reveal; then
    echo "ACTIVE/reveal missing -> building from sources (tail-offset $TAIL_OFFSET)..."
    "$PY" experiments/expl_enrich/apply_real_enrichment.py --only reveal --tail-offset "$TAIL_OFFSET" \
        || { echo "ERROR: apply_real_enrichment (reveal) failed" >&2; exit 1; }
fi

# --code-enc codet5p: CodeT5+ code channel (FuSEVul's encoder).
# --cache-name final_reveal_l1_cache: fresh, independent output dir under experiments/runs/.
# --max-text 512: FuSEVul text budget; UNIFORM across the final_reveal ladder (L1/L2/L3).
# --epochs 12: explicit (matches train_rung default; guaranteed for this run).
rc=0
"$PY" experiments/expl_enrich/reproduce_real.py --only reveal --rungs L1 \
      --cache-name final_reveal_l1_cache --seeds "${SEEDS[@]}" --fields "$COLS" \
      --code-enc codet5p --max-text 512 --max-code 512 --batch512 "$BATCH512" --epochs 12 ${EW[@]+"${EW[@]}"} \
      --focal-alpha "$FOCAL_ALPHA" --focal-gamma "$FOCAL_GAMMA" || rc=$?
if [[ $rc -ne 0 ]]; then echo "WARNING: L1 exited $rc (partial kept)" >&2; fi
