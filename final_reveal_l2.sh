#!/usr/bin/env bash
# FINAL_REVEAL clean-Qwen run - Train ReVeal L2 (code + explanation channel)
#   -> experiments/runs/final_reveal_l2_cache/
# Text channel contains generator-produced structured fields only.
#   L2 - L1 = the explanation contribution. 5 seeds, 12 epochs (overnight).
#   max-code 512 + max-text 512 (matches FuSEVul + devign).
#
#   ./final_reveal_l2.sh                  # batch 2 (8GB); 512-token code window
#   ./final_reveal_l2.sh --batch512 4     # >=16GB GPU
# Resumable within the clean-Qwen cache family.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
. "scripts/cache_complete.sh"

BATCH512=2  # 2 fits 8GB at 512-tok code
EW=()       # --evidence-window (opt-in A/B): evidence-centered code span for L2/L3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch512) BATCH512="$2"; shift 2 ;;
        --evidence-window) EW=(--evidence-window); shift ;;
        *) echo "Unknown argument: $1 (supported: --batch512 N, --evidence-window)" >&2; exit 1 ;;
    esac
done

# Five fixed seeds for a paired ladder comparison.
SEEDS=(1 2 3 4 5)
# Qwen-only structured text channel; risk_level is deliberately excluded.
COLS="confidence,purpose,data_flow,risky_operations,missing_checks,evidence_tokens,safety_indicators,risk_summary"

# ---- ReVeal treatment knobs (HARDCODED; no env vars) ----
FOCAL_ALPHA=0.85
FOCAL_GAMMA=2.0
# ---------------------------------------------------------

if cache_complete reveal L2 final_reveal_l2_cache "${SEEDS[@]}"; then
    echo "[cache] final_reveal_l2_cache is complete; skipping validation and training."
    exit 0
fi

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

# Use the original cache family; the driver skips completed seed results and
# trains only missing seeds.
export SEMVUL_LEGACY_CACHE=1

# --fields $COLS: generator-produced structured fields only.
# --cache-name final_reveal_l2_cache: canonical cache; the driver verifies provenance.
# --max-text 512: FuSEVul text budget.
# --epochs 12: explicit (matches train_rung default; guaranteed for this run).
# --evidence-window (opt-in A/B): evidence-centered code span for L2/L3.
rc=0
"$PY" experiments/expl_enrich/reproduce_real.py --only reveal --rungs L2 \
      --cache-name final_reveal_l2_cache --seeds "${SEEDS[@]}" --fields "$COLS" \
      --code-enc codet5p --max-text 512 --max-code 512 --batch512 "$BATCH512" --epochs 12 ${EW[@]+"${EW[@]}"} \
      --focal-alpha "$FOCAL_ALPHA" --focal-gamma "$FOCAL_GAMMA" || rc=$?
if [[ $rc -ne 0 ]]; then echo "WARNING: L2 exited $rc (partial kept)" >&2; fi
