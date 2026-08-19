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
# Clean-Qwen results use a separate cache, so legacy runs cannot be reused.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
. "scripts/cache_complete.sh"

BATCH512=2  # 2 fits 8GB at 512-tok code
EW=()       # --evidence-window (opt-in A/B)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch512) BATCH512="$2"; shift 2 ;;
        --evidence-window) EW=(--evidence-window); shift ;;
        *) echo "Unknown argument: $1 (supported: --batch512 N, --evidence-window)" >&2; exit 1 ;;
    esac
done

# Seeds HARDCODED (final_reveal: 5 seeds, matching final_reveal_l2/l3).
SEEDS=(1 2 3 4 5)
if cache_complete reveal L1 final_reveal_l1_cache "${SEEDS[@]}"; then
    echo "[cache] final_reveal_l1_cache is complete; skipping validation and training."
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
# The clean Qwen-only explanation channel (unused by L1, retained for parity).
# L1 ignores the text channel; kept identical to L2/L3 for a uniform launcher.
COLS="confidence,purpose,data_flow,risky_operations,missing_checks,evidence_tokens,safety_indicators,risk_summary"

# ---- ReVeal treatment knobs (HARDCODED; no env vars) ----
FOCAL_ALPHA=0.85
FOCAL_GAMMA=2.0
# ---------------------------------------------------------

# Use the original cache family; the driver skips completed seed results and
# trains only missing seeds.
export SEMVUL_LEGACY_CACHE=1

# --code-enc codet5p: CodeT5+ code channel (FuSEVul's encoder).
# --cache-name final_reveal_l1_cache: canonical cache; the driver verifies provenance.
# --max-text 512: FuSEVul text budget; UNIFORM across the final_reveal ladder (L1/L2/L3).
# --epochs 12: explicit (matches train_rung default; guaranteed for this run).
rc=0
"$PY" experiments/expl_enrich/reproduce_real.py --only reveal --rungs L1 \
      --cache-name final_reveal_l1_cache --seeds "${SEEDS[@]}" --fields "$COLS" \
      --code-enc codet5p --max-text 512 --max-code 512 --batch512 "$BATCH512" --epochs 12 ${EW[@]+"${EW[@]}"} \
      --focal-alpha "$FOCAL_ALPHA" --focal-gamma "$FOCAL_GAMMA" || rc=$?
if [[ $rc -ne 0 ]]; then echo "WARNING: L1 exited $rc (partial kept)" >&2; fi
