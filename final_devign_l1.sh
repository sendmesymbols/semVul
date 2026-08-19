#!/usr/bin/env bash
# FINAL_DEVIGN run - Train Devign L1 (code-only baseline)
#   -> experiments/runs/final_devign_l1_cache/
# Code channel = CodeT5+ (Salesforce/codet5p-110m-embedding), FuSEVul's encoder
# (loaded via the is_decoder shim in train.py). Text channel = RoBERTa. L1 is
# code-only, so the text channel/fields do NOT bind here -- kept for a uniform
# launcher and a matched config across the final_devign_l{1,2,3} ladder.
# L2 - L1 = the explanation contribution. 5 seeds, 12 epochs, max_code 512 +
# max_text 512 (mirrors reveal / devign L2/L3).
# PURE INPUT: no post-generation enrichment or identifier recovery. Feeds
# ACTIVE/devign/{train,val}.jsonl exactly as they sit on disk.
#
#   ./final_devign_l1.sh                  # 5 seeds, batch 2
#   ./final_devign_l1.sh --batch512 4     # >=16GB GPU
# Resumable: a finished rung JSON is skipped.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
. "scripts/cache_complete.sh"

BATCH512=2  # 2 fits 8GB; use 4 on >=16GB
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch512) BATCH512="$2"; shift 2 ;;
        *) echo "Unknown argument: $1 (supported: --batch512 N)" >&2; exit 1 ;;
    esac
done

# Seeds HARDCODED: 5 seeds (matching final_reveal + final_devign_l2/l3).
SEEDS=(1 2 3 4 5)
if cache_complete devign L1 final_devign_l1_cache "${SEEDS[@]}"; then
    echo "[cache] final_devign_l1_cache is complete; skipping validation and training."
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

# Use the original cache family; the driver skips completed seed results and
# trains only missing seeds.
export SEMVUL_LEGACY_CACHE=1

# --code-enc codet5p: CodeT5+ (FuSEVul's encoder). --max-text 512: mirror reveal
# (both windows 512 for devign). --epochs 12: explicit. Devign balanced -> no focal.
# --cache-name final_devign_l1_cache: canonical cache; the driver verifies provenance.
rc=0
"$PY" experiments/expl_enrich/reproduce_real.py --only devign --rungs L1 \
      --cache-name final_devign_l1_cache --seeds "${SEEDS[@]}" --batch512 "$BATCH512" --fields "$COLS" \
      --code-enc codet5p --max-text 512 --epochs 12 || rc=$?
if [[ $rc -ne 0 ]]; then echo "WARNING: L1 exited $rc (partial kept)" >&2; fi
