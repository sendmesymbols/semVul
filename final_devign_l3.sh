#!/usr/bin/env bash
# FINAL_DEVIGN L3 - Train Devign L3 (soft routing gate)
#   -> experiments/runs/final_devign_l3_cache/
# Same encoders/inputs as final_devign_l2.sh (fine-tuned CodeT5+ + RoBERTa)
# PLUS a soft routing gate: w = sigmoid(MLP([expl_pooled; confidence])),
# pooled = w * code_pooled + (1-w) * expl_pooled.  Gate LR x100 so it
# actually moves.  No quality features (removed -- proven useless).
# 5 seeds, 12 epochs, max_code 320 (default, matches L1/L2) + max_text 512.
# Devign balanced -> no focal (train_rung auto-off), matching L1/L2.
# PURE INPUT: no post-generation enrichment or identifier recovery. Feeds
# ACTIVE/devign/{train,val}.jsonl exactly as they sit on disk.
#
#   ./final_devign_l3.sh                  # default: gate enabled, LR x100
#   ./final_devign_l3.sh --batch512 4     # >=16GB GPU
#   ./final_devign_l3.sh --seeds 1        # single seed (fast)
#   ./final_devign_l3.sh --hard-conf-switch --hard-conf-threshold 85   # hard if/else instead
# Resumable: a finished rung JSON is skipped.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
. "scripts/cache_complete.sh"

BATCH512=2
SEEDS=(1 2 3 4 5)
# Hard confidence switch (alternative to gate): per-sample if/else on
# confidence field. Confidence >= threshold -> code ALONE; below ->
# explanation ALONE. Validated direction: HIGH conf -> code (on reveal;
# not yet checked on devign).
HARD_CONF_SWITCH=0
HARD_CONF_THRESH=85
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch512) BATCH512="$2"; shift 2 ;;
        --seeds) IFS=', ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --hard-conf-switch) HARD_CONF_SWITCH=1; shift ;;
        --hard-conf-threshold) HARD_CONF_THRESH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1 (supported: --batch512 N, --seeds LIST, --hard-conf-switch, --hard-conf-threshold N)" >&2; exit 1 ;;
    esac
done

CACHE_NAME="final_devign_l3_cache"
if [[ "$HARD_CONF_SWITCH" == "1" ]]; then CACHE_NAME="final_devign_l3_hardswitch_cache"; fi
if cache_complete devign L3 "$CACHE_NAME" "${SEEDS[@]}"; then
    echo "[cache] $CACHE_NAME is complete; skipping validation and training."
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

# The clean Qwen-only explanation channel (no spaces).
# IDENTICAL to final_devign_l2.sh so L2 vs L3 isolates only the gate.
COLS="confidence,purpose,data_flow,risky_operations,missing_checks,evidence_tokens,safety_indicators,risk_summary"

export SEMVUL_QUAL_V2=0
# Default: soft routing gate enabled, LR x100.
# --hard-conf-switch overrides: hard if/else, no gate.
if [[ "$HARD_CONF_SWITCH" == "1" ]]; then
    export SEMVUL_HARD_CONF_SWITCH=1
    export SEMVUL_HARD_CONF_THRESH="$HARD_CONF_THRESH"
    unset SEMVUL_QUAL_GATE SEMVUL_GATE_LR_MULT || true
else
    export SEMVUL_QUAL_GATE=1
    export SEMVUL_GATE_LR_MULT=100
    unset SEMVUL_HARD_CONF_SWITCH SEMVUL_HARD_CONF_THRESH || true
fi

# Use the original cache family; the driver skips completed seed results and
# trains only missing seeds.
export SEMVUL_LEGACY_CACHE=1

# --fields $COLS: the clean Qwen-only channel (same as final_devign_l2.sh).
# --cache-name: keep the hard-switch arm separate so resumable runs never mix
# the clean L3 baseline with the switched variant.
# --code-enc codet5p: CodeT5+ (FuSEVul's encoder). --max-text 512: mirror L1/L2.
# --epochs 12: explicit. No --max-code override (default 320, matches L1/L2).
# NOTE: NOT setting SEMVUL_FROZEN -- encoders fine-tune (unlike the frozen RQ2
# study in src/rqs/rq2.py), so L3 sits in the SAME regime as L1/L2 and
# aggregate_seeds.py reports all three rungs on one consistent scale.
rc=0
GATE_FLAG=(--qual-gate)
if [[ "$HARD_CONF_SWITCH" == "1" ]]; then
    CACHE_NAME="final_devign_l3_hardswitch_cache"
    GATE_FLAG=()
fi
"$PY" experiments/expl_enrich/reproduce_real.py --only devign --rungs L3 \
      --cache-name "$CACHE_NAME" --seeds "${SEEDS[@]}" --batch512 "$BATCH512" --fields "$COLS" ${GATE_FLAG[@]+"${GATE_FLAG[@]}"} \
      --code-enc codet5p --max-text 512 --epochs 12 || rc=$?
if [[ $rc -ne 0 ]]; then echo "WARNING: L3 exited $rc (partial kept)" >&2; fi
