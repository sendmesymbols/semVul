#!/usr/bin/env bash
# FINAL_REVEAL L3 - Train ReVeal L3 (soft routing gate)
#   -> experiments/runs/final_reveal_l3_cache/
# Same encoders/inputs as final_reveal_l2.sh (fine-tuned CodeT5+ + RoBERTa)
# PLUS a soft routing gate: w = sigmoid(MLP([expl_pooled; confidence])),
# pooled = w * code_pooled + (1-w) * expl_pooled.  Gate LR x100 so it
# actually moves.  No quality features (removed -- proven useless).
# 5 seeds, 12 epochs. max-code 512 + max-text 512 (matches L1/L2).
#
#   ./final_reveal_l3.sh                  # default: gate enabled, LR x100
#   ./final_reveal_l3.sh --batch512 4     # >=16GB GPU
#   ./final_reveal_l3.sh --hard-conf-switch --hard-conf-threshold 85
# Resumable: a finished rung JSON is skipped.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

BATCH512=2
EW=()
HARD_CONF_SWITCH=0
HARD_CONF_THRESH=85
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch512) BATCH512="$2"; shift 2 ;;
        --evidence-window) EW=(--evidence-window); shift ;;
        --hard-conf-switch) HARD_CONF_SWITCH=1; shift ;;
        --hard-conf-threshold) HARD_CONF_THRESH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1 (supported: --batch512 N, --evidence-window, --hard-conf-switch, --hard-conf-threshold N)" >&2; exit 1 ;;
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

# Seeds HARDCODED (final_reveal: 5 seeds, matching final_reveal_l1/l2).
SEEDS=(1 2 3 4 5)
# The 8-column text channel (7-col decisive set PLUS purpose at the end; no spaces).
# IDENTICAL to final_reveal_l2.sh so L2 vs L3 isolates only the gate.
COLS="confidence,risky_operations,missing_checks,function_name,called_functions,risky_apis,risk_summary,purpose"

# ---- ReVeal treatment knobs (HARDCODED; IDENTICAL to final_reveal_l2.sh) ----
TAIL_OFFSET=220
FOCAL_ALPHA=0.85
FOCAL_GAMMA=2.0
# -------------------------------------------------------------------------

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

# Self-contained: if ACTIVE/reveal/{train,val}.jsonl exist we do NOT touch the
# enriched source files (ACTIVE already carries tail_digest). Only build when
# ACTIVE is absent. To change TAIL_OFFSET, delete explanations/SemanticVul/
# ACTIVE/reveal/ first so this rebuilds with the new offset.
if ! "$PY" experiments/expl_enrich/apply_real_enrichment.py --check --only reveal; then
    echo "ACTIVE/reveal missing -> building from sources (tail-offset $TAIL_OFFSET)..."
    "$PY" experiments/expl_enrich/apply_real_enrichment.py --only reveal --tail-offset "$TAIL_OFFSET" \
        || { echo "ERROR: apply_real_enrichment (reveal) failed" >&2; exit 1; }
fi

# --fields $COLS: the 7-column decisive text channel (see COLS above).
# --cache-name: keep the hard-switch arm separate so resumable runs never mix
# the clean L3 baseline with the switched variant.
# --max-text 512: FuSEVul text budget; the 7 columns are short so this is ample.
# --epochs 12: explicit (matches L1/L2; guaranteed for this run).
# --evidence-window (opt-in A/B): evidence-centered code span for L2/L3.
# NOTE: NOT setting SEMVUL_FROZEN -- encoders fine-tune (unlike the frozen RQ2
# study in src/rqs/rq2.py), so L3 sits in the SAME regime as L1/L2 and
# aggregate_seeds.py reports all three rungs on one consistent scale.
rc=0
CACHE_NAME="final_reveal_l3_cache"
GATE_FLAG=(--qual-gate)
if [[ "$HARD_CONF_SWITCH" == "1" ]]; then
    CACHE_NAME="final_reveal_l3_hardswitch_cache"
    GATE_FLAG=()
fi
"$PY" experiments/expl_enrich/reproduce_real.py --only reveal --rungs L3 \
      --cache-name "$CACHE_NAME" --seeds "${SEEDS[@]}" --fields "$COLS" ${GATE_FLAG[@]+"${GATE_FLAG[@]}"} \
      --code-enc codet5p --max-text 512 --max-code 512 --batch512 "$BATCH512" --epochs 12 ${EW[@]+"${EW[@]}"} \
      --focal-alpha "$FOCAL_ALPHA" --focal-gamma "$FOCAL_GAMMA" || rc=$?
if [[ $rc -ne 0 ]]; then echo "WARNING: L3 exited $rc (partial kept)" >&2; fi
