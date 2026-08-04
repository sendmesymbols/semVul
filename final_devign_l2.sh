#!/usr/bin/env bash
# FINAL_DEVIGN 7-col run - Train Devign L2 (code + explanation channel)
#   -> experiments/runs/final_devign_l2_cache/
# Text channel = same 7 explanation columns as final_reveal_l2/l3 (8-col set
# MINUS risk_level): confidence, risky_operations, missing_checks,
# function_name, called_functions, risky_apis, risk_summary (via --fields
# comma-list -> SEMVUL_EXPL_FIELDS; serialized by src/data_io.py).
# L2 - L1 = the explanation contribution. Seeds are configurable via --seeds
# (default 1..5, matching final_devign_l1 for a paired L1->L2 delta; pass
# --seeds 1 for a fast single-seed run -- Devign ensemble reportedly buys ~0).
# PURE INPUT: no apply_real_enrichment.py rebuild/de-anon step. Feeds
# ACTIVE/devign/{train,val}.jsonl exactly as they sit on disk (data.py already
# reads ACTIVE verbatim) -- fails loudly instead of silently regenerating them.
#
#   ./final_devign_l2.sh                  # 5 seeds, batch 2 (this laptop)
#   ./final_devign_l2.sh --batch512 4     # 5 seeds, >=16GB GPU
#   ./final_devign_l2.sh --seeds 1        # single seed (fast)
#   ./final_devign_l2.sh --seeds 1,2,3    # comma-list also accepted
# Resumable: a finished rung JSON is skipped.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

BATCH512=2         # 2 fits 8GB, 4 on >=16GB
SEEDS=(1 2 3 4 5)  # default 1..5; see header
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch512) BATCH512="$2"; shift 2 ;;
        --seeds) IFS=', ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        *) echo "Unknown argument: $1 (supported: --batch512 N, --seeds LIST)" >&2; exit 1 ;;
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

# The 8-column text channel (7-col decisive set PLUS purpose at the end; no spaces).
COLS="confidence,risky_operations,missing_checks,function_name,called_functions,risky_apis,risk_summary,purpose"

# No enrichment/rebuild: just require ACTIVE/devign/{train,val}.jsonl as-is.
ACTIVE="$PWD/explanations/SemanticVul/ACTIVE/devign"
for f in train.jsonl val.jsonl; do
    if [[ ! -f "$ACTIVE/$f" ]]; then
        echo "ERROR: ACTIVE/devign/$f missing -- this script does not rebuild/enhance explanations; place the pure file there first." >&2
        exit 1
    fi
done

# --fields $COLS: the 7-column decisive text channel (see COLS above), same
# columns used for final_reveal_l2/l3 instead of the round-3 "prefix" channel.
# --cache-name final_devign_l2_cache: fresh, independent output dir under experiments/runs/.
# --code-enc codet5p: CodeT5+ (FuSEVul's encoder). --max-text 512: mirror reveal
# (both windows 512 for devign; heavier -> lower --batch512 if VRAM-bound).
# --epochs 12: explicit. Devign is balanced -> no focal (train_rung auto-off).
rc=0
"$PY" experiments/expl_enrich/reproduce_real.py --only devign --rungs L2 \
      --cache-name final_devign_l2_cache --seeds "${SEEDS[@]}" --batch512 "$BATCH512" --fields "$COLS" \
      --code-enc codet5p --max-text 512 --epochs 12 || rc=$?
if [[ $rc -ne 0 ]]; then echo "WARNING: L2 exited $rc (partial kept)" >&2; fi
