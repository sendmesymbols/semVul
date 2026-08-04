#!/usr/bin/env bash
# Run the L2 explanation run for both datasets sequentially: reveal -> devign.
# The delegated scripts already hardcode 5 seeds and 12 epochs each:
#   - final_reveal_l2.sh
#   - final_devign_l2.sh
# This wrapper keeps their resumable behavior while exposing one shared batch arg.
# Default batch512=6, tuned for a 24GB GPU (linear scaling off the documented
# 8GB->2 / 16GB->4 points in the per-dataset scripts). Lower it if you hit OOM.
#
#   ./final_reveal_devign_l2.sh                  # batch 6 (24GB default)
#   ./final_reveal_devign_l2.sh --batch512 4
#   ./final_reveal_devign_l2.sh --batch512 4 --evidence-window
#
# Notes:
# - --evidence-window is forwarded only to reveal, because devign L2 does not
#   accept that flag.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

BATCH512=6
REVEAL_EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch512) BATCH512="$2"; shift 2 ;;
        --evidence-window) REVEAL_EXTRA=(--evidence-window); shift ;;
        *) echo "Unknown argument: $1 (supported: --batch512 N, --evidence-window)" >&2; exit 1 ;;
    esac
done

echo "==================================================================="
echo "==== final_reveal_l2  --  started $(date) ===="
echo "==================================================================="
bash "final_reveal_l2.sh" --batch512 "$BATCH512" "${REVEAL_EXTRA[@]}"

echo "==================================================================="
echo "==== final_devign_l2  --  started $(date) ===="
echo "==================================================================="
bash "final_devign_l2.sh" --batch512 "$BATCH512"

echo "==== final_reveal_devign_l2 complete  --  $(date) ===="
