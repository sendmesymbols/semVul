#!/usr/bin/env bash
# Run the L1 baseline for both datasets sequentially: reveal -> devign.
# The delegated scripts already hardcode 5 seeds and 12 epochs each:
#   - final_reveal_l1.sh
#   - final_devign_l1.sh
# This wrapper keeps their resumable behavior while exposing one shared batch arg.
#
#   ./final_reveal_devign_l1.sh
#   ./final_reveal_devign_l1.sh --batch512 4
#   ./final_reveal_devign_l1.sh --batch512 4 --evidence-window
#
# Notes:
# - --evidence-window is forwarded only to reveal, because devign L1 does not
#   accept that flag.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

BATCH512=2
REVEAL_EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch512) BATCH512="$2"; shift 2 ;;
        --evidence-window) REVEAL_EXTRA=(--evidence-window); shift ;;
        *) echo "Unknown argument: $1 (supported: --batch512 N, --evidence-window)" >&2; exit 1 ;;
    esac
done

echo "==================================================================="
echo "==== final_reveal_l1  --  started $(date) ===="
echo "==================================================================="
bash "final_reveal_l1.sh" --batch512 "$BATCH512" "${REVEAL_EXTRA[@]}"

echo "==================================================================="
echo "==== final_devign_l1  --  started $(date) ===="
echo "==================================================================="
bash "final_devign_l1.sh" --batch512 "$BATCH512"

echo "==== final_reveal_devign_l1 complete  --  $(date) ===="
