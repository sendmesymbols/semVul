#!/usr/bin/env bash
# Run the full final_reveal ladder sequentially: L1 -> L2 -> L3.
# All arguments are passed through to every rung, e.g.:
#   ./final_reveal_all.sh --batch512 8
# Each rung is resumable (finished rung JSONs are skipped), so rerunning this
# after an interruption continues where it left off. A hard setup failure in a
# rung (missing venv / missing ACTIVE data) aborts the chain; a plain training
# failure only warns (partial kept, matching the individual scripts).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

for rung in 1 2 3; do
    echo "==================================================================="
    echo "==== final_reveal_l$rung  --  started $(date) ===="
    echo "==================================================================="
    bash "final_reveal_l$rung.sh" "$@"
done
echo "==== final_reveal ladder complete  --  $(date) ===="
