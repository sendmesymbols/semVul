#!/usr/bin/env bash
# Run the full final_devign ladder sequentially: L1 -> L2 -> L3.
# All arguments are passed through to every rung, e.g.:
#   ./final_devign_all.sh --batch512 8
# NOTE: L1 ignores --seeds (hardcoded 1..5); L2/L3 accept it. Passing --seeds
# here would abort at L1's arg check, so only pass --batch512 to this runner.
# Each rung is resumable (finished rung JSONs are skipped), so rerunning this
# after an interruption continues where it left off. A hard setup failure in a
# rung (missing venv / missing ACTIVE data) aborts the chain; a plain training
# failure only warns (partial kept, matching the individual scripts).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

for rung in 1 2 3; do
    echo "==================================================================="
    echo "==== final_devign_l$rung  --  started $(date) ===="
    echo "==================================================================="
    bash "final_devign_l$rung.sh" "$@"
done
echo "==== final_devign ladder complete  --  $(date) ===="
