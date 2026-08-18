#!/usr/bin/env bash
# GENERATE EXPLANATIONS - THE single entry point for the explanation dataset.
#
# Runs three stages and ends with a clean ACTIVE/ pair per dataset:
#
#   1 generate    purpose, data_flow, risky_operations, missing_checks,
#                 evidence_tokens, safety_indicators, risk_summary, risk_level,
#                 confidence  (MEASURED from decode-time logprobs, not self-reported)
#   2 install     stage-1 output -> canonical dataset files
#   3 activate    validate Qwen-only fields and copy to ACTIVE/
#
# By default it builds into experiments/explanation/work/ and does NOT touch the
# shipped explanations/SemanticVul/. Pass --promote to build in the shipped tree.
#
#   ./generate_explanations.sh --smoke                  # 6 rows/split, end-to-end proof
#   ./generate_explanations.sh                          # FULL both datasets (see note)
#   ./generate_explanations.sh --dataset reveal --split val
#   ./generate_explanations.sh --stratified 300 --workers 4
#   ./generate_explanations.sh --from-stage 2           # reuse stage-1 output
#   ./generate_explanations.sh --promote                # OVERWRITES shipped data
#
# RUNTIME: ~70 s/sample on qwen2.5-coder:14b. All four splits = 70,802 rows
# ~= 57 days sequential, ~7 days at --workers 8. Use --stratified/--smoke to prove
# the pipeline, then scale up. Resumable: re-running skips finished rows.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

MODEL="qwen2.5-coder:14b"
OLLAMA_HOST="http://localhost:9999"
FROM_STAGE=1
PASS=()          # forwarded verbatim to pipeline.py

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; PASS+=(--model "$2"); shift 2 ;;
        --host)        OLLAMA_HOST="$2"; PASS+=(--host "$2"); shift 2 ;;
        --from-stage)  FROM_STAGE="$2"; PASS+=(--from-stage "$2"); shift 2 ;;
        --dataset|--split|--mode|--stratified|--workers|--num-ctx|--timeout|\
        --tag|--work-dir|--to-stage)
                       PASS+=("$1" "$2"); shift 2 ;;
        --no-think|--smoke|--promote)
                       PASS+=("$1"); shift ;;
        -h|--help)     PASS+=(--help); shift ;;
        *) echo "Unknown argument: $1" >&2
           echo "Supported: --dataset devign|reveal|both --split train|val|both" >&2
           echo "           --model M --host URL --mode auto|anon|real" >&2
           echo "           --stratified N --workers N --num-ctx N --timeout S --tag T" >&2
           echo "           --work-dir D --from-stage 1-3 --to-stage 1-3" >&2
           echo "           --no-think --smoke --promote" >&2
           exit 1 ;;
    esac
done

# Prefer the already-activated venv; fall back to .venv/ or venv/ in the repo.
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PY="$VIRTUAL_ENV/bin/python"
elif [[ -x "$PWD/.venv/bin/python" ]]; then
    PY="$PWD/.venv/bin/python"
elif [[ -x "$PWD/.venv/Scripts/python.exe" ]]; then   # git-bash on Windows
    PY="$PWD/.venv/Scripts/python.exe"
elif [[ -x "$PWD/venv/bin/python" ]]; then
    PY="$PWD/venv/bin/python"
else
    echo "ERROR: no venv python found -- activate your venv or create one (python3 -m venv .venv)" >&2
    exit 1
fi

# Preflight the model only when stage 1 will actually run.
if [[ "$FROM_STAGE" -le 1 ]]; then
    echo "[gen] checking Ollama at $OLLAMA_HOST ..."
    if ! TAGS=$(curl -sf --max-time 20 "$OLLAMA_HOST/api/tags"); then
        echo "ERROR: cannot reach Ollama at $OLLAMA_HOST -- is the container up?" >&2
        exit 1
    fi
    if ! grep -q "\"$MODEL\"" <<<"$TAGS"; then
        echo "ERROR: model '$MODEL' not on the server at $OLLAMA_HOST" >&2
        echo "       available: $(grep -o '"name":"[^"]*"' <<<"$TAGS" | cut -d'"' -f4 | paste -sd, -)" >&2
        exit 1
    fi
    echo "[gen] Ollama OK, model '$MODEL' present."
fi

exec "$PY" experiments/explanation/pipeline.py ${PASS[@]+"${PASS[@]}"}
