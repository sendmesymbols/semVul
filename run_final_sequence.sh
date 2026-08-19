#!/usr/bin/env bash
# One-command final sequence launcher.
#
# Default behavior:
#   run the requested final L1/L2/L3 launchers in ladder order; each launcher
#   reuses its matching final_<dataset>_<rung>_cache folder.
#
# Clean-Qwen generation is opt-in with --generate-clean-qwen.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

DATASET="both"
RUNGS="L1,L2,L3"
BATCH512=2
WORKERS=1
STRATIFIED=0
MODEL="qwen2.5-coder:14b"
HOST="http://localhost:9999"
FROM_STAGE=1
SKIP_GENERATION=0
GENERATE_CLEAN_QWEN=0
NO_THINK=0
EVIDENCE_WINDOW=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --rungs) RUNGS="$2"; shift 2 ;;
        --batch512) BATCH512="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --stratified) STRATIFIED="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --from-stage) FROM_STAGE="$2"; shift 2 ;;
        --skip-generation) SKIP_GENERATION=1; shift ;;
        --generate-clean-qwen) GENERATE_CLEAN_QWEN=1; shift ;;
        --no-think) NO_THINK=1; shift ;;
        --evidence-window) EVIDENCE_WINDOW=1; shift ;;
        -h|--help)
            cat <<'EOF'
Usage: ./run_final_sequence.sh [options]

Options:
  --dataset devign|reveal|both   default: both
  --rungs L1,L2,L3               comma-separated, default: L1,L2,L3
  --batch512 N                   default: 2
  --workers N                    explanation generation workers, default: 1
  --stratified N                 optional generation sample size
  --model NAME                   default: qwen2.5-coder:14b
  --host URL                     default: http://localhost:9999
  --from-stage 1|2|3             generation pipeline start stage
  --skip-generation              validate existing ACTIVE and train only
  --generate-clean-qwen          generate/promote clean ACTIVE before training
  --no-think                     forward to generation
  --evidence-window              forward to reveal final scripts
EOF
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "$DATASET" in
    devign|reveal|both) ;;
    *) echo "ERROR: --dataset must be devign, reveal, or both" >&2; exit 1 ;;
esac

run_step() {
    local name="$1"; shift
    echo
    echo "===== $name ====="
    "$@"
}

if [[ "$GENERATE_CLEAN_QWEN" -eq 1 && "$SKIP_GENERATION" -eq 1 ]]; then
    echo "ERROR: use either --generate-clean-qwen or --skip-generation, not both" >&2
    exit 1
fi

if [[ "$GENERATE_CLEAN_QWEN" -eq 1 ]]; then
    gen_args=(--dataset "$DATASET" --promote --workers "$WORKERS" --model "$MODEL" --host "$HOST" --from-stage "$FROM_STAGE")
    if [[ "$STRATIFIED" -gt 0 ]]; then gen_args+=(--stratified "$STRATIFIED"); fi
    if [[ "$NO_THINK" -eq 1 ]]; then gen_args+=(--no-think); fi
    run_step "Generate and promote clean ACTIVE explanations" bash generate_explanations.sh "${gen_args[@]}"
else
    echo "[sequence] Reusing existing final_*_cache results; generation is disabled."
fi

if [[ "$DATASET" == "both" ]]; then
    datasets=(devign reveal)
else
    datasets=("$DATASET")
fi

IFS=',' read -r -a rung_list <<< "$RUNGS"
for ds in "${datasets[@]}"; do
    for rung in "${rung_list[@]}"; do
        case "$rung" in
            L1|L2|L3) ;;
            *) echo "ERROR: unsupported rung '$rung' (use L1,L2,L3)" >&2; exit 1 ;;
        esac
        lower="$(tr '[:upper:]' '[:lower:]' <<< "$rung")"
        args=(--batch512 "$BATCH512")
        if [[ "$EVIDENCE_WINDOW" -eq 1 && "$ds" == "reveal" ]]; then
            args+=(--evidence-window)
        fi
        run_step "$ds $rung" bash "final_${ds}_${lower}.sh" "${args[@]}"
    done
done

echo
echo "Final sequence complete."
