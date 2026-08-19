cache_complete() {
    local dataset="$1"
    local rung="$2"
    local cache_name="$3"
    shift 3
    local seed
    for seed in "$@"; do
        if [[ ! -f "$PWD/experiments/runs/$cache_name/s$seed/semanticvul_${dataset}_${rung}.json" &&
              ! -f "$PWD/experiments/runs/$cache_name/s$seed/fusevul_ladder_${dataset}_${rung}.json" ]]; then
            return 1
        fi
    done
    return 0
}
