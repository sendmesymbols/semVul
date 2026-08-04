# FINAL_DEVIGN L3 - Train Devign L3 (soft routing gate)
#   -> experiments\runs\final_devign_l3_cache\
# Same encoders/inputs as final_devign_l2.ps1 (fine-tuned CodeT5+ + RoBERTa)
# PLUS a soft routing gate: w = sigmoid(MLP([expl_pooled; confidence])),
# pooled = w * code_pooled + (1-w) * expl_pooled.  Gate LR x100 so it
# actually moves.  No quality features (removed -- proven useless).
# 5 seeds, 12 epochs, max_code 320 (default, matches L1/L2) + max_text 512.
# Devign balanced -> no focal (train_rung auto-off), matching L1/L2.
# PURE INPUT: no apply_real_enrichment.py rebuild/de-anon step. Feeds
# ACTIVE\devign\{train,val}.jsonl exactly as they sit on disk.
#
#   .\final_devign_l3.ps1                 # default: gate enabled, LR x100
#   .\final_devign_l3.ps1 -Batch512 4     # >=16GB GPU
#   .\final_devign_l3.ps1 -HardConfSwitch -HardConfThreshold 85   # hard if/else instead
# Resumable: a finished rung JSON is skipped.
param(
    [int]$Batch512 = 2,
    [int[]]$Seeds = @(1, 2, 3, 4, 5),
    # Hard confidence switch (alternative to gate): per-sample if/else on
    # confidence field. Confidence >= threshold -> code ALONE; below ->
    # explanation ALONE. Validated direction: HIGH conf -> code (on reveal;
    # not yet checked on devign).
    [switch]$HardConfSwitch,
    [double]$HardConfThreshold = 85
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# The 8-column text channel (7-col decisive set PLUS purpose at the end; no spaces).
# IDENTICAL to final_devign_l2.ps1 so L2 vs L3 isolates only the gate.
$Cols = "confidence,risky_operations,missing_checks,function_name,called_functions,risky_apis,risk_summary,purpose"

$env:SEMVUL_QUAL_V2 = "0"
# Default: soft routing gate enabled (SEMVUL_QUAL_GATE=1), LR x100 so it moves.
# -HardConfSwitch overrides: uses hard if/else instead, no gate.
if ($HardConfSwitch) {
    $env:SEMVUL_HARD_CONF_SWITCH = "1"
    $env:SEMVUL_HARD_CONF_THRESH = "$HardConfThreshold"
    Remove-Item Env:SEMVUL_QUAL_GATE -ErrorAction SilentlyContinue
    Remove-Item Env:SEMVUL_GATE_LR_MULT -ErrorAction SilentlyContinue
} else {
    $env:SEMVUL_QUAL_GATE = "1"
    $env:SEMVUL_GATE_LR_MULT = "100"
    Remove-Item Env:SEMVUL_HARD_CONF_SWITCH -ErrorAction SilentlyContinue
    Remove-Item Env:SEMVUL_HARD_CONF_THRESH -ErrorAction SilentlyContinue
}

# No enrichment/rebuild: just require ACTIVE\devign\{train,val}.jsonl as-is.
$active = Join-Path $PSScriptRoot "explanations\SemanticVul\ACTIVE\devign"
foreach ($f in @("train.jsonl", "val.jsonl")) {
    if (-not (Test-Path (Join-Path $active $f))) {
        throw "ACTIVE\devign\$f missing -- this script does not rebuild/enhance explanations; place the pure file there first."
    }
}

$seedArgs = $Seeds | ForEach-Object { "$_" }
$CacheName = if ($HardConfSwitch) { "final_devign_l3_hardswitch_cache" } else { "final_devign_l3_cache" }
$gateFlag = @(); if (-not $HardConfSwitch) { $gateFlag = @("--qual-gate") }
# --fields $Cols: the 7-column decisive text channel (same as final_devign_l2.ps1).
# --cache-name: keep the hard-switch arm separate so resumable runs never mix
# the clean L3 baseline with the switched variant.
# --code-enc codet5p: CodeT5+ (FuSEVul's encoder). --max-text 512: mirror L1/L2.
# --epochs 12: explicit. No --max-code override (default 320, matches L1/L2).
# NOTE: NOT setting SEMVUL_FROZEN -- encoders fine-tune (unlike the frozen RQ2
# study in src/rqs/rq2.py), so L3 sits in the SAME regime as L1/L2 and
# aggregate_seeds.py reports all three rungs on one consistent scale.
& $py experiments\expl_enrich\reproduce_real.py --only devign --rungs L3 `
      --cache-name $CacheName --seeds @seedArgs --batch512 $Batch512 --fields $Cols @gateFlag `
      --code-enc codet5p --max-text 512 --epochs 12
if ($LASTEXITCODE -ne 0) { Write-Warning "L3 exited $LASTEXITCODE (partial kept)" }
