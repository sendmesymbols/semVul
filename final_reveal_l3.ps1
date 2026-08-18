# FINAL_REVEAL L3 - Train ReVeal L3 (soft routing gate)
#   -> experiments\runs\final_reveal_l3_cache\
# Same encoders/inputs as final_reveal_l2.ps1 (fine-tuned CodeT5+ + RoBERTa)
# PLUS a soft routing gate: w = sigmoid(MLP([expl_pooled; confidence])),
# pooled = w * code_pooled + (1-w) * expl_pooled.  Gate LR x100 so it
# actually moves.  No quality features (removed -- proven useless).
# 5 seeds, 12 epochs. max-code 512 + max-text 512 (matches L1/L2).
#
#   .\final_reveal_l3.ps1                 # default: gate enabled, LR x100
#   .\final_reveal_l3.ps1 -Batch512 4     # >=16GB GPU
#   .\final_reveal_l3.ps1 -HardConfSwitch -HardConfThreshold 85   # hard if/else instead
# Resumable: a finished rung JSON is skipped.
param(
    [int]$Batch512 = 2,
    [switch]$EvidenceWindow,
    # Hard confidence switch (alternative to gate): per-sample if/else on
    # confidence field. Confidence >= threshold -> code ALONE; below ->
    # explanation ALONE. Validated direction: HIGH conf -> code.
    [switch]$HardConfSwitch,
    [double]$HardConfThreshold = 85
)  # 2 fits 8GB at 512-tok code
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds HARDCODED (final_reveal: 5 seeds, matching final_reveal_l1/l2).
$Seeds = @(1, 2, 3, 4, 5)
# The clean Qwen-only explanation channel (no spaces).
# IDENTICAL to final_reveal_l2.ps1 so L2 vs L3 isolates only the gate.
$Cols = "confidence,purpose,data_flow,risky_operations,missing_checks,evidence_tokens,safety_indicators,risk_summary"

# ---- ReVeal treatment knobs (HARDCODED; IDENTICAL to final_reveal_l2.ps1) ----
$FocalAlpha = 0.85
$FocalGamma = 2.0
# -------------------------------------------------------------------------

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

# Reject missing or legacy enriched ACTIVE inputs before training.
& $py experiments\explanation\validate_clean.py --dataset reveal
if ($LASTEXITCODE -ne 0) { throw "ACTIVE/reveal is missing or contains legacy enriched inputs" }

$seedArgs = $Seeds | ForEach-Object { "$_" }
$CacheName = if ($HardConfSwitch) { "final_reveal_l3_hardswitch_cache" }
             else { "final_reveal_l3_cache" }
$gateFlag = @(); if (-not $HardConfSwitch) { $gateFlag = @("--qual-gate") }
$ew = @(); if ($EvidenceWindow) { $ew = @("--evidence-window") }
& $py experiments\expl_enrich\reproduce_real.py --only reveal --rungs L3 `
      --cache-name $CacheName --seeds @seedArgs --fields $Cols @gateFlag `
      --code-enc codet5p --max-text 512 --max-code 512 --batch512 $Batch512 --epochs 12 @ew `
      --focal-alpha $FocalAlpha --focal-gamma $FocalGamma
if ($LASTEXITCODE -ne 0) { Write-Warning "L3 exited $LASTEXITCODE (partial kept)" }
