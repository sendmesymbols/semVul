# FINAL_REVEAL clean-Qwen run - Train ReVeal L2 (code + explanation channel)
#   -> experiments\runs\final_reveal_l2_cache\
# Text channel = generator-produced structured explanation fields only. Legacy
# static analysis, de-anonymisation, lexical digests, and recovered identifiers
# are rejected before training.
#   L2 - L1 = the explanation contribution. 5 seeds, 12 epochs (overnight).
#   max-code 512 + max-text 512 (matches FuSEVul + devign).
#
#   .\final_reveal_l2.ps1                 # batch 2 (8GB); 512-token code window
#   .\final_reveal_l2.ps1 -Batch512 4     # >=16GB GPU
# Resumable within the clean-Qwen cache family.
param([int]$Batch512 = 2, [switch]$EvidenceWindow)  # 2 fits 8GB at 512-tok code
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Five fixed seeds for a paired ladder comparison.
$Seeds = @(1, 2, 3, 4, 5)
# Qwen-only structured text channel. risk_level remains excluded; confidence is
# retained for the currently reported configuration and must be ablated before
# attributing gains exclusively to descriptive explanations.
$Cols = "confidence,purpose,data_flow,risky_operations,missing_checks,evidence_tokens,safety_indicators,risk_summary"

# ---- ReVeal treatment knobs (HARDCODED; no env vars) ----
$FocalAlpha = 0.85
$FocalGamma = 2.0
# ---------------------------------------------------------

# Fail closed if ACTIVE is absent or still contains legacy enriched fields.
& $py experiments\explanation\validate_clean.py --dataset reveal
if ($LASTEXITCODE -ne 0) { throw "ACTIVE/reveal is missing or contains legacy enriched inputs" }

$seedArgs = $Seeds | ForEach-Object { "$_" }
# --fields $Cols: generator-produced structured fields only (see $Cols above).
# --cache-name final_reveal_l2_cache: canonical cache; the driver verifies provenance.
# --max-text 512: FuSEVul text budget; the structured fields are short enough.
# --epochs 12: explicit (matches train_rung default; guaranteed for this run).
# -EvidenceWindow (opt-in A/B): evidence-centered code span for L2/L3.
$ew = @(); if ($EvidenceWindow) { $ew = @("--evidence-window") }
& $py experiments\expl_enrich\reproduce_real.py --only reveal --rungs L2 `
      --cache-name final_reveal_l2_cache --seeds @seedArgs --fields $Cols `
      --code-enc codet5p --max-text 512 --max-code 512 --batch512 $Batch512 --epochs 12 @ew `
      --focal-alpha $FocalAlpha --focal-gamma $FocalGamma
if ($LASTEXITCODE -ne 0) { Write-Warning "L2 exited $LASTEXITCODE (partial kept)" }
