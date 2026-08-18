# FINAL_REVEAL run - Train ReVeal L1 (code-only baseline)
#   -> experiments\runs\final_reveal_l1_cache\
# Code channel = CodeT5+ (Salesforce/codet5p-110m-embedding), FuSEVul's encoder
# (loaded via the is_decoder shim in train.py). Text channel = RoBERTa. L1 is
# code-only (no explanation, no quality features), so the text/focal knobs below
# do NOT bind here -- kept for a uniform launcher and a matched config across the
# final_reveal_l{1,2,3} ladder. L2 - L1 = the explanation contribution.
# 5 seeds, 12 epochs, max-code 512 + max-text 512 (matches FuSEVul + devign).
#
#   .\final_reveal_l1.ps1                 # batch 2 (8GB); 512-token code window
#   .\final_reveal_l1.ps1 -Batch512 4     # >=16GB GPU
# Resumable: a finished rung JSON is skipped -- to RETRAIN at 512, first clear/
# Clean-Qwen results use a separate cache, so legacy runs cannot be reused.
param([int]$Batch512 = 2, [switch]$EvidenceWindow)  # 2 fits 8GB at 512-tok code
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds HARDCODED (final_reveal: 5 seeds, matching final_reveal_l2/l3).
$Seeds = @(1, 2, 3, 4, 5)
# The clean Qwen-only explanation channel (unused by L1, retained for parity).
# L1 ignores the text channel; kept identical to L2/L3 for a uniform launcher.
$Cols = "confidence,purpose,data_flow,risky_operations,missing_checks,evidence_tokens,safety_indicators,risk_summary"

# ---- ReVeal treatment knobs (HARDCODED; no env vars) ----
$FocalAlpha = 0.85
$FocalGamma = 2.0
# ---------------------------------------------------------

# Reject missing or legacy enriched ACTIVE inputs before training.
& $py experiments\explanation\validate_clean.py --dataset reveal
if ($LASTEXITCODE -ne 0) { throw "ACTIVE/reveal is missing or contains legacy enriched inputs" }

$seedArgs = $Seeds | ForEach-Object { "$_" }
# --code-enc codet5p: CodeT5+ code channel (FuSEVul's encoder).
# --cache-name final_reveal_l1_cache: canonical cache; the driver verifies provenance.
# --max-text 512: FuSEVul text budget; UNIFORM across the final_reveal ladder (L1/L2/L3).
# --epochs 12: explicit (matches train_rung default; guaranteed for this run).
$ew = @(); if ($EvidenceWindow) { $ew = @("--evidence-window") }
& $py experiments\expl_enrich\reproduce_real.py --only reveal --rungs L1 `
      --cache-name final_reveal_l1_cache --seeds @seedArgs --fields $Cols `
      --code-enc codet5p --max-text 512 --max-code 512 --batch512 $Batch512 --epochs 12 @ew `
      --focal-alpha $FocalAlpha --focal-gamma $FocalGamma
if ($LASTEXITCODE -ne 0) { Write-Warning "L1 exited $LASTEXITCODE (partial kept)" }
