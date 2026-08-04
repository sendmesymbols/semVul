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
# rename experiments\runs\final_reveal_l1_cache (else the old 320 runs are kept).
param([int]$Batch512 = 2, [switch]$EvidenceWindow)  # 2 fits 8GB at 512-tok code
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds HARDCODED (final_reveal: 5 seeds, matching final_reveal_l2/l3).
$Seeds = @(1, 2, 3, 4, 5)
# The 8-column text channel (7-col decisive set PLUS purpose at the end; no spaces).
# L1 ignores the text channel; kept identical to L2/L3 for a uniform launcher.
$Cols = "confidence,risky_operations,missing_checks,function_name,called_functions,risky_apis,risk_summary,purpose"

# ---- ReVeal treatment knobs (HARDCODED; no env vars) ----
$TailOffset = 220
$FocalAlpha = 0.85
$FocalGamma = 2.0
# ---------------------------------------------------------

# Self-contained: if ACTIVE\reveal\{train,val}.jsonl exist we do NOT touch the
# enriched source files (ACTIVE already carries tail_digest). Only build when
# ACTIVE is absent. To change $TailOffset, delete explanations\SemanticVul\
# ACTIVE\reveal\ first so this rebuilds with the new offset.
& $py experiments\expl_enrich\apply_real_enrichment.py --check --only reveal
if ($LASTEXITCODE -ne 0) {
    Write-Host "ACTIVE/reveal missing -> building from sources (tail-offset $TailOffset)..."
    & $py experiments\expl_enrich\apply_real_enrichment.py --only reveal --tail-offset $TailOffset
    if ($LASTEXITCODE -ne 0) { throw "apply_real_enrichment (reveal) failed" }
}

$seedArgs = $Seeds | ForEach-Object { "$_" }
# --code-enc codet5p: CodeT5+ code channel (FuSEVul's encoder).
# --cache-name final_reveal_l1_cache: fresh, independent output dir under experiments\runs\.
# --max-text 512: FuSEVul text budget; UNIFORM across the final_reveal ladder (L1/L2/L3).
# --epochs 12: explicit (matches train_rung default; guaranteed for this run).
$ew = @(); if ($EvidenceWindow) { $ew = @("--evidence-window") }
& $py experiments\expl_enrich\reproduce_real.py --only reveal --rungs L1 `
      --cache-name final_reveal_l1_cache --seeds @seedArgs --fields $Cols `
      --code-enc codet5p --max-text 512 --max-code 512 --batch512 $Batch512 --epochs 12 @ew `
      --focal-alpha $FocalAlpha --focal-gamma $FocalGamma
if ($LASTEXITCODE -ne 0) { Write-Warning "L1 exited $LASTEXITCODE (partial kept)" }
