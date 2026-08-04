# FINAL_REVEAL 7-col run - Train ReVeal L2 (code + explanation channel)
#   -> experiments\runs\final_reveal_l2_cache\
# Text channel = 7 explanation columns (8-col set MINUS risk_level):
#   confidence, risky_operations, missing_checks, function_name,
#   called_functions, risky_apis, risk_summary   (via --fields comma-list ->
#   SEMVUL_EXPL_FIELDS; serialized by src/data_io.py). Focal loss knobs below.
#   L2 - L1 = the explanation contribution. 5 seeds, 12 epochs (overnight).
#   max-code 512 + max-text 512 (matches FuSEVul + devign).
#
#   .\final_reveal_l2.ps1                 # batch 2 (8GB); 512-token code window
#   .\final_reveal_l2.ps1 -Batch512 4     # >=16GB GPU
# Resumable: a finished rung JSON is skipped -- to RETRAIN at 512, first clear/
# rename experiments\runs\final_reveal_l2_cache (else the old 320 runs are kept).
param([int]$Batch512 = 2, [switch]$EvidenceWindow)  # 2 fits 8GB at 512-tok code
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds HARDCODED. s1,s2 already complete in final_reveal_l2_cache (final JSON
# present) -> resume logic skips them; only s3,s4,s5 train. 5 seeds => stability.
$Seeds = @(1, 2, 3, 4, 5)
# The 8-column text channel (7-col decisive set PLUS purpose at the end; no spaces).
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
# --fields $Cols: the 7-column decisive text channel (see $Cols above).
# --cache-name final_reveal_l2_cache: fresh, independent output dir under experiments\runs\.
# --max-text 512: FuSEVul text budget; the 7 columns are short so this is ample.
# --epochs 12: explicit (matches train_rung default; guaranteed for this run).
# -EvidenceWindow (opt-in A/B): evidence-centered code span for L2/L3.
$ew = @(); if ($EvidenceWindow) { $ew = @("--evidence-window") }
& $py experiments\expl_enrich\reproduce_real.py --only reveal --rungs L2 `
      --cache-name final_reveal_l2_cache --seeds @seedArgs --fields $Cols `
      --code-enc codet5p --max-text 512 --max-code 512 --batch512 $Batch512 --epochs 12 @ew `
      --focal-alpha $FocalAlpha --focal-gamma $FocalGamma
if ($LASTEXITCODE -ne 0) { Write-Warning "L2 exited $LASTEXITCODE (partial kept)" }
