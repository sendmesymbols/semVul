# DECISIVE_REVEAL 7-col experiment - Train ReVeal L3 (code + explanation + 44-dim
#   quality features) -> experiments\runs\decisive_reveal_l3_7col\
# Full ladder top rung; L3 - L2 = quality-feature contribution. Text channel =
# 7 explanation columns (8-col set MINUS risk_level):
#   confidence, risky_operations, missing_checks, function_name,
#   called_functions, risky_apis, risk_summary   (via --fields comma-list ->
#   SEMVUL_EXPL_FIELDS). Focal loss below. 2 seeds, 12 epochs (overnight).
#
#   .\reproduce_reveal_l3.ps1
# Resumable: a finished rung JSON is skipped.
param([switch]$EvidenceWindow)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds HARDCODED (decisive_reveal: reduced from 1,3,7 to 2 seeds).
$Seeds = @(1, 2)
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
# --qual-gate: L3 quality-gated fusion (c) -> SEMVUL_QUAL_GATE=1 (L3-only).
# --fields $Cols: the 7-column decisive text channel (see $Cols above).
# --cache-name decisive_reveal_l3_7col: identifiable output dir under experiments\runs\.
# --max-text 320: ample for the short 7-column channel.
# --epochs 12: explicit (matches train_rung default; guaranteed for this run).
$ew = @(); if ($EvidenceWindow) { $ew = @("--evidence-window") }
& $py experiments\expl_enrich\reproduce_real.py --only reveal --rungs L3 `
      --cache-name decisive_reveal_l3_7col --seeds @seedArgs --qual-gate --fields $Cols `
      --max-text 320 --epochs 12 @ew `
      --focal-alpha $FocalAlpha --focal-gamma $FocalGamma
if ($LASTEXITCODE -ne 0) { Write-Warning "L3 exited $LASTEXITCODE (partial kept)" }
