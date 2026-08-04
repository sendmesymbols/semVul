# Train Devign L2 (code + de-anonymized explanation) -> runs\l2_devign_cache\
# This is where the real-enrichment explanation channel enters (L2 - L1 = the
# explanation contribution). 512-token window, train=enriched.clean.aug.real.
#
#   .\reproduce_devign_l2.ps1                 # seed 1337, batch 4 (>=16GB GPU)
#   .\reproduce_devign_l2.ps1 -Seeds 1337,2024 -Batch512 2
# Resumable: a finished rung JSON is skipped.
param([int]$Batch512 = 2)  # 2 fits 8GB (this laptop); use 4 on >=16GB
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds HARDCODED. Devign = single seed (ensemble buys ~0 on Devign; gap is
# structural truncation, not seed variance). Edit this one line to change.
$Seeds = @(1)

# Self-contained: if ACTIVE\devign\{train,val}.jsonl exist we do NOT touch the
# enriched source files at all. Only build them when ACTIVE is absent.
& $py experiments\expl_enrich\apply_real_enrichment.py --check --only devign
if ($LASTEXITCODE -ne 0) {
    Write-Host "ACTIVE/devign missing -> building from sources..."
    & $py experiments\expl_enrich\apply_real_enrichment.py --only devign --missing-only
    if ($LASTEXITCODE -ne 0) { throw "apply_real_enrichment (devign) failed" }
}

$seedArgs = $Seeds | ForEach-Object { "$_" }
# --fields prefix (2026-07-15): text channel = round-3 explanation.prefix —
# includes the REAL-code head + real literals (devign_real join), which the
# old field list lacked entirely. Fresh cache for the round-3 data.
& $py experiments\expl_enrich\reproduce_real.py --only devign --rungs L2 `
      --cache-name l2_devign_prefix3 --seeds @seedArgs --batch512 $Batch512 --fields prefix
if ($LASTEXITCODE -ne 0) { Write-Warning "L2 exited $LASTEXITCODE (partial kept)" }
