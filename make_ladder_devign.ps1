# Gather Devign L1/L2/L3 caches -> ladder report (console + experiments\reports\ladder_devign.md).
# Reads runs\l1_devign_cache, l2_devign_cache, l3_devign_cache. No training.
#   .\make_ladder_devign.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py experiments\expl_enrich\make_ladder.py --ds devign
