# Gather ReVeal L1/L2/L3 caches -> ladder report (console + experiments\reports\ladder_reveal.md).
# Reads the identifier-first arm: runs\l{1,2,3}_reveal_idfirst_gate. No training.
#   .\make_ladder_reveal.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py experiments\expl_enrich\make_ladder.py --ds reveal --cache-prefix "{rung}_{ds}_idfirst_gate"
