# Devign FROM CACHE: no training. Reads the existing l1/l2/l3_devign_cache
# folders and produces the ladder report + pooled-ensemble beat-both verdict.
# Use this on any machine that has the caches copied in (e.g. after merging
# rungs trained on different machines).
#
#   .\produce_devign.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& "$PSScriptRoot\make_ladder_devign.ps1"
