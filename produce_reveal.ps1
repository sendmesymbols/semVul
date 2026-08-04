# ReVeal FROM CACHE: no training. Reads l{1,2,3}_reveal_idfirst_gate and produces
# the ladder report + pooled-ensemble beat-both verdict. Use after merging rungs
# trained on different machines.
#
#   .\produce_reveal.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& "$PSScriptRoot\make_ladder_reveal.ps1"
