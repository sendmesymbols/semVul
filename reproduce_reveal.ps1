# ReVeal FROM SCRATCH: train L1 -> L2 -> L3 (identifier-first arm: new text channel
# + L3 quality-gate + focal) each into its own cache, then report. Run on the ReVeal
# machine. Rungs are separate scripts so a crash only loses the rung in flight.
#
#   .\reproduce_reveal.ps1                     # seeds 1,3,7 (hardcoded per rung)
#
# Caches: runs\l1_reveal_idfirst_gate, l2_reveal_idfirst_gate, l3_reveal_idfirst_gate.
# WAIT for the single-seed L3 confirmation (runs\l3_reveal_idfirst_gate\s1) to FINISH
# before running this, or L3 seed 1 will double-train concurrently.
# ReVeal 320-token window is cheaper than Devign 512, but L2/L3 are ~7h/seed -> the
# full 3-seed ladder is multi-day. The old-channel baseline stays in l*_reveal_cache.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Seeds (1,3,7) are hardcoded inside each per-rung script for a consistent ladder.
& "$PSScriptRoot\reproduce_reveal_l1.ps1"
& "$PSScriptRoot\reproduce_reveal_l2.ps1"
& "$PSScriptRoot\reproduce_reveal_l3.ps1"

& "$PSScriptRoot\make_ladder_reveal.ps1"
