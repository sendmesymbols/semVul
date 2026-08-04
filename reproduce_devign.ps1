# Devign FROM SCRATCH: train L1 -> L2 -> L3 (each into its own cache) then report.
# This is the full overnight run for Devign. Rungs are separate scripts so a
# crash/kill only loses the rung in flight; re-launching skips finished rungs.
#
#   .\reproduce_devign.ps1                    # seed 1337, batch 4 (>=16GB GPU e.g. laptop2)
#   .\reproduce_devign.ps1 -Seeds 1337,2024   # add a seed for the ensemble (2x time)
#   .\reproduce_devign.ps1 -Batch512 2        # 8GB GPU
#
# Per-rung time ~2.5h at batch4 (~5h at batch2). 3 rungs, 1 seed ~= one night.
# Caches: runs\l1_devign_cache, l2_devign_cache, l3_devign_cache.
param([int]$Batch512 = 4)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Seeds (1,3,7) are hardcoded inside each per-rung script for a consistent ladder.
& "$PSScriptRoot\reproduce_devign_l1.ps1" -Batch512 $Batch512
& "$PSScriptRoot\reproduce_devign_l2.ps1" -Batch512 $Batch512
& "$PSScriptRoot\reproduce_devign_l3.ps1" -Batch512 $Batch512

# gather the three caches into the ladder report
& "$PSScriptRoot\make_ladder_devign.ps1"
