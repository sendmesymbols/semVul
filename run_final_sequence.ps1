# One-command final sequence launcher.
#
# Default behavior:
#   run the requested final L1/L2/L3 launchers in ladder order; completed
#   legacy cache folders are detected and reused by the launchers.
#
# Clean-Qwen generation is opt-in with -GenerateCleanQwen. It is not part of
# the default reviewer/reproduction sequence.
#
# Examples:
#   .\run_final_sequence.ps1
#   .\run_final_sequence.ps1 -Dataset reveal -Rungs L1,L2,L3
#   .\run_final_sequence.ps1 -GenerateCleanQwen -Workers 8
param(
    [ValidateSet("devign", "reveal", "both")]
    [string]$Dataset = "both",
    [ValidateSet("L1", "L2", "L3")]
    [string[]]$Rungs = @("L1", "L2", "L3"),
    [int]$Batch512 = 2,
    [int]$Workers = 1,
    [int]$Stratified = 0,
    [string]$Model = "qwen2.5-coder:14b",
    [string]$OllamaHost = "http://localhost:9999",
    [ValidateRange(1, 3)]
    [int]$FromStage = 1,
    [switch]$NoThink,
    [switch]$EvidenceWindow,
    [switch]$GenerateCleanQwen,
    [switch]$SkipGeneration
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Step {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Script,
        [hashtable]$Parameters = @{}
    )
    Write-Host ""
    Write-Host "===== $Name ====="
    $stepPath = Join-Path $PSScriptRoot $Script
    if (-not (Test-Path -LiteralPath $stepPath -PathType Leaf)) {
        throw "Step script not found: $stepPath"
    }
    & $stepPath @Parameters
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

if ($GenerateCleanQwen -and $SkipGeneration) {
    throw "Use either -GenerateCleanQwen or -SkipGeneration, not both."
}

if ($GenerateCleanQwen) {
    $genParams = @{
        Dataset = $Dataset
        Promote = $true
        Workers = $Workers
        Model = $Model
        OllamaHost = $OllamaHost
        FromStage = $FromStage
    }
    if ($Stratified -gt 0) { $genParams.Stratified = $Stratified }
    if ($NoThink) { $genParams.NoThink = $true }
    Invoke-Step -Name "Generate and promote clean ACTIVE explanations" `
        -Script "generate_explanations.ps1" -Parameters $genParams
} else {
    Write-Host "[sequence] Reusing existing final_*_cache results; generation is disabled."
}

$datasets = if ($Dataset -eq "both") { @("devign", "reveal") } else { @($Dataset) }
foreach ($ds in $datasets) {
    foreach ($rung in $Rungs) {
        $rungName = $rung.ToLowerInvariant()
        $script = "final_${ds}_${rungName}.ps1"
        $stepParams = @{ Batch512 = $Batch512 }
        if ($EvidenceWindow -and $ds -eq "reveal") { $stepParams.EvidenceWindow = $true }
        if ($GenerateCleanQwen) { $stepParams.CleanQwen = $true }
        Invoke-Step -Name "$ds $rung" -Script $script -Parameters $stepParams
    }
}

Write-Host ""
Write-Host "Final sequence complete."
