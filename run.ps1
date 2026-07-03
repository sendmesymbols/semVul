#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet("devign","reveal")]
    [string]$Dataset,

    [ValidateSet("L1","L2","L3")][string[]]$Ladders = @("L1","L2","L3"),

    [switch]$Ablate
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Route ALL HuggingFace downloads/loads into the project models/ dir
$env:HF_HOME            = Join-Path $PSScriptRoot "models"
$env:HF_HUB_CACHE       = Join-Path $env:HF_HOME  "hub"
$env:TRANSFORMERS_CACHE = $env:HF_HUB_CACHE

$logDir = Join-Path $PSScriptRoot "experiments\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "${Dataset}_run_${stamp}.log"
Start-Transcript -Path $logFile -Append | Out-Null

try {
    $args = @("-m","src.run","--dataset",$Dataset,"--ladders") + $Ladders
    if ($Ablate) { $args += "--ablate" }
    Write-Host "python $($args -join ' ')" -ForegroundColor Cyan
    & python @args
    if ($LASTEXITCODE -ne 0) { throw "run.py failed with exit code $LASTEXITCODE" }

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host " DONE. Reports:"                                              -ForegroundColor Green
    $reportsDir = Join-Path $PSScriptRoot "experiments\reports"
    Get-ChildItem -Path $reportsDir -Filter "${Dataset}_*.md" -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "   $($_.FullName)" }
    Write-Host " Log: $logFile"                                                -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
}
finally {
    Stop-Transcript | Out-Null
}
