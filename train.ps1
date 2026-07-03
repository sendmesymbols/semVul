#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet("devign","reveal")]
    [string]$Dataset,

    [Parameter(Mandatory=$true)][ValidateSet("L1","L2")]
    [string]$Ladder,

    [string]$Tag = "full",
    [switch]$NoExpl,
    [switch]$NoQual,
    [ValidateSet("gated","concat")][string]$Fusion = "gated",
    [int[]]$Seeds
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Route ALL HuggingFace downloads/loads into the project models/ dir
$env:HF_HOME            = Join-Path $PSScriptRoot "models"
$env:HF_HUB_CACHE       = Join-Path $env:HF_HOME  "hub"
$env:TRANSFORMERS_CACHE = $env:HF_HUB_CACHE

$args = @("-m","src.train","--dataset",$Dataset,"--ladder",$Ladder,"--tag",$Tag,"--fusion",$Fusion)
if ($NoExpl) { $args += "--no-expl" }
if ($NoQual) { $args += "--no-qual" }
if ($Seeds)  { $args += "--seeds"; $args += ($Seeds | ForEach-Object { "$_" }) }

Write-Host "python $($args -join ' ')" -ForegroundColor Cyan
& python @args
if ($LASTEXITCODE -ne 0) { throw "train.py failed with exit code $LASTEXITCODE" }
