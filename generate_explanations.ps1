# GENERATE EXPLANATIONS - THE single entry point for the explanation dataset.
#
# Runs all six stages and ends with a complete ACTIVE/ pair per dataset, carrying
# every explanation.* column the training wrappers read:
#
#   1 generate    purpose, data_flow, risky_operations, missing_checks,
#                 evidence_tokens, safety_indicators, risk_summary, risk_level,
#                 confidence  (MEASURED from decode-time logprobs, not self-reported)
#   2 install     stage-1 output -> the filename later stages read
#   3 enrich      llm_v1, code_metrics, tail_facts, enrich
#   4 clean/aug   the .clean / .clean.aug variants ACTIVE is built from
#   5 real        function_name, called_functions, risky_apis, string_literals,
#                 lexical_digest, real_enrich, tail_digest
#   6 prefix      prefix, prefix_recipe  + ACTIVE/README.md
#
# organize_explanations.ps1 is NOT needed any more -- it was stages 5/6 by hand.
#
# By default it builds into experiments\explanation\work\ and does NOT touch the
# shipped explanations\SemanticVul\. Pass -Promote to build in the shipped tree.
#
#   .\generate_explanations.ps1 -Smoke                  # 6 rows/split, end-to-end proof
#   .\generate_explanations.ps1                         # FULL both datasets (see note)
#   .\generate_explanations.ps1 -Dataset reveal -Split val
#   .\generate_explanations.ps1 -Stratified 300 -Workers 4
#   .\generate_explanations.ps1 -FromStage 2            # reuse stage-1 output
#   .\generate_explanations.ps1 -Promote                # OVERWRITES shipped data
#
# RUNTIME: ~70 s/sample on qwen2.5-coder:14b. All four splits = 70,802 rows
# ~= 57 days sequential, ~7 days at -Workers 8. Use -Stratified/-Smoke to prove
# the pipeline, then scale up. Resumable: re-running skips finished rows.
param(
    [ValidateSet("devign", "reveal", "both")][string]$Dataset = "both",
    [ValidateSet("train", "val", "both")][string]$Split = "both",
    [string]$Model = "qwen2.5-coder:14b",
    [string]$OllamaHost = "http://localhost:9999",
    [ValidateSet("auto", "anon", "real")][string]$Mode = "auto",
    [int]$Stratified = 0,
    [int]$Workers = 1,
    [int]$NumCtx = 8192,
    [int]$Timeout = 600,
    [string]$Tag = "",
    [string]$WorkDir = "",
    [ValidateRange(1, 6)][int]$FromStage = 1,
    [ValidateRange(1, 6)][int]$ToStage = 6,
    [int]$AugCopies = 1,
    [int]$TailOffset = 220,
    [switch]$NoThink,
    [switch]$Smoke,
    [switch]$Promote
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No venv python at $py -- create one (python -m venv .venv)" }

# Preflight the model only when stage 1 will actually run.
if ($FromStage -le 1) {
    Write-Host "[gen] checking Ollama at $OllamaHost ..."
    try { $tags = Invoke-RestMethod -Uri "$OllamaHost/api/tags" -TimeoutSec 20 }
    catch { throw "Cannot reach Ollama at $OllamaHost -- is the container up? ($($_.Exception.Message))" }
    if ($tags.models.name -notcontains $Model) {
        throw "Model '$Model' not on the server. Available: $($tags.models.name -join ', ')"
    }
    Write-Host "[gen] Ollama OK, model '$Model' present."
}

$a = @("--dataset", $Dataset, "--split", $Split, "--model", $Model,
       "--host", $OllamaHost, "--mode", $Mode, "--workers", $Workers,
       "--num-ctx", $NumCtx, "--timeout", $Timeout,
       "--from-stage", $FromStage, "--to-stage", $ToStage,
       "--aug-copies", $AugCopies, "--tail-offset", $TailOffset)
if ($Stratified -gt 0) { $a += @("--stratified", $Stratified) }
if ($Tag)              { $a += @("--tag", $Tag) }
if ($WorkDir)          { $a += @("--work-dir", $WorkDir) }
if ($NoThink)          { $a += "--no-think" }
if ($Smoke)            { $a += "--smoke" }
if ($Promote)          { $a += "--promote" }

& $py (Join-Path $PSScriptRoot "experiments\explanation\pipeline.py") @a
exit $LASTEXITCODE
