# DEPRECATED SHIM -- kept so existing docs and habits keep working.
#
# This used to be a separate, hand-run step you had to remember AFTER a run:
# copy the four canonical .real files into explanations\SemanticVul\ACTIVE\,
# write the manifest, then build explanation.prefix. Forgetting it left ACTIVE
# stale, or without explanation.prefix -- which makes reproduce_real.py hard-exit,
# since its --fields default is 'prefix'.
#
# That is now stages 5-6 of the single pipeline, so it can no longer be skipped:
#
#   .\generate_explanations.ps1                          # all six stages
#   .\generate_explanations.ps1 -FromStage 5 -Promote    # exactly what this does
#
# Running this file forwards to that, against the SHIPPED tree (-Promote), which
# is the behaviour this script always had.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Write-Warning ("organize_explanations.ps1 is deprecated: it is now stages 5-6 of " +
               "generate_explanations.ps1. Forwarding to " +
               "'.\generate_explanations.ps1 -FromStage 5 -Promote'.")
& (Join-Path $PSScriptRoot "generate_explanations.ps1") -FromStage 5 -Promote @args
exit $LASTEXITCODE
