# FINAL_DEVIGN 7-col run - Train Devign L2 (code + explanation channel)
#   -> experiments\runs\final_devign_l2_cache\
# Text channel = same 7 explanation columns as final_reveal_l2/l3 (8-col set
# MINUS risk_level): confidence, risky_operations, missing_checks,
# function_name, called_functions, risky_apis, risk_summary (via --fields
# comma-list -> SEMVUL_EXPL_FIELDS; serialized by src/data_io.py).
# L2 - L1 = the explanation contribution. Seeds are configurable via -Seeds
# (default 1..5, matching final_devign_l1 for a paired L1->L2 delta; pass
# -Seeds 1 for a fast single-seed run -- Devign ensemble reportedly buys ~0).
# PURE INPUT: no apply_real_enrichment.py rebuild/de-anon step. Feeds
# ACTIVE\devign\{train,val}.jsonl exactly as they sit on disk (data.py already
# reads ACTIVE verbatim) -- fails loudly instead of silently regenerating them.
#
#   .\final_devign_l2.ps1                 # 5 seeds, batch 2 (this laptop)
#   .\final_devign_l2.ps1 -Batch512 4     # 5 seeds, >=16GB GPU
#   .\final_devign_l2.ps1 -Seeds 1        # single seed (fast)
# Resumable: a finished rung JSON is skipped.
param([int]$Batch512 = 2, [int[]]$Seeds = @(1, 2, 3, 4, 5))  # Batch512: 2 fits 8GB, 4 on >=16GB
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds come from the -Seeds param (default 1..5; see header).
# The 8-column text channel (7-col decisive set PLUS purpose at the end; no spaces).
$Cols = "confidence,risky_operations,missing_checks,function_name,called_functions,risky_apis,risk_summary,purpose"

# No enrichment/rebuild: just require ACTIVE\devign\{train,val}.jsonl as-is.
$active = Join-Path $PSScriptRoot "explanations\SemanticVul\ACTIVE\devign"
foreach ($f in @("train.jsonl", "val.jsonl")) {
    if (-not (Test-Path (Join-Path $active $f))) {
        throw "ACTIVE\devign\$f missing -- this script does not rebuild/enhance explanations; place the pure file there first."
    }
}

$seedArgs = $Seeds | ForEach-Object { "$_" }
# --fields $Cols: the 7-column decisive text channel (see $Cols above), same
# columns used for final_reveal_l2/l3 instead of the round-3 "prefix" channel.
# --cache-name final_devign_l2_cache: fresh, independent output dir under experiments\runs\.
# --code-enc codet5p: CodeT5+ (FuSEVul's encoder). --max-text 512: mirror reveal
# (both windows 512 for devign; heavier -> lower -Batch512 if VRAM-bound).
# --epochs 12: explicit. Devign is balanced -> no focal (train_rung auto-off).
& $py experiments\expl_enrich\reproduce_real.py --only devign --rungs L2 `
      --cache-name final_devign_l2_cache --seeds @seedArgs --batch512 $Batch512 --fields $Cols `
      --code-enc codet5p --max-text 512 --epochs 12
if ($LASTEXITCODE -ne 0) { Write-Warning "L2 exited $LASTEXITCODE (partial kept)" }
