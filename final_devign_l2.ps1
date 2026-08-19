# FINAL_DEVIGN clean-Qwen run - Train Devign L2 (code + explanation channel)
#   -> experiments\runs\final_devign_l2_cache\
# Text channel = the validated Qwen fields listed in $Cols below.
# L2 - L1 = the explanation contribution. Seeds are configurable via -Seeds
# (default 1..5, matching final_devign_l1 for a paired L1->L2 delta; pass
# -Seeds 1 for a fast single-seed run -- Devign ensemble reportedly buys ~0).
# PURE INPUT: no post-generation enrichment or identifier recovery. Feeds
# ACTIVE\devign\{train,val}.jsonl exactly as they sit on disk (data.py already
# reads ACTIVE verbatim) -- fails loudly instead of silently regenerating them.
#
#   .\final_devign_l2.ps1                 # 5 seeds, batch 2 (this laptop)
#   .\final_devign_l2.ps1 -Batch512 4     # 5 seeds, >=16GB GPU
#   .\final_devign_l2.ps1 -Seeds 1        # single seed (fast)
# Resumable: a finished rung JSON is skipped.
param([int]$Batch512 = 2, [int[]]$Seeds = @(1, 2, 3, 4, 5), [switch]$CleanQwen)  # Batch512: 2 fits 8GB, 4 on >=16GB
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "scripts\cache_complete.ps1")
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds come from the -Seeds param (default 1..5; see header).
if (Test-CacheComplete -Dataset devign -Rung L2 -CacheName final_devign_l2_cache -Seeds $Seeds) {
    Write-Host "[cache] final_devign_l2_cache is complete; skipping validation and training."
    exit 0
}
# The clean Qwen-only explanation channel (no spaces).
$Cols = "confidence,purpose,data_flow,risky_operations,missing_checks,evidence_tokens,safety_indicators,risk_summary"

if ($CleanQwen) {
    Remove-Item Env:SEMVUL_LEGACY_CACHE -ErrorAction SilentlyContinue
    & $py experiments\explanation\validate_clean.py --dataset devign
    if ($LASTEXITCODE -ne 0) { throw "ACTIVE/devign is missing or contains legacy enriched inputs" }
} else {
    $env:SEMVUL_LEGACY_CACHE = "1"
}

$seedArgs = $Seeds | ForEach-Object { "$_" }
# --fields $Cols: the same clean Qwen-only channel used by the Reveal runs.
# --cache-name final_devign_l2_cache: canonical cache; the driver verifies provenance.
# --code-enc codet5p: CodeT5+ (FuSEVul's encoder). --max-text 512: mirror reveal
# (both windows 512 for devign; heavier -> lower -Batch512 if VRAM-bound).
# --epochs 12: explicit. Devign is balanced -> no focal (train_rung auto-off).
& $py experiments\expl_enrich\reproduce_real.py --only devign --rungs L2 `
      --cache-name final_devign_l2_cache --seeds @seedArgs --batch512 $Batch512 --fields $Cols `
      --code-enc codet5p --max-text 512 --epochs 12
if ($LASTEXITCODE -ne 0) { Write-Warning "L2 exited $LASTEXITCODE (partial kept)" }
