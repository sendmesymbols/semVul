# FINAL_DEVIGN run - Train Devign L1 (code-only baseline)
#   -> experiments\runs\final_devign_l1_cache\
# Code channel = CodeT5+ (Salesforce/codet5p-110m-embedding), FuSEVul's encoder
# (loaded via the is_decoder shim in train.py). Text channel = RoBERTa. L1 is
# code-only, so the text channel/fields do NOT bind here -- kept for a uniform
# launcher and a matched config across the final_devign_l{1,2,3} ladder.
# L2 - L1 = the explanation contribution. 5 seeds, 12 epochs, max_code 512 +
# max_text 512 (mirrors reveal / devign L2/L3).
# PURE INPUT: no post-generation enrichment or identifier recovery. Feeds
# ACTIVE\devign\{train,val}.jsonl exactly as they sit on disk.
#
#   .\final_devign_l1.ps1                 # 5 seeds, batch 2
#   .\final_devign_l1.ps1 -Batch512 4     # >=16GB GPU
# Resumable: a finished rung JSON is skipped.
param([int]$Batch512 = 2, [switch]$CleanQwen)  # 2 fits 8GB; use 4 on >=16GB
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "scripts\cache_complete.ps1")
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds HARDCODED: 5 seeds (matching final_reveal + final_devign_l2/l3).
$Seeds = @(1, 2, 3, 4, 5)
if (Test-CacheComplete -Dataset devign -Rung L1 -CacheName final_devign_l1_cache -Seeds $Seeds) {
    Write-Host "[cache] final_devign_l1_cache is complete; skipping validation and training."
    exit 0
}
# The clean Qwen-only explanation channel (unused by L1, retained for parity).
# L1 ignores the text channel; kept identical to L2/L3 for a uniform launcher.
$Cols = "confidence,purpose,data_flow,risky_operations,missing_checks,evidence_tokens,safety_indicators,risk_summary"

if ($CleanQwen) {
    Remove-Item Env:SEMVUL_LEGACY_CACHE -ErrorAction SilentlyContinue
    & $py experiments\explanation\validate_clean.py --dataset devign
    if ($LASTEXITCODE -ne 0) { throw "ACTIVE/devign is missing or contains legacy enriched inputs" }
} else {
    $env:SEMVUL_LEGACY_CACHE = "1"
}

$seedArgs = $Seeds | ForEach-Object { "$_" }
# --code-enc codet5p: CodeT5+ (FuSEVul's encoder). --max-text 512: mirror reveal
# (both windows 512 for devign). --epochs 12: explicit. Devign balanced -> no focal.
# --cache-name final_devign_l1_cache: canonical cache; the driver verifies provenance.
& $py experiments\expl_enrich\reproduce_real.py --only devign --rungs L1 `
      --cache-name final_devign_l1_cache --seeds @seedArgs --batch512 $Batch512 --fields $Cols `
      --code-enc codet5p --max-text 512 --epochs 12
if ($LASTEXITCODE -ne 0) { Write-Warning "L1 exited $LASTEXITCODE (partial kept)" }
