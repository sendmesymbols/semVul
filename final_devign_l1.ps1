# FINAL_DEVIGN run - Train Devign L1 (code-only baseline)
#   -> experiments\runs\final_devign_l1_cache\
# Code channel = CodeT5+ (Salesforce/codet5p-110m-embedding), FuSEVul's encoder
# (loaded via the is_decoder shim in train.py). Text channel = RoBERTa. L1 is
# code-only, so the text channel/fields do NOT bind here -- kept for a uniform
# launcher and a matched config across the final_devign_l{1,2,3} ladder.
# L2 - L1 = the explanation contribution. 5 seeds, 12 epochs, max_code 512 +
# max_text 512 (mirrors reveal / devign L2/L3).
# PURE INPUT: no apply_real_enrichment.py rebuild/de-anon step. Feeds
# ACTIVE\devign\{train,val}.jsonl exactly as they sit on disk.
#
#   .\final_devign_l1.ps1                 # 5 seeds, batch 2
#   .\final_devign_l1.ps1 -Batch512 4     # >=16GB GPU
# Resumable: a finished rung JSON is skipped.
param([int]$Batch512 = 2)  # 2 fits 8GB; use 4 on >=16GB
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds HARDCODED: 5 seeds (matching final_reveal + final_devign_l2/l3).
$Seeds = @(1, 2, 3, 4, 5)
# The 8-column text channel (7-col decisive set PLUS purpose at the end; no spaces).
# L1 ignores the text channel; kept identical to L2/L3 for a uniform launcher.
$Cols = "confidence,risky_operations,missing_checks,function_name,called_functions,risky_apis,risk_summary,purpose"

# No enrichment/rebuild: just require ACTIVE\devign\{train,val}.jsonl as-is.
$active = Join-Path $PSScriptRoot "explanations\SemanticVul\ACTIVE\devign"
foreach ($f in @("train.jsonl", "val.jsonl")) {
    if (-not (Test-Path (Join-Path $active $f))) {
        throw "ACTIVE\devign\$f missing -- this script does not rebuild/enhance explanations; place the pure file there first."
    }
}

$seedArgs = $Seeds | ForEach-Object { "$_" }
# --code-enc codet5p: CodeT5+ (FuSEVul's encoder). --max-text 512: mirror reveal
# (both windows 512 for devign). --epochs 12: explicit. Devign balanced -> no focal.
# --cache-name final_devign_l1_cache: fresh, independent output dir under experiments\runs\.
& $py experiments\expl_enrich\reproduce_real.py --only devign --rungs L1 `
      --cache-name final_devign_l1_cache --seeds @seedArgs --batch512 $Batch512 --fields $Cols `
      --code-enc codet5p --max-text 512 --epochs 12
if ($LASTEXITCODE -ne 0) { Write-Warning "L1 exited $LASTEXITCODE (partial kept)" }
