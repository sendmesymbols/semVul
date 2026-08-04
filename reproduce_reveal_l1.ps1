# Train ReVeal L1 (code-only baseline) -> experiments\runs\l1_reveal_cache\
# ReVeal treated arm: text channel = round-3 materialized explanation.prefix
# (--fields prefix, this launcher), focal loss
# (alpha/gamma below). 320-token code window. L1 is code-only (no explanation), so the
# channel/focal knobs only bind at L2/L3 — kept here for a uniform launcher.
#
#   .\reproduce_reveal_l1.ps1                  # seed 1337
#   .\reproduce_reveal_l1.ps1 -Seeds 1337,2024
# Resumable: a finished rung JSON is skipped.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
# Seeds are HARDCODED for a consistent ladder across L1/L2/L3. Edit this one line to change.
$Seeds = @(1, 3, 7)

# ---- ReVeal treatment knobs (HARDCODED; no env vars) ----
$TailOffset = 220     # code-token offset ~= 320 GraphCodeBERT subwords (window)
$FocalAlpha = 0.85    # focal positive weight (baseline auto ~0.80)
$FocalGamma = 2.0
# ---------------------------------------------------------

# Self-contained: if ACTIVE\reveal\{train,val}.jsonl exist we do NOT touch the
# enriched source files (ACTIVE already carries tail_digest). Only build when
# ACTIVE is absent. To change $TailOffset, delete explanations\SemanticVul\
# ACTIVE\reveal\ first so this rebuilds with the new offset.
& $py experiments\expl_enrich\apply_real_enrichment.py --check --only reveal
if ($LASTEXITCODE -ne 0) {
    Write-Host "ACTIVE/reveal missing -> building from sources (tail-offset $TailOffset)..."
    & $py experiments\expl_enrich\apply_real_enrichment.py --only reveal --tail-offset $TailOffset
    if ($LASTEXITCODE -ne 0) { throw "apply_real_enrichment (reveal) failed" }
}

$seedArgs = $Seeds | ForEach-Object { "$_" }
# NOTE (2026-07-15): ACTIVE now carries the round-3 data (train 21,695 incl.
# tagged oversampled dups that _dedup_train collapses back to ~17.3k denoised
# rows; val 2,269). L1 is text-channel-invariant but NOT data-invariant, so the
# old l1_reveal_* caches do NOT transfer — this cache name forces a retrain.
# --max-text 320 kept uniform with L2/L3 (L1's model ignores the text channel).
& $py experiments\expl_enrich\reproduce_real.py --only reveal --rungs L1 `
      --cache-name l1_reveal_prefix3 --seeds @seedArgs --fields prefix `
      --max-text 320 `
      --focal-alpha $FocalAlpha --focal-gamma $FocalGamma
if ($LASTEXITCODE -ne 0) { Write-Warning "L1 exited $LASTEXITCODE (partial kept)" }
