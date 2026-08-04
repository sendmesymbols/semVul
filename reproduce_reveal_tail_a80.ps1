# ISOLATED tail_digest arm (Lever A only) — one-variable A/B to attribute the
# beyond-window tail field on its own. tail_digest ON, focal at BASELINE
# (alpha auto=0.80, gamma=2.0 -> NO --focal-* flags, which is numerically
# identical to the overnight baseline's focal). The ONLY difference vs
# runs/enriched_real is the tail_digest text field.
#
# L2 / seed 1337 = fast attribution (where the +1.39 lives). Widen with
# -Seeds 1337,2024 or edit --rungs to expand once the isolated lift is known.
#
#   .\reproduce_reveal_tail_a80.ps1
#
# Queue AFTER reproduce_reveal.ps1 (the _tail_a85 arm) finishes — sequential,
# shares the one GPU. Baseline member (runs/enriched_real/s1337) must already
# exist from the overnight run; that member carries run-to-run noise, so read
# the paired-bootstrap CI, not the point delta.
param(
    [int[]]$Seeds = @(1337)
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

$TailOffset = 220          # same window offset as the main run (keep A/B comparable)
$OutTag     = "_tail_a80"  # -> runs/enriched_real_tail_a80/

# 1. ensure ReVeal *.real.jsonl carry tail_digest (idempotent; --only reveal is Devign-safe)
& $py experiments\expl_enrich\apply_real_enrichment.py --only reveal --tail-offset $TailOffset
if ($LASTEXITCODE -ne 0) { throw "apply_real_enrichment (reveal) failed" }

# 2. train ISOLATED tail arm: --tail-digest ON, NO focal override -> baseline focal (a=0.80, g=2.0)
$seedArgs = $Seeds | ForEach-Object { "$_" }
& $py experiments\expl_enrich\reproduce_real.py --only reveal --seeds @seedArgs `
      --rungs L2 --tail-digest --out-tag $OutTag
if ($LASTEXITCODE -ne 0) { Write-Warning "reproduce_real exited $LASTEXITCODE (partial results kept)" }

# 3. attribution readouts
& $py experiments\expl_enrich\dual_eval.py
Write-Host "`n[tail lever, ISOLATED]  base enriched_real -> treat enriched_real_tail_a80"
& $py experiments\expl_enrich\paired_bootstrap.py `
      --base-sub enriched_real --treat-sub "enriched_real$OutTag" `
      --ds reveal --rung L2 --seed 1337
# Bonus: focal lever given tail (needs the _tail_a85 arm present; non-fatal if absent).
Write-Host "`n[focal lever, given tail]  base enriched_real_tail_a80 -> treat enriched_real_tail_a85"
& $py experiments\expl_enrich\paired_bootstrap.py `
      --base-sub enriched_real_tail_a80 --treat-sub enriched_real_tail_a85 `
      --ds reveal --rung L2 --seed 1337
