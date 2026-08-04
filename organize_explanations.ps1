# Consolidate the explanation files ACTUALLY USED by the current runs into one
# clear folder, WITHOUT deleting or moving anything. Pure copy + manifest.
#
# The pipeline scatters many intermediate .jsonl files (.clean, .enriched.clean,
# .real, .aug, devign_real\, full_code\ ...). Only TWO per dataset actually feed a
# run (verified against src\data_io.py + experiments\fusevul_ladder\data.py +
# experiments\expl_enrich\reproduce_real.py). This copies those four into
# explanations\SemanticVul\ACTIVE\ with simple names so you can see, at a glance,
# exactly what a run consumes. Originals stay untouched (the pipeline still uses
# the long names).
#
#   .\organize_explanations.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$root = Join-Path $PSScriptRoot "explanations\SemanticVul"
$active = Join-Path $root "ACTIVE"
New-Item -ItemType Directory -Force -Path (Join-Path $active "devign") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $active "reveal") | Out-Null

# (source relative to explanations\SemanticVul, dest under ACTIVE)
$map = @(
  @{ src = "devign\devign_train.enriched.clean.aug.real.jsonl"; dst = "devign\train.jsonl" },
  @{ src = "devign\devign_val.enriched.real.jsonl";             dst = "devign\val.jsonl"   },
  @{ src = "reveal\reveal_train.enriched.clean.real.jsonl";     dst = "reveal\train.jsonl" },
  @{ src = "reveal\reveal_val.enriched.real.jsonl";             dst = "reveal\val.jsonl"   }
)
foreach ($m in $map) {
  $s = Join-Path $root $m.src
  $d = Join-Path $active $m.dst
  if (-not (Test-Path $s)) { Write-Warning "MISSING source: $s"; continue }
  Copy-Item -Path $s -Destination $d -Force
  $mb = [math]::Round((Get-Item $d).Length / 1MB, 1)
  Write-Host ("copied {0,-55} -> ACTIVE\{1}  ({2} MB)" -f $m.src, $m.dst, $mb)
}

$manifest = @"
# ACTIVE explanations — the ONLY files a run consumes

Copied by organize_explanations.ps1 (non-destructive; originals untouched).
Each dataset uses exactly two files. Do not train from the files in this folder
directly — they are a readable snapshot; the pipeline reads the long-named
originals via env-var resolution in reproduce_real.py.

| dataset | role  | ACTIVE copy        | real source (what the run reads)                     |
|---------|-------|--------------------|------------------------------------------------------|
| devign  | train | devign\train.jsonl | devign_train.enriched.clean.aug.real.jsonl           |
| devign  | val   | devign\val.jsonl   | devign_val.enriched.real.jsonl                       |
| reveal  | train | reveal\train.jsonl | reveal_train.enriched.clean.real.jsonl               |
| reveal  | val   | reveal\val.jsonl   | reveal_val.enriched.real.jsonl                       |

Devign: explanation text = de-anonymized fields + evidence_tokens + lexical_digest.
        raw_code stays the anon benchmark code (comparable to FuSEVul). val = full
        benchmark val (2732 rows, ~67% carry de-anon/digest, rest pass through).
ReVeal: real code already; explanation = CORE fields + lexical_digest + tail_digest.
        NOTE: the ReVeal .real files are REGENERATED with tail_digest every run by
        'apply_real_enrichment.py --only reveal'; re-run this script after a ReVeal
        run to refresh the snapshot.

Everything else under devign\ and reveal\ (.clean, .enriched.clean, plain .real,
devign_real\, full_code\) is intermediate/source and is NOT read at train time.
"@
Set-Content -Path (Join-Path $active "README.md") -Value $manifest -Encoding UTF8
Write-Host "`nwrote ACTIVE\README.md"
Write-Host "ACTIVE folder ready: $active"
