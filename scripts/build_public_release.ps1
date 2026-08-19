[CmdletBinding()]
param(
    [ValidateSet("Code", "Reviewer")]
    [string]$Profile = "Code",
    [string]$Destination = "release/SemanticVul-public",
    [switch]$KeepIdentities,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destinationPath = if ([IO.Path]::IsPathRooted($Destination)) {
    [IO.Path]::GetFullPath($Destination)
} else {
    [IO.Path]::GetFullPath((Join-Path $repo $Destination))
}

if (-not $destinationPath.StartsWith($repo + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination must be inside the repository: $destinationPath"
}
if ($destinationPath -eq $repo) {
    throw "Destination cannot be the repository root"
}
if (Test-Path -LiteralPath $destinationPath) {
    if (-not $Force) {
        throw "Destination exists. Choose another path or pass -Force: $destinationPath"
    }
    Remove-Item -LiteralPath $destinationPath -Recurse -Force
}
New-Item -ItemType Directory -Path $destinationPath | Out-Null

function Get-ReleaseRelativePath {
    param([Parameter(Mandatory)][string]$Path)
    $base = $destinationPath.TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar) +
        [IO.Path]::DirectorySeparatorChar
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($base, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the release directory: $full"
    }
    return $full.Substring($base.Length).Replace('\', '/')
}

function Copy-ReleaseFile {
    param([Parameter(Mandatory)][string]$RelativePath)
    $source = Join-Path $repo $RelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required release file is missing: $RelativePath"
    }
    $target = Join-Path $destinationPath $RelativePath
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $target
}

function Copy-ReleaseTree {
    param([Parameter(Mandatory)][string]$RelativePath)
    $source = Join-Path $repo $RelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Required release directory is missing: $RelativePath"
    }
    $target = Join-Path $destinationPath $RelativePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Recurse
}

function Write-ReleaseText {
    param(
        [Parameter(Mandatory)][string]$RelativePath,
        [Parameter(Mandatory)][string]$Text
    )
    $target = Join-Path $destinationPath $RelativePath
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    [IO.File]::WriteAllText($target, $Text, [Text.UTF8Encoding]::new($false))
}

$files = @(
    ".gitattributes",
    ".gitignore",
    "requirements.txt",
    "PUBLIC_RELEASE.md",
    "run_final_sequence.ps1",
    "run_final_sequence.sh",
    "generate_explanations.ps1",
    "generate_explanations.sh",
    "organize_explanations.ps1",
    "scripts/cache_complete.ps1",
    "scripts/cache_complete.sh",
    "final_devign_l1.ps1", "final_devign_l1.sh",
    "final_devign_l2.ps1", "final_devign_l2.sh",
    "final_devign_l3.ps1", "final_devign_l3.sh",
    "final_reveal_l1.ps1", "final_reveal_l1.sh",
    "final_reveal_l2.ps1", "final_reveal_l2.sh",
    "final_reveal_l3.ps1", "final_reveal_l3.sh",
    "experiments/explanation/generate.py",
    "experiments/explanation/pipeline.py",
    "experiments/explanation/prompt.py",
    "experiments/explanation/validate_clean.py",
    "experiments/expl_enrich/reproduce_real.py",
    "experiments/fusevul_ladder/data.py",
    "experiments/fusevul_ladder/model.py",
    "experiments/fusevul_ladder/train.py",
    "src/__init__.py",
    "src/config.py",
    "src/data_io.py",
    "src/quality_features.py",
    "src/quality_features_v2.py",
    "src/rqs/aggregate_seeds.py",
    "src/rqs/plots.py",
    "src/rqs/rq1.py",
    "src/rqs/rq2.py",
    "src/rqs/rq2_oracle_gate.py",
    "src/rqs/rq3.py",
    "src/rqs/rq4.py"
)
foreach ($file in $files) { Copy-ReleaseFile $file }

# Copy a public README. Anonymous mode removes reviewer-identifying acknowledgments.
$readme = Get-Content -LiteralPath (Join-Path $repo "README.md") -Raw
if (-not $KeepIdentities) {
    $readme = [regex]::Replace(
        $readme,
        '(?ms)^## Acknowledgments\s*.*\z',
        "## Acknowledgments`r`n`r`nWithheld during anonymous review.`r`n"
    )
}
$profileNotice = if ($Profile -eq "Code") {
    "> **Artifact profile:** Code-only public release. Large datasets, generated explanations, and result caches referenced below are intentionally not included. Use the release policy for the reviewer-artifact profile.`r`n`r`n"
} else {
    "> **Artifact profile:** Reviewer reproduction artifact. Dataset and generated-artifact redistribution remains subject to their respective licenses and the venue's policy.`r`n`r`n"
}
$readme = $profileNotice + $readme
[IO.File]::WriteAllText((Join-Path $destinationPath "README.md"), $readme,
    [Text.UTF8Encoding]::new($false))

if ($Profile -eq "Reviewer") {
    Copy-ReleaseTree "data"
    foreach ($dataset in @("devign", "reveal")) {
        foreach ($split in @("train", "val")) {
            Copy-ReleaseFile "explanations/SemanticVul/ACTIVE/$dataset/$split.jsonl"
        }
    }
    Get-ChildItem -LiteralPath (Join-Path $repo "experiments/runs") -Directory |
        Where-Object { $_.Name -match '^final_(devign|reveal)_l[123]_cache$' } |
        ForEach-Object { Copy-ReleaseTree "experiments/runs/$($_.Name)" }
    if (Test-Path -LiteralPath (Join-Path $repo "reports/plots")) {
        Copy-ReleaseTree "reports/plots"
    }
} else {
    Write-ReleaseText "data/README.md" @"
# Data

The minimal code profile intentionally excludes dataset CSV files. Use the
Reviewer profile only when the venue and upstream dataset licenses permit
redistribution.
"@
    Write-ReleaseText "explanations/SemanticVul/ACTIVE/README.md" @"
# ACTIVE Explanations

The minimal code profile intentionally excludes generated explanation JSONL
files. Regenerate them with the documented generation pipeline, or use the
Reviewer profile only when redistribution is permitted.
"@
    Write-ReleaseText "experiments/runs/README.md" @"
# Runs

The minimal code profile intentionally excludes generated result caches. Rerun
the documented launchers to create fresh caches, or use the Reviewer profile only
when redistribution is permitted.
"@
    Write-ReleaseText "reports/plots/README.md" @"
# Plots

The minimal code profile intentionally excludes generated figures. Regenerate
them from result caches with `python src/rqs/plots.py both --cache-prefix final`.
"@
}

if (-not $KeepIdentities) {
    $identityPattern = '(?i)(Ihtesham Ul Islam|Rabia Khan|Muhammad Sohail|Nazia Bibi|Military College of Signals|NUST)'
    $identityHits = Get-ChildItem -LiteralPath $destinationPath -Recurse -File |
        Where-Object { $_.Extension -in @(".md", ".txt", ".py", ".ps1", ".sh", ".json", ".jsonl", ".csv") } |
        Select-String -Pattern $identityPattern
    if ($identityHits) {
        $paths = $identityHits |
            ForEach-Object { Get-ReleaseRelativePath $_.Path } |
            Sort-Object -Unique
        throw "Anonymous release contains identifying text in: $($paths -join ', ')"
    }
}

$manifest = Get-ChildItem -LiteralPath $destinationPath -Recurse -File |
    ForEach-Object {
        [pscustomobject]@{
            Path = Get-ReleaseRelativePath $_.FullName
            Bytes = $_.Length
            SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    } | Sort-Object Path
$manifest | ConvertTo-Json -Depth 3 |
    Set-Content -LiteralPath (Join-Path $destinationPath "MANIFEST.json") -Encoding utf8

$totalBytes = ($manifest | Measure-Object -Property Bytes -Sum).Sum
Write-Host "Built $Profile release at $destinationPath"
Write-Host ("Files: {0}; size: {1:N2} MiB" -f $manifest.Count, ($totalBytes / 1MB))
Write-Host "Identities in README: $([bool]$KeepIdentities)"
Write-Warning "No LICENSE file exists. Resolve licensing before publication."
