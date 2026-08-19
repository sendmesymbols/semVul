function Test-CacheComplete {
    param(
        [Parameter(Mandatory)][string]$Dataset,
        [Parameter(Mandatory)][string]$Rung,
        [Parameter(Mandatory)][string]$CacheName,
        [Parameter(Mandatory)][int[]]$Seeds
    )

    foreach ($seed in $Seeds) {
        $seedDir = Join-Path (Get-Location) "experiments\runs\$CacheName\s$seed"
        $current = Join-Path $seedDir "semanticvul_${Dataset}_${Rung}.json"
        $legacy = Join-Path $seedDir "fusevul_ladder_${Dataset}_${Rung}.json"
        if (-not ((Test-Path -LiteralPath $current -PathType Leaf) -or
                  (Test-Path -LiteralPath $legacy -PathType Leaf))) {
            return $false
        }
    }
    return $true
}
