param(
    [ValidateSet('Both','IAT','PS')]
    [string]$Dataset = 'Both',
    [int[]]$Seeds = @(42,48,54,62,72)
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$datasets = if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
$seedTag = 'seeds_' + ($Seeds -join '_')
Push-Location $repo
try {
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Existing staged changes found; commit or unstage them first' }
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed' }
    $folders = @()
    foreach ($name in $datasets) {
        $root = Join-Path $PSScriptRoot "ANT_Burst_${name}_TEST/protocol_runs/final_multiseed_fixedcnn_v1"
        $summary = Join-Path $root "Results_final_sequential_eps0p05_v1/$seedTag"
        if (-not (Test-Path (Join-Path $summary 'completed.json'))) {
            throw "Incomplete sequential summary: $summary"
        }
        $folders += $summary
        foreach ($seed in $Seeds) {
            $folder = Join-Path $root "seed$seed/final_sequential_eps0p05_v1"
            if (-not (Test-Path (Join-Path $folder 'completed.json'))) {
                throw "Incomplete sequential run: $folder"
            }
            $folders += $folder
        }
    }
    $count = 0
    foreach ($folder in $folders) {
        Get-ChildItem $folder -Recurse -File |
            Where-Object { $_.Extension -in '.json','.csv' } |
            ForEach-Object {
                git add -f -- $_.FullName
                if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
                $count++
            }
    }
    if ($count -eq 0) { throw 'No sequential result files found' }
    Write-Host "Staged $count sequential result files; local resume checkpoints are excluded"
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add final AMCAL sequential retention results: $Dataset"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged changes failed' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; results remain committed; do not force push' }
} finally {
    Pop-Location
}
