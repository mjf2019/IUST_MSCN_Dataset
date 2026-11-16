param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('AF','RAL','AMCAL')]
    [string]$Method,
    [ValidateSet('Both','IAT','PS')]
    [string]$Dataset = 'Both'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$datasets = if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
Push-Location $repo
try {
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Existing staged changes found; commit or unstage them first' }
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed' }
    $folders = @()
    foreach ($name in $datasets) {
        $runs = Join-Path $PSScriptRoot "ANT_Burst_${name}_TEST/protocol_runs"
        if ($Method -eq 'AF') {
            $summary = Join-Path $runs 'AF_checkpoint20_five_seed_tables_v1'
            if (-not (Test-Path (Join-Path $summary 'Results/completed.json'))) { throw "Incomplete AF summary: $summary" }
            $folders += $summary
            foreach ($seed in @(42,48,54,62,72)) {
                $folder = Join-Path $runs "AF_checkpoint20_online_seed${seed}_v2"
                if (-not (Test-Path (Join-Path $folder 'evaluation_complete.json'))) { throw "Incomplete AF: $folder" }
                $folders += $folder
            }
        } elseif ($Method -eq 'RAL') {
            $folder = Join-Path $runs 'ral_full_paired_endpoints_v1'
            if (-not (Test-Path (Join-Path $folder 'Results/completed.json'))) { throw "Incomplete RAL: $folder" }
            $folders += $folder
        } else {
            $root = Join-Path $runs 'final_multiseed_fixedcnn_v1'
            $summary = Join-Path $root 'Results_independent_gain_eps0p05_v1'
            if (-not (Test-Path (Join-Path $summary 'completed.json'))) { throw "Incomplete independent Gain summary: $summary" }
            $folders += $summary
            foreach ($seed in @(42,48,54,62,72)) {
                $folder = Join-Path $root "seed$seed/independent_gain_eps0p05_v1"
                if (-not (Test-Path (Join-Path $folder 'completed.json'))) { throw "Incomplete independent Gain: $folder" }
                $folders += $folder
            }
        }
    }
    $count = 0
    foreach ($folder in $folders) {
        Get-ChildItem $folder -Recurse -File |
            Where-Object { $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]' } |
            ForEach-Object {
                git add -f -- $_.FullName
                if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
                $count++
            }
    }
    if ($count -eq 0) { throw 'No table-completion results found' }
    Write-Host "Staged $count result files"
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Complete Burst table metrics: $Method $Dataset"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged changes failed' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; results remain committed; do not force push' }
} finally {
    Pop-Location
}
