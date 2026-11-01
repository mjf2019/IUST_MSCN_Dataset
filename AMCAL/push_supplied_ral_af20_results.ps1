param(
    [ValidateSet('Both','RAL','AF')][string]$Method = 'Both',
    [ValidateSet('All','SDNCampus','ISCX-TOR','IUST','IoT')][string]$Dataset = 'All',
    [int]$Seed = 42
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$folders = @{'SDNCampus'='SDNCampus_TEST';'ISCX-TOR'='ISCX_TOR_TEST';'IUST'='IUST_MSCN_TEST';'IoT'='UNSW_IoT_TEST'}
$tags = @{'RAL'='RAL_full20';'AF'='AF_author_full20'}
$methods = if ($Method -eq 'Both') { @('RAL','AF') } else { @($Method) }
$datasets = if ($Dataset -eq 'All') { @('SDNCampus','ISCX-TOR','IUST','IoT') } else { @($Dataset) }
Push-Location $repo
try {
    $branch = git branch --show-current
    if ($LASTEXITCODE -ne 0 -or $branch -ne 'experiment/paironly-paid-replay') {
        throw 'Switch to experiment/paironly-paid-replay before exporting these results.'
    }
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Unrelated staged changes exist. Commit or unstage them first.' }
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed.' }
    $files = @()
    foreach ($item in $datasets) {
        foreach ($methodItem in $methods) {
            $root = Join-Path $repo "AMCAL/$($folders[$item])/protocol_runs/clean0_$($tags[$methodItem])_seed${Seed}_v1"
            if (-not (Test-Path $root)) { throw "Result directory missing: $root" }
            $files += @(Get-ChildItem $root -Recurse -File |
                Where-Object { $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]' })
        }
    }
    if ($files.Count -eq 0) { throw 'No result JSON/CSV files found.' }
    foreach ($file in $files) {
        Write-Host $file.FullName
        git add -f -- $file.FullName
        if ($LASTEXITCODE -ne 0) { throw 'Adding result file failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add matched clean0 full RAL / author AF20 results: $Dataset / $Method seed $Seed"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged results failed.' }
    git -c pull.ff=false pull --rebase --autostash origin experiment/paironly-paid-replay
    if ($LASTEXITCODE -ne 0) { throw 'Synchronization failed. Results are preserved; resolve the Git error before pushing.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Results remain committed; do not force push.' }
} finally { Pop-Location }
