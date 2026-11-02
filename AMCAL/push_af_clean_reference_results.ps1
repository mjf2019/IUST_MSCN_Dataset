param(
    [ValidateSet('All','SDNCampus','ISCX-TOR','IUST','IoT','IAT','PS')]
    [string]$Dataset = 'All',
    [int]$Seed = 42
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$folders = @{
    'SDNCampus'='SDNCampus_TEST'; 'ISCX-TOR'='ISCX_TOR_TEST'
    'IUST'='IUST_MSCN_TEST'; 'IoT'='UNSW_IoT_TEST'
    'IAT'='ANT_Burst_IAT_TEST'; 'PS'='ANT_Burst_PS_TEST'
}
$datasets = @($Dataset)
if ($Dataset -eq 'All') { $datasets = @('SDNCampus','ISCX-TOR','IUST','IoT') }
Push-Location $repo
try {
    $branch = git branch --show-current
    if ($LASTEXITCODE -ne 0 -or $branch -ne 'experiment/paironly-paid-replay') {
        throw 'Switch to experiment/paironly-paid-replay first.'
    }
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Other changes are already staged; commit or unstage them first.' }
    $files = @()
    foreach ($name in $datasets) {
        $root = "AMCAL/$($folders[$name])/protocol_runs/AF_checkpoint20_online_seed${Seed}_v2"
        if (-not (Test-Path "$root/evaluation_complete.json")) {
            throw "Completed reference audit not found: $name"
        }
        $files += @(Get-ChildItem $root -Recurse -File |
            Where-Object { $_.Extension -in '.json','.csv' } |
            ForEach-Object { $_.FullName })
    }
    if ($files.Count -eq 0) { throw 'No AF reference result files found.' }
    foreach ($file in $files) {
        git add -f -- $file
        if ($LASTEXITCODE -ne 0) { throw 'Adding AF reference results failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $difference = $LASTEXITCODE
    if ($difference -eq 1) {
        git commit -m "Add AF pre-feedback online and checkpoint retention results: $Dataset seed$Seed"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed; results remain staged.' }
    } elseif ($difference -ne 0) { throw 'Checking staged results failed.' }
    git pull --rebase --autostash origin experiment/paironly-paid-replay
    if ($LASTEXITCODE -ne 0) { throw 'Synchronization failed; results remain committed. Do not force push.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; results remain committed. Do not force push.' }
} finally { Pop-Location }
