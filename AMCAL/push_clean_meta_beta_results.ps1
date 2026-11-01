param(
    [ValidateSet('All','SDNCampus','ISCX-TOR','IUST','IoT')]
    [string]$Dataset = 'All',
    [int]$Seed = 42,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$folders = @{
    'SDNCampus' = 'SDNCampus_TEST'
    'ISCX-TOR' = 'ISCX_TOR_TEST'
    'IUST' = 'IUST_MSCN_TEST'
    'IoT' = 'UNSW_IoT_TEST'
}
if ($Dataset -eq 'All' -and $Output) { throw 'Use one dataset with an explicit Output.' }
$items = if ($Dataset -eq 'All') { @('SDNCampus','ISCX-TOR','IUST','IoT') } else { @($Dataset) }
Push-Location $repo
try {
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Unrelated staged changes exist. Commit or unstage them first.' }
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed.' }
    $files = @()
    foreach ($item in $items) {
        $root = if ($Output) { $Output } else {
            Join-Path $repo "AMCAL/$($folders[$item])/protocol_runs/clean0_meta_beta_ddqn_seed${Seed}_v1"
        }
        if (-not (Test-Path $root)) { throw "Result directory missing: $root" }
        $files += @(Get-ChildItem $root -Recurse -File |
            Where-Object { $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]' })
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
        git commit -m "Add clean-only learned-beta DDQN results: $Dataset seed $Seed"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged results failed.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Results remain committed; do not force push.' }
} finally {
    Pop-Location
}
