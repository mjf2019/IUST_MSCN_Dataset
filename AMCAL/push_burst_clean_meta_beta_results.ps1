param(
    [ValidateSet('Both','IAT','PS')][string]$Dataset = 'Both',
    [int]$Seed = 42
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Unrelated staged changes exist. Commit or unstage them first.' }
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed.' }
    $items = if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
    $count = 0
    foreach ($item in $items) {
        $root = Join-Path $repo "AMCAL/ANT_Burst_${item}_TEST/protocol_runs/clean0_light_meta_beta_ddqn_seed${Seed}_v1"
        if (-not (Test-Path $root)) { throw "Result directory missing: $root" }
        Get-ChildItem $root -Recurse -File |
            Where-Object { $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]' } |
            ForEach-Object {
                Write-Host $_.FullName
                git add -f -- $_.FullName
                if ($LASTEXITCODE -ne 0) { throw 'Adding results failed.' }
                $count++
            }
    }
    if ($count -eq 0) { throw 'No result JSON/CSV files found.' }
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add clean-only light Burst learned-beta DDQN results: $Dataset seed $Seed"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged results failed.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Results remain committed; do not force push.' }
} finally { Pop-Location }
