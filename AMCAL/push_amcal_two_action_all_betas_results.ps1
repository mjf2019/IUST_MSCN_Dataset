param([ValidateSet('IAT','PS','Both')][string]$Dataset = 'Both')
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $allowed = '^AMCAL/ANT_Burst_(IAT|PS)_TEST/protocol_runs/two_action_paper_gain_[A-Za-z0-9_-]+/.+\.(json|csv)$'
    $staged = @(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'Reading staged files failed' }
    foreach ($path in $staged) {
        if ($path -notmatch $allowed -or $path -match '/Models/') {
            throw "Unrelated staged file; unstage it before continuing: $path"
        }
    }
    $datasets = if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
    $count = 0
    foreach ($name in $datasets) {
        $root = "AMCAL/ANT_Burst_${name}_TEST/protocol_runs"
        if (-not (Test-Path $root)) { continue }
        Get-ChildItem $root -Directory -Filter 'two_action_paper_gain_*' | ForEach-Object {
            Get-ChildItem $_.FullName -Recurse -File | Where-Object {
                $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]'
            } | ForEach-Object {
                Write-Host $_.FullName
                git add -f -- $_.FullName
                if ($LASTEXITCODE -ne 0) { throw 'Adding paper-plus-gain results failed' }
                $count++
            }
        }
    }
    if ($count -eq 0) { throw 'No paper-plus-gain DDQN JSON/CSV results found' }
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add all-beta paper-plus-gain training and test results"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged changes failed' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; preserve results and resolve the reported Git error' }
} finally { Pop-Location }
