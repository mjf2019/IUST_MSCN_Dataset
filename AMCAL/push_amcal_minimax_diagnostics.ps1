param([ValidateSet('IAT','PS','Both')][string]$Dataset = 'Both')
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $names = if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
    $pattern = '^AMCAL/ANT_Burst_(' + ($names -join '|') + ')_TEST/protocol_runs/minimax_diagnostics_seed[^/]+/.+\.(json|csv)$'
    $staged = @(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged paths failed.' }
    if (@($staged | Where-Object { $_ -notmatch $pattern }).Count) {
        throw 'Unrelated changes are staged. Commit them separately before pushing diagnostics.'
    }
    $count = 0
    foreach ($name in $names) {
        $root = "AMCAL/ANT_Burst_${name}_TEST/protocol_runs"
        Get-ChildItem $root -Directory -Filter 'minimax_diagnostics_seed*' | ForEach-Object {
            Get-ChildItem $_.FullName -Recurse -File | Where-Object { $_.Extension -in '.json','.csv' } | ForEach-Object {
                git add -f -- $_.FullName
                if ($LASTEXITCODE -ne 0) { throw 'Adding diagnostics failed.' }
                $count++
            }
        }
    }
    if ($count -eq 0) { throw 'No diagnostic JSON/CSV outputs were found.' }
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m 'Add TRAIN target, Actor-fit and routing diagnostics'
        if ($LASTEXITCODE -ne 0) { throw 'Diagnostic commit failed.' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged diagnostics failed.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Synchronize the branch before retrying.' }
} finally { Pop-Location }
