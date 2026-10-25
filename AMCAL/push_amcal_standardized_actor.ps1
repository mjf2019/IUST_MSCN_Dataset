param([ValidateSet('IAT','PS','Both')][string]$Dataset = 'Both')
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $names = if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
    $pattern = '^AMCAL/ANT_Burst_(' + ($names -join '|') + ')_TEST/protocol_runs/minimax_query_batch5_seed[^/]+_v2_h16_c(0|0\.05)_actorstd_(natural|balanced)(_trainuniform)?(_pilot[0-9]+_ep[0-9]+)?(_logitreg[0-9pEe+\-]+)?(_utility)?/(.+\.(json|csv))$'
    $staged = @(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged paths failed.' }
    if (@($staged | Where-Object { $_ -notmatch $pattern }).Count) {
        throw 'Unrelated changes are staged. Commit them separately first.'
    }
    $paths = @()
    foreach ($name in $names) {
        $runs = Get-ChildItem "AMCAL/ANT_Burst_${name}_TEST/protocol_runs" -Directory -Filter 'minimax_query_batch5_seed*_v2_h16_c*_actorstd_*'
        foreach ($run in $runs) {
            if (-not (Test-Path (Join-Path $run.FullName 'training_complete.json'))) { continue }
            $paths += @(Get-ChildItem $run.FullName -Recurse -File | Where-Object {
                $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]'
            } | Select-Object -ExpandProperty FullName)
        }
    }
    if ($paths.Count -eq 0) { throw 'No completed standardized Actor training outputs were found.' }
    foreach ($path in $paths) {
        git add -f -- $path
        if ($LASTEXITCODE -ne 0) { throw 'Adding Actor results failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $code = $LASTEXITCODE
    if ($code -eq 1) {
        git commit -m 'Add standardized Actor TRAIN and VAL control results'
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($code -ne 0) { throw 'Checking staged results failed.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Synchronize the branch before retrying.' }
} finally { Pop-Location }


