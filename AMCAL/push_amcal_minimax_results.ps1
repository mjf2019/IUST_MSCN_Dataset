param([ValidateSet('IAT','PS','Both')][string]$Dataset = 'Both')
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $datasets = if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
    $paths = @()
    foreach ($name in $datasets) {
        $root = "AMCAL/ANT_Burst_${name}_TEST/protocol_runs"
        $runs = @(Get-ChildItem $root -Directory -Filter 'minimax_query_batch5_seed*')
        foreach ($run in $runs) {
            $training = Join-Path $run.FullName 'training.json'
            if (Test-Path $training) { $paths += $training }
            $completed = Join-Path $run.FullName 'training_complete.json'
            if (Test-Path $completed) { $paths += $completed }
            $results = Join-Path $run.FullName 'Results'
            if (Test-Path $results) {
                Get-ChildItem $results -Directory -Filter 'evaluation_*' | ForEach-Object {
                    if (Test-Path (Join-Path $_.FullName 'completed.json')) {
                        $paths += @(Get-ChildItem $_.FullName -File | Where-Object {
                            $_.Extension -in '.json','.csv'
                        } | Select-Object -ExpandProperty FullName)
                    }
                }
            }
        }
    }
    if ($paths.Count -eq 0) { throw 'No experimental results were found.' }
    $staged = @(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged paths failed.' }
    $outside = @($staged | Where-Object {
        $_ -notmatch '^AMCAL/ANT_Burst_(IAT|PS)_TEST/protocol_runs/minimax_query_batch5_seed[^/]+/(training(_complete)?\.json|Results/)'
    })
    if ($outside.Count) { throw 'Unrelated changes are staged. Commit them separately before pushing results.' }
    foreach ($path in $paths) {
        git add -f -- $path
        if ($LASTEXITCODE -ne 0) { throw 'Adding experimental results failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m 'Add experimental minimax PairOnly results'
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged results failed.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Synchronize the branch before retrying.' }
} finally { Pop-Location }
