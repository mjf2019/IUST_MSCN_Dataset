param([ValidateSet('IAT','PS','Both')][string]$Dataset = 'Both')
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $names = if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
    $files = @()
    foreach ($name in $names) {
        $stem = if ($name -eq 'PS') { 'light_ps_paironly_batch5' } else { 'paironly_batch5' }
        $runs = Get-ChildItem "AMCAL/ANT_Burst_${name}_TEST/protocol_runs" -Directory -Filter "${stem}_trainseed*"
        foreach ($run in $runs) {
            $results = Join-Path $run.FullName 'paironly_batch5/Results'
            if (-not (Test-Path $results)) { continue }
            foreach ($directory in (Get-ChildItem $results -Directory -Filter 'evaluation_*')) {
                $executionPath = Join-Path $directory.FullName 'execution.json'
                if (-not (Test-Path $executionPath)) { continue }
                $execution = Get-Content $executionPath -Raw | ConvertFrom-Json
                if ($execution.evaluation_tag -notmatch '^paironly_contextonly_frontloaded_seed[0-9]+_endpoint_metrics_v1$') { continue }
                if (-not $execution.context_only_frontloaded) { continue }
                $files += @(Get-ChildItem $directory.FullName -File | Where-Object { $_.Extension -in '.json','.csv' } | ForEach-Object {
                    (Resolve-Path -Relative $_.FullName).Replace('\','/').Replace('./','')
                })
            }
        }
    }
    if ($files.Count -eq 0) { throw 'No matching PairOnly Context-only results found.' }
    $staged = @(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged paths failed.' }
    if (@($staged | Where-Object { $_ -notin $files }).Count) { throw 'Unrelated changes are staged; commit them separately first.' }
    foreach ($file in $files) {
        git add -f -- $file
        if ($LASTEXITCODE -ne 0) { throw 'Adding results failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $code = $LASTEXITCODE
    if ($code -eq 1) {
        git commit -m 'Add original PairOnly Context-only front-loaded feedback results'
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($code -ne 0) { throw 'Checking staged changes failed.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; synchronize the branch before retrying.' }
} finally { Pop-Location }
