param([int]$Seed = 42)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $roots = @(
        'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/paironly_batch5_trainseed42/paironly_batch5/Results',
        'AMCAL/ANT_Burst_PS_TEST/protocol_runs/light_ps_paironly_batch5_trainseed42/paironly_batch5/Results'
    )
    $dirs = @(
        Get-ChildItem $roots -Recurse -Filter execution.json -File -Force |
            Where-Object {
                $e = Get-Content $_.FullName -Raw | ConvertFrom-Json
                $e.seed -eq $Seed -and (
                    $e.evaluation_tag -like "component_full_random_seed${Seed}*" -or
                    $e.evaluation_tag -like "component_full_uncertainty_q75_seed${Seed}*"
                )
            } |
            Select-Object -ExpandProperty DirectoryName -Unique
    )
    if ($dirs.Count -ne 4) { throw 'Expected four completed runs: two controls for each dataset.' }
    git restore --staged .
    if ($LASTEXITCODE -ne 0) { throw 'Unstage failed' }
    foreach ($dir in $dirs) {
        Get-ChildItem $dir -File -Force |
            Where-Object { $_.Extension -in '.json','.csv' } |
            ForEach-Object {
                git add -f -- $_.FullName
                if ($LASTEXITCODE -ne 0) { throw 'git add failed' }
            }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add full random and uncertainty AL controls for IAT and PS seed $Seed"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    } elseif ($diffCode -ne 0) {
        throw 'Checking staged changes failed'
    }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
} finally {
    Pop-Location
}

