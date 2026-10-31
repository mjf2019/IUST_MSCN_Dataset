param([ValidateSet('IAT','PS','Both')][string]$Dataset='Both')
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $names=if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
    $pattern='^AMCAL/ANT_Burst_(' + ($names -join '|') + ')_TEST/protocol_runs/shared_loss_rl_[A-Za-z0-9_-]+/(?!Models/).+\.(json|csv)$'
    $staged=@(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed.' }
    if (@($staged | Where-Object { $_ -notmatch $pattern }).Count) {
        throw 'Unrelated changes are staged. Commit them separately first.'
    }
    $files=@()
    foreach ($name in $names) {
        $parent="AMCAL/ANT_Burst_${name}_TEST/protocol_runs"
        foreach ($run in (Get-ChildItem $parent -Directory -Filter 'shared_loss_rl_*')) {
            $files += @(Get-ChildItem $run.FullName -Recurse -File | Where-Object {
                $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]'
            } | Select-Object -ExpandProperty FullName)
        }
    }
    if ($files.Count -eq 0) { throw 'No shared-loss RL results found.' }
    foreach ($file in $files) {
        Write-Host $file
        git add -f -- $file
        if ($LASTEXITCODE -ne 0) { throw 'Adding results failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $code=$LASTEXITCODE
    if ($code -eq 1) {
        git commit -m 'Add shared-loss DDQN pilot results'
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($code -ne 0) { throw 'Checking staged changes failed.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Synchronize the branch before retrying.' }
} finally { Pop-Location }
