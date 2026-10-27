param([ValidateSet('IAT','PS','Both')][string]$Dataset='Both')
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $names=if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
    $pattern='^AMCAL/ANT_Burst_(' + ($names -join '|') + ')_TEST/protocol_runs/budget_actor_q(05|10)_seed[0-9]+_pilot[0-9]+_ep[0-9]+(_scaled_early20(_rankwarm)?)?/(?!Models/).+\.(json|csv)$'
    $staged=@(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed.' }
    if (@($staged | Where-Object { $_ -notmatch $pattern }).Count) { throw 'Unrelated changes are staged; commit them separately first.' }
    $files=@()
    foreach ($name in $names) {
        foreach ($run in (Get-ChildItem "AMCAL/ANT_Burst_${name}_TEST/protocol_runs" -Directory -Filter 'budget_actor_q*')) {
            $files += @(Get-ChildItem $run.FullName -Recurse -File | Where-Object {
                $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]'
            } | Select-Object -ExpandProperty FullName)
        }
    }
    if ($files.Count -eq 0) { throw 'No budget Actor outputs found.' }
    foreach ($file in $files) {
        git add -f -- $file
        if ($LASTEXITCODE -ne 0) { throw 'Adding results failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $code=$LASTEXITCODE
    if ($code -eq 1) {
        git commit -m 'Add fixed-Context budget Actor and matched controls'
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($code -ne 0) { throw 'Checking staged changes failed.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; synchronize the branch before retrying.' }
} finally { Pop-Location }
