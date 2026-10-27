param([ValidateSet('IAT','PS','Both')][string]$Dataset='Both')
$ErrorActionPreference='Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $names=if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
    $pattern='^AMCAL/ANT_Burst_(' + ($names -join '|') + ')_TEST/protocol_runs/competitive_selector_seed[0-9]+_p[0-9]+_r[0-9]+_v1/(?!Models/).+\.(json|csv)$'
    $staged=@(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed.' }
    if (@($staged | Where-Object { $_ -notmatch $pattern }).Count) { throw 'Unrelated changes are staged; commit them separately first.' }
    $files=@()
    foreach ($name in $names) {
        foreach ($run in (Get-ChildItem "AMCAL/ANT_Burst_${name}_TEST/protocol_runs" -Directory -Filter 'competitive_selector_*')) {
            $files += @(Get-ChildItem $run.FullName -Recurse -File | Where-Object {
                $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]'
            } | Select-Object -ExpandProperty FullName)
        }
    }
    if ($files.Count -eq 0) { throw 'No competitive selector outputs found.' }
    foreach ($file in $files) {
        Write-Host $file
        git add -f -- $file
        if ($LASTEXITCODE -ne 0) { throw 'Adding results failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $code=$LASTEXITCODE
    if ($code -eq 1) {
        git commit -m 'Add compact competitive selector results'
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($code -ne 0) { throw 'Checking staged changes failed.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; synchronize the branch before retrying.' }
} finally { Pop-Location }
