param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    $root = "AMCAL/ANT_Burst_$($Dataset)_TEST/protocol_runs/final_multiseed_fixedcnn_v1"
    if (-not (Test-Path "$root/Results/completed.json")) {
        throw 'Complete all planned training seeds and TEST levels before pushing.'
    }
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Existing staged changes found. Commit or unstage them first.' }
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect staged changes.' }
    $files = @(Get-ChildItem $root -Recurse -File |
        Where-Object { $_.Extension -in '.json','.csv' -and
            $_.FullName -notmatch '[\\/]Models[\\/]|\.partial_' })
    if ($files.Count -eq 0) { throw 'No JSON/CSV results found.' }
    Write-Host "Staging $($files.Count) result files from $root"
    foreach ($file in $files) {
        git add -f -- $file.FullName
        if ($LASTEXITCODE -ne 0) { throw 'git add failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add $Dataset final AMCAL independent-training multiseed results"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($diffCode -ne 0) { throw 'Cannot inspect staged results.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Results remain committed; do not force push.' }
} finally { Pop-Location }
