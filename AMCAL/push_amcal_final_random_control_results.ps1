param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [int]$Seed = 42
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    $root = "AMCAL/ANT_Burst_$($Dataset)_TEST/protocol_runs/final_random_full_seed$($Seed)_v1"
    if (-not (Test-Path "$root/Results/completed.json")) {
        throw 'Complete all seven random-control TEST levels before pushing.'
    }
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Existing staged changes found. Commit or unstage them first.' }
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect staged changes.' }
    $files = @(Get-ChildItem $root -Recurse -File |
        Where-Object { $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]' })
    Write-Host "Staging $($files.Count) result files from $root"
    foreach ($file in $files) {
        git add -f -- $file.FullName
        if ($LASTEXITCODE -ne 0) { throw 'git add failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add $Dataset final AMCAL non-window random control"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($diffCode -ne 0) { throw 'Cannot inspect staged diff.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Results remain committed; do not force push.' }
} finally { Pop-Location }
