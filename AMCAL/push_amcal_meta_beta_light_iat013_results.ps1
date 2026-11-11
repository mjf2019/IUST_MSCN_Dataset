param(
    [int]$Seed = 42,
    [int]$Epochs = 3,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    if (-not $Output) {
        $Output = "AMCAL/ANT_Burst_IAT_TEST/protocol_runs/light_iat013_meta_beta_ddqn_seed$($Seed)_p300_e$($Epochs)_v1"
    }
    $root = (Resolve-Path $Output).Path
    $runs = (Resolve-Path 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs').Path
    if (-not $root.StartsWith($runs + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Output must be a specific IAT protocol_runs subdirectory.'
    }
    if (-not (Test-Path (Join-Path $root 'training_complete.json'))) {
        throw 'Training completion marker missing.'
    }
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Existing staged changes found. Commit or unstage them first.' }
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect staged changes.' }
    $files = @(Get-ChildItem $root -Recurse -File |
        Where-Object { $_.Extension -in '.json','.csv' -and $_.FullName -notmatch '[\\/]Models[\\/]' })
    if ($files.Count -eq 0) { throw 'No result JSON/CSV files found.' }
    Write-Host "Staging $($files.Count) result files from $root"
    foreach ($file in $files) {
        git add -f -- $file.FullName
        if ($LASTEXITCODE -ne 0) { throw 'git add failed.' }
    }
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add IAT Light learned-beta 0-1-3 full-test results"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
    } elseif ($diffCode -ne 0) { throw 'Cannot inspect staged result diff.' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed. Results remain committed; do not force push.' }
} finally { Pop-Location }
