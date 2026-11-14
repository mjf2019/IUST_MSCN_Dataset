param(
    [ValidateSet('Both', 'IAT', 'PS')]
    [string]$Dataset = 'Both'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
Push-Location $repo
try {
    $staged = @(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed' }
    if ($staged.Count -gt 0) { throw 'Existing staged changes found; commit or unstage them first' }
    $datasets = if ($Dataset -eq 'Both') { @('IAT', 'PS') } else { @($Dataset) }
    $folders = @()
    foreach ($name in $datasets) {
        $folder = Join-Path $PSScriptRoot "ANT_Burst_${name}_TEST/protocol_runs/final_multiseed_fixedcnn_v1/epsilon_val_selection_v1"
        if (-not (Test-Path (Join-Path $folder 'completed.json'))) {
            throw "Incomplete VAL results: $folder"
        }
        $folders += $folder
    }
    $selection = Join-Path $PSScriptRoot 'final_epsilon_val_selection_v1'
    if (Test-Path (Join-Path $selection 'selection.json')) { $folders += $selection }
    $count = 0
    foreach ($folder in $folders) {
        Get-ChildItem $folder -Recurse -File |
            Where-Object { $_.Extension -in '.json', '.csv' } |
            ForEach-Object {
                git add -f -- $_.FullName
                if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
                $count++
            }
    }
    if ($count -eq 0) { throw 'No VAL result files found' }
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add validation-selected inference epsilon results"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged changes failed' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; results remain local; do not force push' }
} finally {
    Pop-Location
}
