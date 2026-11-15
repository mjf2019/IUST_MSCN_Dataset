param(
    [ValidateSet('All','IAT','PS')]
    [string]$Dataset = 'All',
    [ValidateSet('Both','Random','Uncertainty')]
    [string]$Method = 'Both'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$datasets = if ($Dataset -eq 'All') { @('IAT','PS') } else { @($Dataset) }
$methods = if ($Method -eq 'Both') { @('Random','Uncertainty') } else { @($Method) }
Push-Location $repo
try {
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Existing staged changes found; commit or unstage them first' }
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed' }
    $folders = @()
    foreach ($name in $datasets) {
        $root = Join-Path $PSScriptRoot "ANT_Burst_${name}_TEST/protocol_runs/final_multiseed_fixedcnn_v1"
        foreach ($item in $methods) {
            $summary = Join-Path $root "Results_test_controls_v1/$item"
            if (-not (Test-Path (Join-Path $summary 'completed.json'))) {
                throw "Incomplete aggregate: $summary"
            }
            $folders += $summary
            foreach ($seed in @(42,48,54,62,72)) {
                $case = Join-Path $root "seed$seed/test_controls_v1/$item"
                if (-not (Test-Path (Join-Path $case 'completed.json'))) {
                    throw "Incomplete TEST control: $case"
                }
                $folders += $case
            }
        }
    }
    $count = 0
    foreach ($folder in $folders) {
        Get-ChildItem $folder -Recurse -File |
            Where-Object { $_.Extension -in '.json','.csv' } |
            ForEach-Object {
                git add -f -- $_.FullName
                if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
                $count++
            }
    }
    if ($count -eq 0) { throw 'No TEST-control result files found' }
    Write-Host "Staged $count TEST-control result files"
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add final five-seed TEST controls: $Dataset $Method"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged changes failed' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; results remain committed; do not force push' }
} finally {
    Pop-Location
}
