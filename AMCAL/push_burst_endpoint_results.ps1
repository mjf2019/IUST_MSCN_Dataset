param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('AMCAL','CNN','CALMID','MICFOAL','RAL','AF','DFE','ARF')]
    [string]$Method
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    $paths = @()
    foreach ($dataset in @('IAT','PS')) {
        $path = "AMCAL/ANT_Burst_${dataset}_TEST/protocol_runs/paper_endpoint_metrics/$Method"
        $manifest = Join-Path $path 'coverage.json'
        if (-not (Test-Path $manifest)) { throw "Missing $dataset results: $manifest" }
        $coverage = Get-Content $manifest -Raw | ConvertFrom-Json
        if ($coverage.entries.Count -ne 35) { throw "Expected 35 seed/level pairs for $dataset; found $($coverage.entries.Count)" }
        $paths += $path
    }
    & git add -f -- @paths
    if ($LASTEXITCODE -ne 0) { throw 'Adding endpoint results failed' }
    & git diff --cached --stat -- @paths
    & git diff --cached --quiet -- @paths
    if ($LASTEXITCODE -eq 1) {
        & git commit --only -m "Add $Method endpoint metrics for IAT and PS across five seeds" -- @paths
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    } elseif ($LASTEXITCODE -ne 0) { throw 'Checking staged endpoint results failed' }
    & git pull --rebase --autostash origin experiment/paironly-paid-replay
    if ($LASTEXITCODE -ne 0) { throw 'Pull failed; resolve the reported error before pushing' }
    & git push origin HEAD:experiment/paironly-paid-replay
    if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
} finally { Pop-Location }
