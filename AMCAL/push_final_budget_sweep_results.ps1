param(
    [ValidateSet('Both','IAT','PS')]
    [string]$Dataset = 'Both',
    [int[]]$Seeds = @(42,48,54,62,72),
    [double[]]$Budgets = @(10,15,20,25,30,35,40,45,50)
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$datasets = if ($Dataset -eq 'Both') { @('IAT','PS') } else { @($Dataset) }
$budgetValues = @($Budgets | Sort-Object -Unique)
$tags = @($budgetValues | ForEach-Object {
    if ([double]::IsNaN($_) -or [double]::IsInfinity($_) -or $_ -le 0 -or $_ -gt 100) {
        throw 'Budgets must be finite percentages in (0,100]'
    }
    $_.ToString('G8',[System.Globalization.CultureInfo]::InvariantCulture).Replace('.','p')
})
$seedTag = 'seeds_' + ($Seeds -join '_')
$budgetTag = 'budgets_' + ($tags -join '_')
Push-Location $repo
try {
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Existing staged changes found; commit or unstage them first' }
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed' }
    $folders = @()
    foreach ($name in $datasets) {
        $root = Join-Path $PSScriptRoot "ANT_Burst_${name}_TEST/protocol_runs/final_multiseed_fixedcnn_v1"
        $summary = Join-Path $root "Results_final_budget_sweep_eps0p05_v1/$seedTag/$budgetTag"
        $marker = Join-Path $summary 'completed.json'
        if (-not (Test-Path $marker)) { throw "Incomplete aggregate: $summary" }
        $completion = Get-Content $marker -Raw | ConvertFrom-Json
        if ($completion.cases -ne ($Seeds.Count * $tags.Count * 7)) {
            throw "Unexpected number of seed/budget/level cases: $summary"
        }
        $folders += $summary
        foreach ($seed in $Seeds) {
            foreach ($tag in $tags) {
                $folder = Join-Path $root "seed$seed/final_budget_sweep_eps0p05_v1/budget$tag"
                if (-not (Test-Path (Join-Path $folder 'completed.json'))) {
                    throw "Incomplete budget evaluation: $folder"
                }
                $folders += $folder
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
    if ($count -eq 0) { throw 'No budget-sweep result files found' }
    Write-Host "Staged $count budget-sweep result files"
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add final five-seed budget sensitivity results: $Dataset"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged changes failed' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; results remain committed; do not force push' }
} finally {
    Pop-Location
}
