param(
    [ValidateSet('All','SDNCampus','ISCX-TOR','IUST','IoT')]
    [string]$Dataset = 'All',
    [int]$Seed = 42,
    [double]$Epsilon = 0.05
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$folders = @{
    'SDNCampus' = 'SDNCampus_TEST'
    'ISCX-TOR' = 'ISCX_TOR_TEST'
    'IUST' = 'IUST_MSCN_TEST'
    'IoT' = 'UNSW_IoT_TEST'
}
if ([double]::IsNaN($Epsilon) -or [double]::IsInfinity($Epsilon) -or $Epsilon -le 0 -or $Epsilon -gt 1) {
    throw 'Epsilon must be in (0,1]'
}
$tag = $Epsilon.ToString('G8', [System.Globalization.CultureInfo]::InvariantCulture).Replace('.', 'p')
$items = if ($Dataset -eq 'All') { @('SDNCampus','ISCX-TOR','IUST','IoT') } else { @($Dataset) }
Push-Location $repo
try {
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 1) { throw 'Existing staged changes found; commit or unstage them first' }
    if ($LASTEXITCODE -ne 0) { throw 'Checking staged changes failed' }
    $roots = @()
    foreach ($item in $items) {
        $root = Join-Path $PSScriptRoot "$($folders[$item])/protocol_runs/clean0_light_meta_beta_ddqn_seed${Seed}_v1/Results_eps${tag}_v1"
        foreach ($scope in @('Online','Independent')) {
            $marker = Join-Path $root "$scope/completed.json"
            if (-not (Test-Path $marker)) { throw "Incomplete results: $marker" }
            $completion = Get-Content $marker -Raw | ConvertFrom-Json
            if ($completion.cases -ne 12) { throw "Expected 12 attack-intensity cases: $marker" }
        }
        $roots += $root
    }
    $count = 0
    foreach ($root in $roots) {
        Get-ChildItem $root -Recurse -File |
            Where-Object { $_.Extension -in '.json','.csv' } |
            ForEach-Object {
                git add -f -- $_.FullName
                if ($LASTEXITCODE -ne 0) { throw 'Staging result failed' }
                $count++
            }
    }
    if ($count -eq 0) { throw 'No result files found' }
    Write-Host "Staged $count epsilon-evaluation result files"
    git diff --cached --stat
    git diff --cached --quiet
    $diffCode = $LASTEXITCODE
    if ($diffCode -eq 1) {
        git commit -m "Add supplied-dataset epsilon $Epsilon TEST results: $Dataset"
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    } elseif ($diffCode -ne 0) { throw 'Checking staged changes failed' }
    git push origin HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; results remain committed; do not force push' }
} finally {
    Pop-Location
}
