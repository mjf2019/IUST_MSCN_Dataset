param(
    [ValidateSet('All','SDNCampus','ISCX-TOR','IUST','IoT')]
    [string]$Dataset = 'All',
    [ValidateSet('All','Train','Evaluate')]
    [string]$Action = 'All',
    [ValidateSet('Both','Online','Independent')]
    [string]$Protocol = 'Both',
    [int]$Seed = 42,
    [ValidateSet('auto','cpu','cuda')]
    [string]$Device = 'cuda',
    [int]$WarmupEpochs = 3,
    [int]$Epochs = 3,
    [string]$SourceRoot = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) {
    throw 'Project venv not found. Activate the intended environment and run the Python command directly.'
}
if ($Dataset -eq 'All' -and ($SourceRoot -or $Output)) {
    throw 'Use a single dataset with an explicit SourceRoot or Output.'
}
$items = if ($Dataset -eq 'All') { @('SDNCampus','ISCX-TOR','IUST','IoT') } else { @($Dataset) }
Push-Location $repo
try {
    foreach ($item in $items) {
        Write-Host "=== FINAL LIGHT CNN / CLEAN LEVEL 0 | DATASET=$item | ACTION=$Action | TEST=$Protocol | BUDGET=20% ==="
        $arguments = @('AMCAL/amcal_supplied_light_meta_beta.py', '--dataset', $item,
            '--action', $Action, '--protocol', $Protocol, '--seed', "$Seed", '--device', $Device,
            '--warmup-epochs', "$WarmupEpochs", '--epochs', "$Epochs")
        if ($SourceRoot) { $arguments += @('--source-root', $SourceRoot) }
        if ($Output) { $arguments += @('--output', $Output) }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Light learned-beta DDQN failed: $item / $Action" }
    }
} finally {
    Pop-Location
}
