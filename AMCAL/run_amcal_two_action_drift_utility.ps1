param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$PilotRows = 300,
    [int]$Epochs = 3,
    [double[]]$Costs = @(0, 0.001),
    [double[]]$ForgettingWeights = @(0.5, 1),
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$ActorRun = '',
    [string]$Teacher = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    $python = Join-Path $repoRoot 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    foreach ($pattern in @('test_two_action_ddqn.py','test_two_action_utility.py','test_two_action_drift_utility.py')) {
        & $python -m unittest discover -s AMCAL/tests -p $pattern -v
        if ($LASTEXITCODE -ne 0) { throw "Contract tests failed: $pattern" }
    }
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $arguments = @('AMCAL/amcal_two_action_drift_utility.py', '--dataset', $Dataset,
        '--action', $Action, '--seed', "$Seed", '--pilot-rows', "$PilotRows",
        '--epochs', "$Epochs", '--device', $Device, '--costs')
    $arguments += @($Costs | ForEach-Object { $_.ToString($culture) })
    $arguments += '--forgetting-weights'
    $arguments += @($ForgettingWeights | ForEach-Object { $_.ToString($culture) })
    if ($ActorRun) { $arguments += @('--actor-run', $ActorRun) }
    if ($Teacher) { $arguments += @('--teacher', $Teacher) }
    if ($Output) { $arguments += @('--output', $Output) }
    Write-Host "Synthetic utility: $Dataset / $Action / $PilotRows rows per domain / $Epochs epochs per candidate"
    Write-Host 'Four utility candidates plus matched paper-reward DDQN; healthy CNN/Context retained'
    Write-Host 'TRAIN drift changes each epoch; independent VAL drift; unchanged official TEST; 20% cap; batch5'
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Synthetic utility experiment failed: $Dataset / $Action" }
} finally { Pop-Location }
