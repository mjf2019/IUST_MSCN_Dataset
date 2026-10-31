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
    & $python -m unittest discover -s AMCAL/tests -p test_two_action_ddqn.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Base DDQN contract tests failed' }
    & $python -m unittest discover -s AMCAL/tests -p test_two_action_utility.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Utility reward contract tests failed' }
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $arguments = @('AMCAL/amcal_two_action_utility.py', '--dataset', $Dataset,
        '--action', $Action, '--seed', "$Seed", '--pilot-rows', "$PilotRows",
        '--epochs', "$Epochs", '--device', $Device, '--costs')
    $arguments += @($Costs | ForEach-Object { $_.ToString($culture) })
    $arguments += '--forgetting-weights'
    $arguments += @($ForgettingWeights | ForEach-Object { $_.ToString($culture) })
    if ($ActorRun) { $arguments += @('--actor-run', $ActorRun) }
    if ($Teacher) { $arguments += @('--teacher', $Teacher) }
    if ($Output) { $arguments += @('--output', $Output) }
    Write-Host "Reward alignment: $Dataset / $Action / $PilotRows rows per domain / $Epochs epochs per candidate"
    Write-Host 'Defaults: four utility candidates plus one matched paper-reward DDQN; selection uses VAL only'
    Write-Host 'Original IAT CNN / Light PS CNN; healthy Context retained; fixed Context -> Frozen; batch5; 20% cap'
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Two-action utility experiment failed: $Dataset / $Action" }
} finally { Pop-Location }
