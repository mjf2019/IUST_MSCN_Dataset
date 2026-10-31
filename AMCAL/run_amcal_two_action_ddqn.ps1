param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$PilotRows = 300,
    [int]$Epochs = 3,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$ActorRun = '',
    [string]$Teacher = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $repoRoot
$python = Join-Path $repoRoot 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
& $python -m unittest discover -s AMCAL/tests -p test_two_action_ddqn.py -v
if ($LASTEXITCODE -ne 0) { throw 'Two-action DDQN contract tests failed' }
$arguments = @('AMCAL/amcal_two_action_ddqn.py', '--dataset', $Dataset,
    '--action', $Action, '--seed', "$Seed", '--pilot-rows', "$PilotRows",
    '--epochs', "$Epochs", '--device', $Device)
if ($ActorRun) { $arguments += @('--actor-run', $ActorRun) }
if ($Teacher) { $arguments += @('--teacher', $Teacher) }
if ($Output) { $arguments += @('--output', $Output) }
Write-Host "Two-action DDQN: $Dataset / $Action / $Epochs epochs / $PilotRows rows per domain / 20% cap"
Write-Host 'Same healthy Context as parent Actor; fixed Context -> Frozen; batch5; no CNN retraining'
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Two-action DDQN failed: $Dataset / $Action" }
