param(
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$Epochs = 3,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$Teacher = '',
    [string]$ActorRun = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw 'Project venv not found.' }
Push-Location $repo
try {
    Write-Host '=== IAT LIGHT CNN | DEVELOPMENT 0/1/3 | LEARNED BETA DDQN ==='
    Write-Host 'Existing CNN/Context reused. No CNN retraining.'
    Write-Host 'TRAIN selector: same 300 rows/domain as original run; 3 fixed-beta + 3 meta epochs by default.'
    Write-Host 'TEST: all eligible rows at 5/7/10/12/15/17/20; no 300-row TEST cap.'
    Write-Host 'Budget: 20% per level; reset model/memory/optimizer/budget; Q and beta fixed.'
    $arguments = @('AMCAL/amcal_meta_beta_light_iat013.py', '--action', $Action,
        '--seed', "$Seed", '--epochs', "$Epochs", '--device', $Device)
    if ($Teacher) { $arguments += @('--teacher', $Teacher) }
    if ($ActorRun) { $arguments += @('--actor-run', $ActorRun) }
    if ($Output) { $arguments += @('--output', $Output) }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "IAT Light0/1/3 failed: $Action" }
} finally { Pop-Location }
