param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$Epochs = 6,
    [int]$Horizon = 16,
    [int]$ReferenceSize = 32,
    [double]$Cost = 0.05,
    [double]$GainWeight = 1,
    [double]$ForgettingWeight = 1,
    [string]$SourceRun = '',
    [string]$Output = '',
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
$culture = [Globalization.CultureInfo]::InvariantCulture
Push-Location $repo
try {
    & $python -c 'import torch, numpy, sklearn, joblib'
    if ($LASTEXITCODE -ne 0) { throw 'Required packages are unavailable in this Python environment. Nothing was installed.' }
    & $python -m unittest discover -s AMCAL/tests -p test_minimax_query.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Minimax CPU preflight checks failed.' }
    $runArgs = @('AMCAL/amcal_minimax_paironly.py', '--dataset', $Dataset,
        '--action', $Action, '--seed', "$Seed", '--device', $Device,
        '--epochs', "$Epochs", '--horizon', "$Horizon", '--cost', $Cost.ToString($culture),
        '--reference-size', "$ReferenceSize",
        '--gain-weight', $GainWeight.ToString($culture),
        '--forgetting-weight', $ForgettingWeight.ToString($culture),
        '--budget-fraction', '0.2', '--context-lr', '0.00025', '--replay-weight', '0.1')
    if ($SourceRun) { $runArgs += @('--source-run', $SourceRun) }
    if ($Output) { $runArgs += @('--output', $Output) }
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Minimax PairOnly failed: $Dataset / $Action" }
} finally { Pop-Location }
