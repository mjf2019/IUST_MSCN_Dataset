param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('All','Collect','Actor','ActorScale','Routing','FixedRouter')][string]$Phase = 'All',
    [int]$Seed = 42,
    [int]$Rows = 256,
    [int]$FitSteps = 300,
    [string]$Output = '',
    [string]$SourceRun = '',
    [string]$V2Run = '',
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
Push-Location $repo
try {
    & $python -c 'import torch, numpy, sklearn, joblib'
    if ($LASTEXITCODE -ne 0) { throw 'Required dependencies are missing. Nothing was installed.' }
    & $python -m unittest discover -s AMCAL/tests -p 'test_minimax*.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Diagnostic preflight checks failed.' }
    $runArgs = @('AMCAL/amcal_minimax_paironly.py', '--dataset', $Dataset,
        '--action', 'Diagnose', '--diagnostic-phase', $Phase,
        '--seed', "$Seed", '--device', $Device, '--diagnostic-rows', "$Rows",
        '--diagnostic-fit-steps', "$FitSteps", '--budget-fraction', '0.2',
        '--context-lr', '0.00025', '--replay-weight', '0.1',
        '--horizon', '16', '--cost', '0.05', '--reference-size', '32')
    if ($Output) { $runArgs += @('--output', $Output) }
    if ($SourceRun) { $runArgs += @('--source-run', $SourceRun) }
    if ($V2Run) { $runArgs += @('--v2-run', $V2Run) }
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Minimax diagnostics failed: $Dataset / $Phase" }
} finally { Pop-Location }
