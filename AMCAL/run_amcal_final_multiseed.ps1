param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('All','Train','Evaluate','Collect')][string]$Action = 'All',
    [int[]]$Seeds = @(42,48,54,62,72),
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$OutputRoot = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw 'Project venv not found; nothing was installed.' }
Push-Location $repo
try {
    & $python -m unittest discover -s AMCAL/tests -p test_final_multiseed_contracts.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Multiseed contract checks failed.' }
    Write-Host "=== DATASET=$Dataset ACTION=$Action TRAINING_SEEDS=$Seeds ==="
    Write-Host 'IAT: original frozen CNN. PS: Light frozen CNN. Source CNN never retrained.'
    Write-Host 'Independent Context development, selector, DDQN and beta training per seed.'
    Write-Host 'Shared pretrained initialization; not independent end-to-end pretraining.'
    Write-Host 'Fixed TEST seed42; all eligible rows at levels5/7/10/12/15/17/20; 20% budget per level.'
    $arguments = @('AMCAL/amcal_final_multiseed.py', '--dataset', $Dataset,
        '--action', $Action, '--device', $Device)
    if ($OutputRoot) { $arguments += @('--output-root', $OutputRoot) }
    $arguments += '--seeds'
    foreach ($seed in $Seeds) { $arguments += "$seed" }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Final multiseed run failed: $Dataset / $Action" }
} finally { Pop-Location }
