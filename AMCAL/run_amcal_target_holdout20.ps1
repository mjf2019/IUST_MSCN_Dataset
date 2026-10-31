param([Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('auto','cpu','cuda')][string]$Device='cuda')
$ErrorActionPreference='Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python=Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python='python' }
    & $python -m unittest discover -s AMCAL/tests -p test_independent_target_holdout.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Independent target holdout tests failed' }
    Write-Host "DATASET=$Dataset | LearnedBetaDDQN versus UncertaintyWindow | all seven levels | no training"
    Write-Host 'Group-disjoint target holdout20%; adaptation stream80%; feedback cap20% of stream, about16% of all rows.'
    & $python AMCAL/amcal_target_holdout20.py --dataset $Dataset --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Independent target TEST failed: $Dataset" }
} finally { Pop-Location }

