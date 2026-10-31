param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference='Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python=Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python='python' }
    & $python -m unittest discover -s AMCAL/tests -p test_fixed_beta_fulltest.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Full TEST candidate checks failed' }
    Write-Host "DATASET=$Dataset | BETAS=10,15,20,25,30 | TEST=ALL eligible rows | BUDGET=20% per level"
    Write-Host 'Evaluation only: existing Q checkpoints reused. No training or CNN tuning.'
    & $python AMCAL/amcal_fixed_beta_fulltest.py --dataset $Dataset --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Full TEST beta sweep failed: $Dataset" }
} finally { Pop-Location }

