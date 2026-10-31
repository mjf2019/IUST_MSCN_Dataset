param([Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('auto','cpu','cuda')][string]$Device='cuda')
$ErrorActionPreference='Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python=Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python='python' }
    & $python -m unittest discover -s AMCAL/tests -p test_q75_query_guide.py -v
    if ($LASTEXITCODE -ne 0) { throw 'q75 guide tests failed' }
    Write-Host "DATASET=$Dataset | LearnedBetaDDQN / Q75GuidedDDQN / Q75Only | no training"
    Write-Host 'Same independent target holdout20%; stream budget20%; immediate feedback; no reward changes.'
    & $python AMCAL/amcal_q75_guided_holdout.py --dataset $Dataset --device $Device
    if ($LASTEXITCODE -ne 0) { throw "q75 guided independent TEST failed: $Dataset" }
} finally { Pop-Location }

