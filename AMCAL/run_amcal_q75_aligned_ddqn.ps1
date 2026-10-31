param([Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('All','Train','Evaluate')][string]$Action='All',
    [int]$Epochs=3,
    [ValidateSet('auto','cpu','cuda')][string]$Device='cuda')
$ErrorActionPreference='Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python=Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python='python' }
    & $python -m unittest discover -s AMCAL/tests -p test_q75_masked_ddqn.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Aligned q75 DDQN tests failed' }
    Write-Host "DATASET=$Dataset ACTION=$Action CONTINUATION_EPOCHS=$Epochs | fixed learned beta and reward; q75 mask in behavior and TD"
    Write-Host 'CNN/initial Context unchanged. Full eligible target TEST split20% independently; stream feedback budget20%.'
    & $python AMCAL/amcal_q75_aligned_ddqn.py --dataset $Dataset --action $Action --epochs $Epochs --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Aligned q75 DDQN failed: $Dataset / $Action" }
} finally { Pop-Location }

