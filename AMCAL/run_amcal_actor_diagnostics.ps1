param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet(5,10)][int]$BudgetPercent=10,
    [int]$Seed=42,
    [int]$PilotRows=300,
    [int]$Epochs=2,
    [ValidateSet('auto','cpu','cuda')][string]$Device='cuda'
)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$python=Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python='python' }
Push-Location $repo
try {
    & $python -c 'import torch,numpy,sklearn,joblib'
    if ($LASTEXITCODE -ne 0) { throw 'Required dependencies are missing. Nothing was installed.' }
    & $python -m unittest discover -s AMCAL/tests -p 'test_actor_diagnostics.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Actor diagnostic preflight failed.' }
    Write-Host 'Offline TRAIN/VAL audit only. Existing weights and TRAIN scales are frozen. No TEST access.'
    & $python AMCAL/diagnose_amcal_budget_actor.py --dataset $Dataset --budget-percent $BudgetPercent --seed $Seed --pilot-rows $PilotRows --epochs $Epochs --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Actor diagnostic failed: $Dataset" }
} finally { Pop-Location }
