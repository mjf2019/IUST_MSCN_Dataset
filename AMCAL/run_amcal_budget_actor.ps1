param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet(5,10)][int]$BudgetPercent = 5,
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$PilotRows = 300,
    [int]$Epochs = 2,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [switch]$UniformControl
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
Push-Location $repo
try {
    & $python -c 'import torch,numpy,sklearn,joblib'
    if ($LASTEXITCODE -ne 0) { throw 'Required dependencies are missing. Nothing was installed.' }
    & $python -m unittest discover -s AMCAL/tests -p 'test_budget_actor.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Budget Actor preflight failed.' }
    $extra = @()
    if ($UniformControl) {
        if ($Action -eq 'Train') { throw 'Uniform control is evaluation only.' }
        $Action = 'Evaluate'
        $extra += '--uniform-control-only'
    }
    & $python AMCAL/amcal_budget_actor.py --dataset $Dataset --budget-percent $BudgetPercent --action $Action --seed $Seed --pilot-rows $PilotRows --epochs $Epochs --device $Device @extra
    if ($LASTEXITCODE -ne 0) { throw "Budget Actor failed: $Dataset / $BudgetPercent / $Action" }
} finally { Pop-Location }
