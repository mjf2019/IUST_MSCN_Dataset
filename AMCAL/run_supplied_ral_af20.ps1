param(
    [Parameter(Mandatory=$true)][ValidateSet('RAL','AF')][string]$Method,
    [ValidateSet('All','SDNCampus','ISCX-TOR','IUST','IoT')][string]$Dataset = 'All',
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [int]$Seed = 42,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [int]$Jobs = -1,
    [int]$ProgressEvery = 100,
    [ValidateSet('prefix','global')][string]$RALBudgetMode = 'prefix',
    [int]$Iterations = 1000,
    [int]$TuningEpochs = 30
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw 'Project venv not found.' }
$items = if ($Dataset -eq 'All') { @('SDNCampus','ISCX-TOR','IUST','IoT') } else { @($Dataset) }
Push-Location $repo
try {
    foreach ($item in $items) {
        Write-Host "=== DATASET=$item | METHOD=$Method | TRAIN=0 | 12 TARGET CASES | BUDGET=20% ==="
        if ($Method -eq 'RAL') {
            Write-Host 'RAL FULL: CPU; predict before feedback; refit on ALL source TRAIN + ALL acquired labels; reset per intensity.'
        } else {
            Write-Host 'AF AUTHOR FULL20: corrected author structure; full target calibration cap; offline query scoring; reset per intensity.'
        }
        & $python 'AMCAL/amcal_supplied_ral_af20.py' --dataset $item --method $Method `
            --action $Action --seed $Seed --device $Device --jobs $Jobs `
            --progress-every $ProgressEvery --ral-budget-mode $RALBudgetMode `
            --iterations $Iterations --tuning-epochs $TuningEpochs
        if ($LASTEXITCODE -ne 0) { throw "Supplied baseline failed: $item / $Method / $Action" }
    }
} finally { Pop-Location }
