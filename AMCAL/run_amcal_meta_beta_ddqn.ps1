param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [int]$Epochs = 3,
    [int]$TestRows = 300,
    [double[]]$Budgets = @(0.2),
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    & $python -m unittest discover -s AMCAL/tests -p test_meta_beta_ddqn.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Learned reward-weight tests failed' }
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $arguments = @('AMCAL/amcal_meta_beta_ddqn.py','--dataset',$Dataset,'--action',$Action,
        '--epochs',"$Epochs",'--test-rows',"$TestRows",'--device',$Device,'--test-budgets')
    foreach ($budget in $Budgets) { $arguments += $budget.ToString($culture) }
    Write-Host "DATASET=$Dataset ACTION=$Action EPOCHS=$Epochs | TRAIN-learned reward weight | original CNN/Context"
    Write-Host "TEST BUDGETS=$Budgets | compare LearnedBeta, SourceFixed and Uncertainty | no CNN retraining"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Learned reward-weight run failed: $Dataset / $Action" }
} finally { Pop-Location }

