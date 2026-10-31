param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [double]$Beta = -1,
    [double[]]$Budgets = @(0.3,0.4,0.5),
    [ValidateSet('Current','CNN13')][string[]]$Variants = @('Current'),
    [int]$TestRows = 300,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    if ($Beta -eq -1) { $Beta = if ($Dataset -eq 'IAT') { 26 } else { 55 } }
    $python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    & $python -m unittest discover -s AMCAL/tests -p test_selected_beta_budgets.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Budget evaluation tests failed' }
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $arguments = @('AMCAL/amcal_selected_beta_budgets.py','--dataset',$Dataset,
        '--beta',$Beta.ToString($culture),'--test-rows',"$TestRows",'--device',$Device,'--budgets')
    foreach ($budget in $Budgets) { $arguments += $budget.ToString($culture) }
    $arguments += @('--variants') + $Variants
    Write-Host "EVALUATION ONLY | DATASET=$Dataset BETA=$Beta VARIANTS=$Variants BUDGETS=$Budgets | no retraining"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Selected-beta budget evaluation failed: $Dataset" }
} finally { Pop-Location }

