param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('Both','DDQN','Uncertainty')][string]$Method = 'Both',
    [ValidateSet(30,40,50,60,70,80)][int[]]$Budgets = @(30,40,50,60,70,80),
    [int]$Seed = 42,
    [int]$PilotRows = 300,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$PaperRun = '',
    [string]$ActorRun = '',
    [string]$Teacher = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    & $python -m unittest discover -s AMCAL/tests -p test_two_action_budget_sweep.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Budget sweep contract tests failed' }
    $arguments = @('AMCAL/amcal_two_action_budget_sweep.py','--dataset',$Dataset,
        '--method',$Method,'--seed',"$Seed",'--pilot-rows',"$PilotRows",'--device',$Device,'--budgets')
    $arguments += @($Budgets | ForEach-Object { "$_" })
    if ($PaperRun) { $arguments += @('--paper-run',$PaperRun) }
    if ($ActorRun) { $arguments += @('--actor-run',$ActorRun) }
    if ($Teacher) { $arguments += @('--teacher',$Teacher) }
    if ($Output) { $arguments += @('--output',$Output) }
    Write-Host "EVALUATION ONLY | DATASET=$Dataset | METHOD=$Method | BUDGETS=$($Budgets -join ',')% | NO RETRAINING"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Budget sweep failed: $Dataset / $Method" }
} finally { Pop-Location }
