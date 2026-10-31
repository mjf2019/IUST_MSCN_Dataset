param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet(1,2)][int[]]$SkipCounts = @(1,2),
    [int]$Seed = 42,
    [int]$PilotRows = 300,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$ActorRun = '',
    [string]$Teacher = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    & $python -m unittest discover -s AMCAL/tests -p test_two_action_periodic.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Periodic query contract tests failed' }
    $arguments = @('AMCAL/amcal_two_action_periodic.py','--dataset',$Dataset,'--seed',"$Seed",
        '--pilot-rows',"$PilotRows",'--device',$Device,'--skip-counts')
    $arguments += @($SkipCounts | ForEach-Object { "$_" })
    if ($ActorRun) { $arguments += @('--actor-run',$ActorRun) }
    if ($Teacher) { $arguments += @('--teacher',$Teacher) }
    if ($Output) { $arguments += @('--output',$Output) }
    Write-Host "EVALUATION ONLY | DATASET=$Dataset | SKIP_COUNTS=$($SkipCounts -join ',') | BUDGET=20% | NO RETRAINING"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Periodic evaluation failed: $Dataset" }
} finally { Pop-Location }
