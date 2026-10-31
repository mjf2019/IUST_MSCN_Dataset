param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$TestRows = 300,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$ParentRun = '',
    [string]$CandidateRun = '',
    [string]$ActorRun = '',
    [string]$Teacher = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    foreach ($pattern in @('test_two_action_paper_gain.py','test_two_action_more_betas.py')) {
        & $python -m unittest discover -s AMCAL/tests -p $pattern -v
        if ($LASTEXITCODE -ne 0) { throw "Contract tests failed: $pattern" }
    }
    $arguments = @('AMCAL/amcal_two_action_more_betas.py','--dataset',$Dataset,'--action',$Action,
        '--seed',"$Seed",'--test-rows',"$TestRows",'--device',$Device)
    if ($ParentRun) { $arguments += @('--parent-run',$ParentRun) }
    if ($CandidateRun) { $arguments += @('--candidate-run',$CandidateRun) }
    if ($ActorRun) { $arguments += @('--actor-run',$ActorRun) }
    if ($Teacher) { $arguments += @('--teacher',$Teacher) }
    if ($Output) { $arguments += @('--output',$Output) }
    Write-Host "DATASET=$Dataset | ADDITIONAL BETAS=25,30,35,40 | ACTION=$Action | TEST_ROWS=$TestRows | BUDGET=20%"
    Write-Host 'Train beta 25/30/35/40 with the same recipe; retain all previous models; TEST all four on seven intensities'
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Additional-beta experiment failed: $Dataset / $Action" }
} finally { Pop-Location }
