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
    foreach ($pattern in @('test_two_action_paper_gain.py','test_two_action_all_betas.py')) {
        & $python -m unittest discover -s AMCAL/tests -p $pattern -v
        if ($LASTEXITCODE -ne 0) { throw "Contract tests failed: $pattern" }
    }
    $arguments = @('AMCAL/amcal_two_action_all_betas.py','--dataset',$Dataset,'--action',$Action,
        '--seed',"$Seed",'--test-rows',"$TestRows",'--device',$Device)
    if ($ParentRun) { $arguments += @('--parent-run',$ParentRun) }
    if ($CandidateRun) { $arguments += @('--candidate-run',$CandidateRun) }
    if ($ActorRun) { $arguments += @('--actor-run',$ActorRun) }
    if ($Teacher) { $arguments += @('--teacher',$Teacher) }
    if ($Output) { $arguments += @('--output',$Output) }
    Write-Host "DATASET=$Dataset | ALL BETAS=0,10,15,20,100 | ACTION=$Action | TEST_ROWS=$TestRows | BUDGET=20%"
    Write-Host 'Reuse existing beta 0/10/100; train only missing beta 15/20; TEST every beta on all seven intensities'
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "All-beta experiment failed: $Dataset / $Action" }
} finally { Pop-Location }
