param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$PilotRows = 300,
    [int]$Epochs = 3,
    [double[]]$Betas = @(0,10,100),
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
    foreach ($pattern in @('test_two_action_ddqn.py','test_two_action_paper_gain.py')) {
        & $python -m unittest discover -s AMCAL/tests -p $pattern -v
        if ($LASTEXITCODE -ne 0) { throw "Contract tests failed: $pattern" }
    }
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $arguments = @('AMCAL/amcal_two_action_paper_gain.py','--dataset',$Dataset,
        '--action',$Action,'--seed',"$Seed",'--pilot-rows',"$PilotRows",'--epochs',"$Epochs",
        '--device',$Device,'--betas')
    $arguments += @($Betas | ForEach-Object { $_.ToString($culture) })
    if ($ActorRun) { $arguments += @('--actor-run',$ActorRun) }
    if ($Teacher) { $arguments += @('--teacher',$Teacher) }
    if ($Output) { $arguments += @('--output',$Output) }
    Write-Host "DATASET=$Dataset | ACTION=$Action | PAPER + BETA*GAIN | BETAS=$($Betas -join ',') | BUDGET=20%"
    Write-Host "$PilotRows rows per domain; $Epochs epochs per beta; CNN/initial Context retained; no synthetic drift"
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Paper-plus-gain DDQN failed: $Dataset / $Action" }
} finally { Pop-Location }
