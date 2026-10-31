param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$PilotRows = 300,
    [int]$Epochs = 3,
    [double]$Cost = 0.05,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$Source = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
Push-Location $repo
try {
    & $python -c 'import torch,numpy,sklearn,pandas,joblib'
    if ($LASTEXITCODE -ne 0) { throw 'Required dependencies are missing. Nothing was installed.' }
    & $python -m unittest discover -s AMCAL/tests -p 'test_shared_loss_rl.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Shared-loss RL preflight failed.' }
    $extra = @()
    if ($Source) { $extra += @('--source', $Source) }
    if ($Output) { $extra += @('--output', $Output) }
    $costText = $Cost.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    & $python AMCAL/amcal_shared_loss_rl.py --dataset $Dataset --action $Action --seed $Seed --pilot-rows $PilotRows --epochs $Epochs --cost $costText --device $Device @extra
    if ($LASTEXITCODE -ne 0) { throw "Shared-loss RL failed: $Dataset / $Action" }
} finally { Pop-Location }
