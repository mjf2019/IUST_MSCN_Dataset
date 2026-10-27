param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'Train',
    [int]$Seed = 42,
    [int]$PilotRows = 300,
    [int]$Rounds = 2,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [switch]$OriginalPaperReward
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
Push-Location $repo
try {
    & $python -c 'import torch,numpy,sklearn,joblib'
    if ($LASTEXITCODE -ne 0) { throw 'Required dependencies are missing. Nothing was installed.' }
    & $python -m unittest discover -s AMCAL/tests -p 'test_competitive_selector.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Competitive selector preflight failed.' }
    $extra = @()
    if ($OriginalPaperReward) { $extra += '--original-paper-reward' }
    & $python AMCAL/amcal_competitive_selector.py --dataset $Dataset --action $Action --seed $Seed --pilot-rows $PilotRows --rounds $Rounds --device $Device @extra
    if ($LASTEXITCODE -ne 0) { throw "Competitive selector failed: $Dataset / $Action" }
} finally { Pop-Location }
