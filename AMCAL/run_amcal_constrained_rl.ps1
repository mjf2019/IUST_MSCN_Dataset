param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('Train','Evaluate','All')][string]$Action='Train',
    [int]$Seed=42,
    [int]$PilotRows=300,
    [int]$EpisodeRows=96,
    [int]$Episodes=12,
    [ValidateRange(0,1)][double]$HardFraction=0.5,
    [ValidateSet('auto','cpu','cuda')][string]$Device='cuda',
    [string]$RunTag='pilot_v1'
)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$python=Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python='python' }
Push-Location $repo
try {
    & $python -c 'import torch,numpy,sklearn,joblib'
    if ($LASTEXITCODE -ne 0) { throw 'Required dependencies are missing. Nothing was installed.' }
    & $python -m unittest discover -s AMCAL/tests -p 'test_constrained_rl.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Constrained RL preflight failed.' }
    & $python AMCAL/amcal_constrained_rl.py --dataset $Dataset --action $Action --seed $Seed --pilot-rows $PilotRows --episode-rows $EpisodeRows --episodes $Episodes --hard-fraction $HardFraction --device $Device --run-tag $RunTag
    if ($LASTEXITCODE -ne 0) { throw "Constrained RL failed: $Dataset / $Action" }
} finally { Pop-Location }
