param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [int]$Seed = 42,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw 'Project venv not found.' }
Push-Location $repo
try {
    Write-Host "=== $Dataset | FINAL AMCAL CHECKPOINT | RANDOM WITHOUT WINDOW | FULL TEST ==="
    Write-Host '20% cap per level; immediate paid feedback; Context -> Frozen; no retraining.'
    $arguments = @('AMCAL/amcal_final_random_control.py', '--dataset', $Dataset,
        '--seed', "$Seed", '--device', $Device)
    if ($Output) { $arguments += @('--output', $Output) }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Final random control failed: $Dataset" }
} finally { Pop-Location }
