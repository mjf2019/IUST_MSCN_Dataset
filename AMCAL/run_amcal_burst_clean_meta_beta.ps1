param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [ValidateSet('Both','Online','Independent')][string]$Protocol = 'Both',
    [int]$Seed = 42,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [int]$CNNEpochs = 60,
    [int]$ContextEpochs = 20,
    [int]$WarmupEpochs = 3,
    [int]$Epochs = 3,
    [string]$Output = '',
    [string]$SourceTemplate = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw 'Project venv not found.' }
Push-Location $repo
try {
    Write-Host "=== DATASET=$Dataset | ACTION=$Action | FRESH LIGHT CNN | TRAIN/VAL=0 ONLY ==="
    Write-Host 'TEST: all available Burst levels, including 0/1/3 | 20% per level | independent reset | no row cap'
    $arguments = @('AMCAL/amcal_burst_clean_meta_beta.py', '--dataset', $Dataset,
        '--action', $Action, '--protocol', $Protocol, '--seed', "$Seed", '--device', $Device,
        '--cnn-epochs', "$CNNEpochs", '--context-epochs', "$ContextEpochs",
        '--warmup-epochs', "$WarmupEpochs", '--epochs', "$Epochs")
    if ($Output) { $arguments += @('--output', $Output) }
    if ($SourceTemplate) { $arguments += @('--source-template', $SourceTemplate) }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Clean-only Burst run failed: $Dataset / $Action" }
} finally { Pop-Location }
