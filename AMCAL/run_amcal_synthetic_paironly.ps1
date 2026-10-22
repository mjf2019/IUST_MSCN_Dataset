param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [ValidateSet(1,3,5,7)][int]$BatchSize = 5,
    [int]$Seed = 42,
    [int]$Epochs = 6,
    [string]$OutputRoot = '',
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $runArgs = @('--dataset',$Dataset,'--action',$Action,'--seed',"$Seed",'--batch-size',"$BatchSize",'--epochs',"$Epochs",'--device',$Device)
    if ($OutputRoot) { $runArgs += @('--output-root',$OutputRoot) }
    & python AMCAL/amcal_synthetic_paironly.py @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Synthetic PairOnly pipeline failed ($Dataset / $Action)." }
} finally { Pop-Location }
