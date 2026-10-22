param(
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$MaxFeatures = 64,
    [string]$Csv = '',
    [string]$Output = '',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $runArgs = @('--action',$Action,'--seed',"$Seed",'--device',$Device,'--max-features',"$MaxFeatures")
    if ($Csv) { $runArgs += @('--csv',$Csv) }
    if ($Output) { $runArgs += @('--output',$Output) }
    & python AMCAL/amcal_tun_paironly.py @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'TUN PairOnly pipeline failed.' }
} finally { Pop-Location }
