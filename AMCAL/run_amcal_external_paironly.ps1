param(
    [Parameter(Mandatory=$true)][ValidateSet('NPBO','DReLAB')][string]$Dataset,
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [int]$Seed = 42,
    [double]$Fraction = 0.1,
    [int]$MaxTrainRows = 0,
    [int]$MaxStreamRows = 0,
    [int]$MaxAuditRows = 0,
    [int]$WarmEpochs = 3,
    [int]$PairEpochs = 6,
    [string]$Output = '',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $runArgs = @('--dataset',$Dataset,'--action',$Action,'--seed',"$Seed",'--device',$Device,
        '--fraction',$Fraction.ToString([Globalization.CultureInfo]::InvariantCulture),
        '--max-train-rows',"$MaxTrainRows",'--max-stream-rows',"$MaxStreamRows",'--max-audit-rows',"$MaxAuditRows",
        '--warm-epochs',"$WarmEpochs",'--pair-epochs',"$PairEpochs")
    if ($Output) { $runArgs += @('--output',$Output) }
    & python AMCAL/amcal_external_paironly.py @runArgs
    if ($LASTEXITCODE -ne 0) { throw "External PairOnly pipeline failed: $Dataset" }
} finally { Pop-Location }
