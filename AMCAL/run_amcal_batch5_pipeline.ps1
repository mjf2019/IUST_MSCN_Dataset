param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('All','Train','Evaluate','Audit')][string]$Action = 'All',
    [int]$Seed = 42,
    [switch]$LightCNN,
    [double]$Bonus = 0,
    [double]$OnlineContextLR = 0.0005,
    [double]$ReplayWeight = -1,
    [ValidateSet(3,5,7)][int]$BatchSize = 5,
    [int]$Epochs = 12,
    [int]$Patience = 5,
    [string]$SourceRun = '',
    [string]$OutputRoot = '',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda',
    [string]$EvaluationTag = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $runArgs = @('--dataset',$Dataset,'--action',$Action,'--seed',"$Seed",'--device',$Device,
        '--batch-size',"$BatchSize",'--epochs',"$Epochs",'--patience',"$Patience")
    if ($LightCNN) { $runArgs += '--light-cnn' }
    $runArgs += @('--query-budget-bonus',$Bonus.ToString([Globalization.CultureInfo]::InvariantCulture),'--online-context-lr',$OnlineContextLR.ToString('0.##########',[Globalization.CultureInfo]::InvariantCulture))
    if ($ReplayWeight -ge 0) { $runArgs += @('--replay-weight',$ReplayWeight.ToString([Globalization.CultureInfo]::InvariantCulture)) }
    if ($SourceRun) { $runArgs += @('--source-run',$SourceRun) }
    if ($OutputRoot) { $runArgs += @('--output-root',$OutputRoot) }
    if ($EvaluationTag) { $runArgs += @('--evaluation-tag',$EvaluationTag) }
    & python AMCAL/amcal_batch5_pipeline.py @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Batch5 pipeline failed ($Dataset / $Action)." }
} finally { Pop-Location }
