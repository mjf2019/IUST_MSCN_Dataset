param(
    [ValidateSet('All','Train','Evaluate','Audit')][string]$Action = 'All',
    [int]$Seed = 42,
    [string]$SourceRun = 'AMCAL/ANT_Burst_PS_TEST/protocol_runs/structured_bank3_balancedref_lr0p0005_seed42',
    [string]$BurstDir = 'AMCAL/ANT_Burst_PS_TEST/Dataset_official_v1/AdvBurst_FTSC-PS',
    [string]$OutputRoot = '',
    [int]$BaseEpochs = 12,
    [int]$UtilityEpochs = 3,
    [int]$PairEpochs = 12,
    [int]$Patience = 5,
    [string]$EvaluationTag = '',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $arguments = @('--action',$Action,'--seed',"$Seed",'--source-run',$SourceRun,
        '--burst-dir',$BurstDir,'--device',$Device,'--base-epochs',"$BaseEpochs",
        '--utility-epochs',"$UtilityEpochs",'--pair-epochs',"$PairEpochs",'--patience',"$Patience")
    if ($OutputRoot) { $arguments += @('--output-root',$OutputRoot) }
    if ($EvaluationTag) { $arguments += @('--evaluation-tag',$EvaluationTag) }
    & python 'AMCAL/amcal_paironly_burst_ps.py' @arguments
    if ($LASTEXITCODE -ne 0) { throw "PS PairOnly pipeline failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
