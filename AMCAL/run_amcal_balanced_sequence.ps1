param(
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$Epochs = 2,
    [ValidateSet('policy','uniform')][string]$QuerySchedule = 'uniform',
    [string]$SourceRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_balancedref_lr0p0005_seed42',
    [string]$OutputRun = '',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
if ($Epochs -lt 1) { throw 'Epochs must be positive.' }
if (-not $OutputRun) {
    $OutputRun = "AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_balancedref_sequence_${QuerySchedule}_seed${Seed}"
}
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string]$Script, [string[]]$Arguments) {
    & python $Script @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Balanced sequence failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if ($Action -in @('Train','All')) {
        Invoke-AMCAL -Script 'AMCAL/amcal_balanced_sequence_training.py' -Arguments @(
            '--source-run',$SourceRun,'--output',$OutputRun,'--burst-dir',$burst,
            '--epochs',"$Epochs",'--patience','2','--query-schedule',$QuerySchedule,
            '--seed',"$Seed",'--device',$Device)
    }
    if ($Action -in @('Evaluate','All')) {
        Invoke-AMCAL -Script 'AMCAL/amcal_burst_sequential_retention.py' -Arguments @(
            '--source-run',$OutputRun,'--output',"$OutputRun/Results",'--burst-dir',$burst,
            '--query-schedule',$QuerySchedule,'--modes','learned','--seed',"$Seed",'--device',$Device)
    }
} finally { Pop-Location }
