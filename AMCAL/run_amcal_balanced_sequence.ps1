param(
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$Epochs = 2,
    [ValidateSet('policy','uniform')][string]$QuerySchedule = 'uniform',
    [ValidateSet('Sequential','Independent','Both')][string]$EvaluationSet = 'Sequential',
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
        if ($EvaluationSet -in @('Sequential','Both')) {
            $sequentialOutput = "$OutputRun/Results"
            if ($EvaluationSet -eq 'Both') { $sequentialOutput = "$OutputRun/Results/Sequential" }
            Invoke-AMCAL -Script 'AMCAL/amcal_burst_sequential_retention.py' -Arguments @(
                '--source-run',$OutputRun,'--output',$sequentialOutput,'--burst-dir',$burst,
                '--query-schedule',$QuerySchedule,'--modes','learned','--seed',"$Seed",'--device',$Device)
        }
        if ($EvaluationSet -in @('Independent','Both')) {
            Invoke-AMCAL -Script 'AMCAL/amcal_burst_structured_bank_protocol.py' -Arguments @(
                'evaluate','--output',$OutputRun,'--burst-dir',$burst,
                '--balanced-reward-reference','--query-schedule',$QuerySchedule,'--update-policy','learned',
                '--context-lr','0.0005','--budget-fraction','0.2','--eval-epsilon','0.05',
                '--modes','learned','--seed',"$Seed",'--device',$Device)
        }
    }
} finally { Pop-Location }
