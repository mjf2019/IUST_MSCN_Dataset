param(
    [ValidateSet('All','Train','Evaluate','Audit')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$Epochs = 12,
    [int]$Patience = 5,
    [string]$SourceRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_balancedref_lr0p0005_seed42',
    [string]$OutputRun = '',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
if (-not $OutputRun) {
    $OutputRun = "AMCAL/ANT_Burst_IAT_TEST/protocol_runs/paper_original_levels013_seed${Seed}"
}
if ($Action -in @('All','Train') -and ($Epochs -lt 3 -or $Patience -lt 3)) {
    throw 'Use at least three epochs and patience three so all three training levels are visited.'
}
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string]$Script, [string[]]$Arguments) {
    & python $Script @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Original-paper AMCAL experiment failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $common = @('--output',$OutputRun,'--burst-dir',$burst,'--seed',"$Seed",'--device',$Device,
        '--balanced-reward-reference','--query-schedule','policy','--update-policy','learned',
        '--context-lr','0.0005','--context-alpha','0.8','--budget-fraction','0.2',
        '--gamma','0.95','--selection-cost','0.0001','--gain-weight','1','--retention-weight','1',
        '--train-epsilon','0.2','--eval-epsilon','0.05')
    if ($Action -in @('All','Train')) {
        foreach ($name in @('initial_cnn_model.pth','adversarial_best_context.pth','protocol_preprocessing.joblib')) {
            if (-not (Test-Path "$SourceRun/Models/$name" -PathType Leaf)) {
                throw "Source artifact missing: $SourceRun/Models/$name"
            }
        }
        Invoke-AMCAL -Script 'AMCAL/amcal_burst_structured_bank_protocol.py' -Arguments (
            @('train','--source-run',$SourceRun,'--levels','0','1','3',
              '--epochs',"$Epochs",'--patience',"$Patience",'--router-warmup-episodes','3') + $common)
    }
    if ($Action -in @('All','Evaluate','Audit')) {
        if (-not (Test-Path "$OutputRun/Models/structured_bank_ddqn.pth" -PathType Leaf)) {
            throw "Trained selector not found under $OutputRun/Models."
        }
        if ($Action -in @('All','Evaluate')) {
            Invoke-AMCAL -Script 'AMCAL/amcal_burst_structured_bank_protocol.py' -Arguments (
                @('evaluate','--levels','5','7','10','12','15','17','20','--modes','learned') + $common)
        }
        if ($Action -in @('All','Audit')) {
            Invoke-AMCAL -Script 'AMCAL/amcal_burst_sequential_retention.py' -Arguments @(
                '--source-run',$OutputRun,'--output',"$OutputRun/Results/Sequential_seed${Seed}",
                '--burst-dir',$burst,'--levels','5','7','10','12','15','17','20',
                '--query-schedule','policy','--modes','learned','--audit-fraction','0.2',
                '--seed',"$Seed",'--device',$Device)
        }
    }
} finally { Pop-Location }
