param(
    [ValidateSet('PairOnly','Connected','LongHorizon')][string]$Variant = 'PairOnly',
    [ValidateSet('All','Train','Evaluate','Audit')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$Epochs = 12,
    [int]$Patience = 5,
    [string]$SourceRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42',
    [string]$OutputRun = '',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$variantName = $Variant.ToLowerInvariant()
$discount = if ($Variant -eq 'LongHorizon') { '0.99' } else { '0.95' }
if (-not $OutputRun) {
    $OutputRun = "AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_query_credit_${variantName}_trainseed${Seed}"
}
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL one-Context query-value-only training experiment failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $common = @('--output',$OutputRun,'--burst-dir',$burst,'--context-count','1',
        '--query-schedule','policy','--update-policy','learned','--seed',"$Seed",'--device',$Device,
        '--eval-epsilon','0.05','--gamma',$discount,'--budget-fraction','0.2','--context-lr','0.0005')
    if ($Action -in @('All','Train')) {
        if (-not (Test-Path "$SourceRun/Models/final_enhancement.pth" -PathType Leaf)) {
            throw "Source enhancement checkpoint not found under $SourceRun/Models."
        }
        $trainingFlags = @('--paired-query-only-training')
        if ($Variant -in @('Connected','LongHorizon')) { $trainingFlags += '--connected-sequence-td' }
        Invoke-AMCAL -Arguments (@('train','--source-run',$SourceRun,'--profile','accuracy',
            '--warm-start-enhancement','--aligned-sequence','--one-context-rl-training','--query-value-only-training',
            '--levels','0','1','3','--epochs',"$Epochs",'--patience',"$Patience") + $common + $trainingFlags)
    }
    if ($Action -in @('All','Evaluate','Audit')) {
        if (-not (Test-Path "$OutputRun/Models/final_enhancement.pth" -PathType Leaf)) {
            throw "Trained checkpoint not found under $OutputRun/Models."
        }
        $evaluation = @('evaluate','--modes','learned') + $common
        if ($Action -in @('All','Evaluate')) {
            Invoke-AMCAL -Arguments $evaluation
        }
        if ($Action -in @('All','Audit')) {
            Invoke-AMCAL -Arguments ($evaluation + @('--continual'))
        }
    }
} finally { Pop-Location }
