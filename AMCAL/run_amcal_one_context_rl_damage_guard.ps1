param(
    [ValidateSet('Both','Sequential','Independent')][string]$EvaluationSet = 'Both',
    [ValidateSet('Both','Baseline','Guard')][string]$Variant = 'Both',
    [int]$Seed = 42,
    [string]$OutputRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL one-Context RL evaluation failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path "$OutputRun/Models/final_enhancement.pth" -PathType Leaf)) {
        throw "Trained checkpoint not found under $OutputRun/Models."
    }
    $common = @('evaluate','--output',$OutputRun,'--burst-dir',$burst,'--context-count','1',
        '--query-schedule','policy','--update-policy','learned','--modes','learned',
        '--seed',"$Seed",'--device',$Device,'--eval-epsilon','0.05',
        '--budget-fraction','0.2','--context-lr','0.0005')
    $variants = if ($Variant -eq 'Both') { @('Baseline','Guard') } else { @($Variant) }
    foreach ($case in $variants) {
        $arguments = @($common)
        if ($case -eq 'Guard') { $arguments += '--context-damage-guard' }
        if ($EvaluationSet -in @('Both','Sequential')) {
            Write-Host "One Context / RL / ${case}: sequential forgetting test."
            Invoke-AMCAL -Arguments ($arguments + @('--continual'))
        }
        if ($EvaluationSet -in @('Both','Independent')) {
            Write-Host "One Context / RL / ${case}: independent per-level test."
            Invoke-AMCAL -Arguments $arguments
        }
    }
} finally { Pop-Location }
