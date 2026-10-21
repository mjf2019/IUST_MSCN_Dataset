param(
    [ValidateSet('Both','Sequential','Independent')][string]$EvaluationSet = 'Both',
    [int]$Seed = 42,
    [string]$OutputRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL RL evaluation failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path "$OutputRun/Models/final_enhancement.pth" -PathType Leaf)) {
        throw "Trained checkpoint not found under $OutputRun/Models. Complete the counterfactual utility training first."
    }
    $common = @('evaluate','--output',$OutputRun,'--burst-dir',$burst,
        '--query-schedule','policy','--update-policy','learned','--modes','learned',
        '--seed',"$Seed",'--device',$Device)
    if ($EvaluationSet -in @('Both','Sequential')) {
        Write-Host 'Sequential forgetting test: retain adaptive state between levels.'
        Invoke-AMCAL -Arguments ($common + @('--continual'))
    }
    if ($EvaluationSet -in @('Both','Independent')) {
        Write-Host 'Independent per-level test: reset adaptive state at each level.'
        Invoke-AMCAL -Arguments $common
    }
} finally { Pop-Location }
