param(
    [ValidateSet('Both','Sequential','Independent')][string]$EvaluationSet = 'Both',
    [ValidateSet('policy','uniform')][string]$QuerySchedule = 'policy',
    [int]$Seed = 42,
    [string]$OutputRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL two-Context evaluation failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path "$OutputRun/Models/final_enhancement.pth" -PathType Leaf)) {
        throw "Trained checkpoint not found under $OutputRun/Models."
    }
    $common = @('evaluate','--output',$OutputRun,'--burst-dir',$burst,'--context-count','2',
        '--query-schedule',$QuerySchedule,'--update-policy','learned','--modes','learned',
        '--seed',"$Seed",'--device',$Device)
    if ($EvaluationSet -in @('Both','Sequential')) {
        Write-Host 'Two Contexts / five actions: sequential forgetting test.'
        Invoke-AMCAL -Arguments ($common + @('--continual'))
    }
    if ($EvaluationSet -in @('Both','Independent')) {
        Write-Host 'Two Contexts / five actions: independent per-level test.'
        Invoke-AMCAL -Arguments $common
    }
} finally { Pop-Location }
