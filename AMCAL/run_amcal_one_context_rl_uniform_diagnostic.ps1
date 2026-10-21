param(
    [ValidateSet('Both','Independent','Sequential')][string]$EvaluationSet = 'Both',
    [int]$Seed = 42,
    [string]$OutputRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_one_context_rl_trainseed42',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL specialized one-Context uniform evaluation failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path "$OutputRun/Models/final_enhancement.pth" -PathType Leaf)) {
        throw "Specialized checkpoint not found under $OutputRun/Models. Run dedicated one-Context RL training first."
    }
    $common = @('evaluate','--output',$OutputRun,'--burst-dir',$burst,'--context-count','1',
        '--query-schedule','uniform','--update-policy','learned','--modes','learned',
        '--seed',"$Seed",'--device',$Device,'--eval-epsilon','0.05',
        '--budget-fraction','0.2','--context-lr','0.0005')
    if ($EvaluationSet -in @('Both','Independent')) {
        Write-Host 'Specialized one-Context checkpoint / uniform queries: independent per-level test.'
        Invoke-AMCAL -Arguments $common
    }
    if ($EvaluationSet -in @('Both','Sequential')) {
        Write-Host 'Specialized one-Context checkpoint / uniform queries: sequential forgetting test.'
        Invoke-AMCAL -Arguments ($common + @('--continual'))
    }
} finally { Pop-Location }
