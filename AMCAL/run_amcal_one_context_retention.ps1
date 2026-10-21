param(
    [ValidateSet('Both','Sequential','Independent')][string]$EvaluationSet = 'Both',
    [int]$Seed = 42,
    [double]$MemoryWeight = 0.3,
    [string]$OutputRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL one-Context retention evaluation failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path "$OutputRun/Models/final_enhancement.pth" -PathType Leaf)) {
        throw "Trained checkpoint not found under $OutputRun/Models."
    }
    $weightText = $MemoryWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    $common = @('evaluate','--output',$OutputRun,'--burst-dir',$burst,'--context-count','1',
        '--query-schedule','uniform','--update-policy','learned','--modes','learned',
        '--seed',"$Seed",'--device',$Device,'--online-memory-ce-weight',$weightText)
    if ($EvaluationSet -in @('Both','Sequential')) {
        Write-Host "One Context / protected replay weight ${weightText}: sequential forgetting test."
        Invoke-AMCAL -Arguments ($common + @('--continual'))
    }
    if ($EvaluationSet -in @('Both','Independent')) {
        Write-Host "One Context / protected replay weight ${weightText}: independent per-level test."
        Invoke-AMCAL -Arguments $common
    }
} finally { Pop-Location }
