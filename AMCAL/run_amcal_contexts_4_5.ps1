param(
    [int[]]$ContextCounts = @(4,5),
    [ValidateSet('Both','Sequential','Independent')][string]$EvaluationSet = 'Both',
    [ValidateSet('uniform','policy')][string]$QuerySchedule = 'uniform',
    [int]$Seed = 42,
    [string]$OutputRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_counterfactual_trainseed42',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
if ($ContextCounts.Count -eq 0 -or @($ContextCounts | Where-Object { $_ -notin @(4,5) }).Count -gt 0) {
    throw 'ContextCounts must contain 4 and/or 5.'
}
if (@($ContextCounts | Select-Object -Unique).Count -ne $ContextCounts.Count) {
    throw 'ContextCounts must not contain duplicates.'
}
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL Context-count evaluation failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path "$OutputRun/Models/final_enhancement.pth" -PathType Leaf)) {
        throw "Trained checkpoint not found under $OutputRun/Models."
    }
    foreach ($count in $ContextCounts) {
        $common = @('evaluate','--output',$OutputRun,'--burst-dir',$burst,'--context-count',"$count",
            '--query-schedule',$QuerySchedule,'--update-policy','learned','--modes','learned',
            '--seed',"$Seed",'--device',$Device)
        if ($EvaluationSet -in @('Both','Sequential')) {
            Write-Host "$count Contexts: sequential forgetting test."
            Invoke-AMCAL -Arguments ($common + @('--continual'))
        }
        if ($EvaluationSet -in @('Both','Independent')) {
            Write-Host "$count Contexts: independent per-level test."
            Invoke-AMCAL -Arguments $common
        }
    }
} finally { Pop-Location }
