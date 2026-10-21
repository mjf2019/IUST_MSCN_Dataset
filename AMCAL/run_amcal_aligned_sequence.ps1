param(
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$Epochs = 2,
    [string]$SourceRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_trainseed42',
    [string]$OutputRun = '',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
if ($Epochs -lt 1) { throw 'Epochs must be positive.' }
if (-not $OutputRun) {
    $OutputRun = "AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_aligned_trainseed${Seed}"
}
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL aligned sequence failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if ($Action -in @('Train','All')) {
        Invoke-AMCAL -Arguments @('train','--source-run',$SourceRun,'--output',$OutputRun,
            '--burst-dir',$burst,'--profile','accuracy','--warm-start-enhancement','--aligned-sequence',
            '--query-schedule','uniform','--update-policy','learned',
            '--epochs',"$Epochs",'--patience','2','--seed',"$Seed",'--device',$Device)
    }
    if ($Action -in @('Evaluate','All')) {
        Invoke-AMCAL -Arguments @('evaluate','--output',$OutputRun,'--burst-dir',$burst,
            '--continual','--query-schedule','uniform','--update-policy','learned',
            '--modes','learned','--seed',"$Seed",'--device',$Device)
    }
} finally { Pop-Location }
