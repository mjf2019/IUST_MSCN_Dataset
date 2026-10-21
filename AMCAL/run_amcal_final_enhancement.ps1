param(
    [ValidateSet('Train','Evaluate','Audit','All')][string]$Action = 'Train',
    [ValidateSet('accuracy','compact','shared','cascade')][string]$Profile = 'compact',
    [string]$DatasetRepo = (Split-Path -Parent $PSScriptRoot),
    [string]$SourceRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_balancedref_lr0p0005_seed42',
    [string]$OutputRun = '',
    [int]$TrainingSeed = 42,
    [int[]]$Seeds = @(42),
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
if (-not $OutputRun) {
    $OutputRun = "AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_${Profile}_trainseed${TrainingSeed}"
}
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL enhancement failed with exit code $LASTEXITCODE" }
}
Push-Location $DatasetRepo
try {
    if ($Action -in @('Train','All')) {
        Invoke-AMCAL -Arguments @('train','--source-run',$SourceRun,'--output',$OutputRun,
            '--burst-dir',$burst,'--profile',$Profile,'--warm-start-selector',
            '--seed',"$TrainingSeed",'--device',$Device)
    }
    if ($Action -in @('Evaluate','Audit','All')) {
        foreach ($seed in $Seeds) {
            $runArgs = @('evaluate','--output',$OutputRun,'--burst-dir',$burst,
                         '--seed',"$seed",'--device',$Device)
            if ($Action -eq 'Audit') { $runArgs += '--continual' }
            Invoke-AMCAL -Arguments $runArgs
        }
    }
} finally { Pop-Location }
