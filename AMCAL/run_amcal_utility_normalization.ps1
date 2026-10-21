param(
    [ValidateSet('All','Calibrate','Evaluate')][string]$Action = 'All',
    [int]$Seed = 42,
    [string]$OutputRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_accuracy_trainseed42',
    [string]$CalibrationPath = '',
    [double]$UtilityWeight = 1.0,
    [double]$UtilityClip = 1.0,
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
if (-not $CalibrationPath) {
    $CalibrationPath = "$OutputRun/Calibration/utility_train_seed${Seed}.json"
}
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL utility normalization failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $common = @('--output',$OutputRun,'--burst-dir',$burst,'--query-schedule','uniform',
        '--update-policy','learned','--seed',"$Seed",'--device',$Device,
        '--utility-calibration',$CalibrationPath)
    if ($Action -in @('All','Calibrate')) {
        Invoke-AMCAL -Arguments (@('calibrate-utility') + $common + @('--levels','0','1','3'))
    }
    if ($Action -in @('All','Evaluate')) {
        $weightText = $UtilityWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        $clipText = $UtilityClip.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        Invoke-AMCAL -Arguments (@('evaluate') + $common + @('--continual','--modes','learned',
            '--utility-weight',$weightText,'--utility-clip',$clipText))
    }
} finally { Pop-Location }
