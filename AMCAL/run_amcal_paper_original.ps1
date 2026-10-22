param(
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$BaseEpochs = 200,
    [int]$ContextEpochs = 300,
    [int]$Patience = 30,
    [string]$SourceRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_balancedref_lr0p0005_seed42',
    [string]$OutputRun = '',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
if (-not $OutputRun) {
    $OutputRun = "AMCAL/ANT_Burst_IAT_TEST/protocol_runs/paper_main_legacy_levels013_seed${Seed}"
}
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_paper_legacy_levels013.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Existing binary-paper AMCAL run failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $common = @('--source-run',$SourceRun,'--output',$OutputRun,'--burst-dir',$burst,
        '--seed',"$Seed",'--device',$Device)
    if ($Action -in @('All','Train')) {
        Invoke-AMCAL -Arguments (@('train','--base-epochs',"$BaseEpochs",
            '--context-epochs',"$ContextEpochs",'--patience',"$Patience") + $common)
    }
    if ($Action -in @('All','Evaluate')) {
        Invoke-AMCAL -Arguments (@('evaluate') + $common)
    }
} finally { Pop-Location }
