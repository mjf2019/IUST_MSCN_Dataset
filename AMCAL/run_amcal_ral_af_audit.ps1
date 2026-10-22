param(
    [ValidateSet('IAT','PS')][string]$Dataset = 'IAT',
    [ValidateSet('All','RAL','AF')][string]$Method = 'All',
    [int]$Seed = 42,
    [int[]]$Shots = @(5),
    [int[]]$Levels = @(5,7,10,12,15,17,20),
    [int]$TestPerClass = 0,
    [string]$SourceRun = '',
    [string]$BurstDir = '',
    [string]$OutputRoot = '',
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $base = "AMCAL/ANT_Burst_${Dataset}_TEST"
    if (-not $SourceRun) { $SourceRun = "$base/protocol_runs/structured_bank3_balancedref_lr0p0005_seed42" }
    if (-not $BurstDir) { $BurstDir = "$base/Dataset_official_v1/AdvBurst_FTSC-$Dataset" }
    if (-not $OutputRoot) { $OutputRoot = "$base/protocol_runs/ral_af_fidelity_seed${Seed}_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')" }
    if ($Method -in @('All','RAL')) {
        $common = @('--burst-dir',$BurstDir,'--output',"$OutputRoot/ral_matched_source",'--methods','ral',
                    '--seed',"$Seed",'--budget-fraction','0.2','--device','cpu','--ral-budget-mode','prefix')
        & python 'AMCAL/amcal_burst_baselines.py' train @common '--source-run' $SourceRun
        if ($LASTEXITCODE -ne 0) { throw "RAL training failed: $LASTEXITCODE" }
        & python 'AMCAL/amcal_burst_baselines.py' evaluate @common '--levels' @Levels
        if ($LASTEXITCODE -ne 0) { throw "RAL evaluation failed: $LASTEXITCODE" }
    }
    if ($Method -in @('All','AF')) {
        & python 'AMCAL/amcal_af_paper_burst.py' '--source-run' $SourceRun '--burst-dir' $BurstDir `
            '--output' "$OutputRoot/af_fewshot" '--seed' $Seed '--device' $Device `
            '--shots' @Shots '--levels' @Levels '--test-per-class' $TestPerClass
        if ($LASTEXITCODE -ne 0) { throw "AF few-shot evaluation failed: $LASTEXITCODE" }
    }
    Write-Host "Saved new audit outputs: $OutputRoot"
    Write-Host 'RAL: online with matched source training. AF: offline target calibration; compare protocols separately.'
} finally { Pop-Location }
