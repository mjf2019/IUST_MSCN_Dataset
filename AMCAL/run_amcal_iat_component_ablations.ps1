param(
    [ValidateSet('IAT','PS')][string]$Dataset = 'IAT',
    [int]$Seed = 42,
    [ValidateSet('all','no_context','no_rl','frozen')][string]$Component = 'all',
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $runName = if ($Dataset -eq 'PS') { 'light_ps_paironly_batch5_trainseed42' } else { 'paironly_batch5_trainseed42' }
    $run = "AMCAL/ANT_Burst_${Dataset}_TEST/protocol_runs/$runName/paironly_batch5"
    $burstDir = "AMCAL/ANT_Burst_${Dataset}_TEST/Dataset_official_v1/AdvBurst_FTSC-$Dataset"
    $cases = if ($Component -eq 'all') { @('no_context','no_rl','frozen') } else { @($Component) }
    foreach ($case in $cases) {
        $bonus = if ($case -eq 'no_context') { '1' } else { '0' }
        $runArgs = @('evaluate',
            '--output',$run,
            '--burst-dir',$burstDir,
            '--architecture-ablation',$case,'--context-count','1','--modes','learned',
            '--query-schedule','policy','--update-policy','learned','--paid-replay-mode','batch5',
            '--context-lr','0.00025','--online-memory-ce-weight','0.1',
            '--query-budget-bonus',$bonus,'--budget-fraction','0.2','--eval-epsilon','0.05',
            '--levels','5','7','10','12','15','17','20','--seed',"$Seed",'--device',$Device,
            '--routing-oracle-audit','--evaluation-tag',"component_$case",'--skip-existing-completed')
        & python AMCAL/amcal_final_enhancement.py @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Independent component ablation failed: $case" }
        & python AMCAL/amcal_final_enhancement.py @runArgs --continual
        if ($LASTEXITCODE -ne 0) { throw "Continual component ablation failed: $case" }
    }
} finally { Pop-Location }
