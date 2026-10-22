param(
    [ValidateSet('IAT','PS')][string]$Dataset = 'IAT',
    [ValidateSet('sequential','batch','batch_all','batch5','batch8')][string]$Mode = 'sequential',
    [int]$Seed = 42,
    [string]$OutputRun = '',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
if (-not $OutputRun) {
    if ($Dataset -eq 'IAT') {
        $OutputRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_query_credit_paironly_trainseed42'
    } else {
        $OutputRun = "AMCAL/ANT_Burst_PS_TEST/protocol_runs/paironly_pipeline_seed${Seed}/paironly"
    }
}
$burst = "AMCAL/ANT_Burst_${Dataset}_TEST/Dataset_official_v1/AdvBurst_FTSC-${Dataset}"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path "$OutputRun/Models/final_enhancement.pth" -PathType Leaf)) {
        throw "Existing PairOnly checkpoint required: $OutputRun/Models/final_enhancement.pth"
    }
    $runArgs = @('evaluate','--output',$OutputRun,'--burst-dir',$burst,
        '--context-count','1','--query-schedule','policy','--update-policy','learned',
        '--modes','learned','--seed',"$Seed",'--device',$Device,
        '--levels','5','7','10','12','15','17','20',
        '--context-lr','0.0005','--gamma','0.95','--eval-epsilon','0.05',
        '--budget-fraction','0.2','--paid-replay-mode',$Mode,
        '--routing-oracle-audit','--skip-existing-completed',
        '--evaluation-tag',"paid_replay_${Dataset}_${Mode}_seed${Seed}")
    & python AMCAL/amcal_final_enhancement.py @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'Independent evaluation failed.' }
    & python AMCAL/amcal_final_enhancement.py @runArgs --continual
    if ($LASTEXITCODE -ne 0) { throw 'Sequential evaluation failed.' }
} finally { Pop-Location }
