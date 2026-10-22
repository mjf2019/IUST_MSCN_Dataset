param(
    [double]$Bonus = 2.0,
    [double]$ReplayWeight = -1,
    [int]$Seed = 42,
    [string]$Run = 'AMCAL/ANT_Burst_PS_TEST/protocol_runs/light_ps_paironly_batch5_trainseed42/paironly_batch5',
    [ValidateSet('cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $value = $Bonus.ToString([Globalization.CultureInfo]::InvariantCulture)
    $tag = "budget_bonus_$value"
    if ($ReplayWeight -ge 0) { $tag += "_replay_" + $ReplayWeight.ToString([Globalization.CultureInfo]::InvariantCulture) }
    $runArgs = @('evaluate','--output',$Run,'--burst-dir','AMCAL/ANT_Burst_PS_TEST/Dataset_official_v1/AdvBurst_FTSC-PS',
        '--context-count','1','--modes','learned','--query-schedule','policy','--update-policy','learned',
        '--paid-replay-mode','batch5','--context-lr','0.0005','--budget-fraction','0.2','--eval-epsilon','0.05',
        '--levels','5','7','10','12','15','17','20','--seed',"$Seed",'--device',$Device,
        '--query-budget-bonus',$value,'--routing-oracle-audit','--evaluation-tag',$tag,'--skip-existing-completed')
    if ($ReplayWeight -ge 0) { $runArgs += @('--online-memory-ce-weight',$ReplayWeight.ToString([Globalization.CultureInfo]::InvariantCulture)) }
    & python AMCAL/amcal_final_enhancement.py @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'Independent evaluation failed.' }
    & python AMCAL/amcal_final_enhancement.py @runArgs --continual
    if ($LASTEXITCODE -ne 0) { throw 'Continual evaluation failed.' }
} finally { Pop-Location }
