param(
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$PilotRows = 300,
    [int]$Epochs = 3,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$SourceRun = ''
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    $python = Join-Path $repoRoot 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    & $python -m unittest discover -s AMCAL/tests -p test_cnn_variant.py -v
    if ($LASTEXITCODE -ne 0) { throw 'CNN variant checks failed' }
    & $python -m unittest discover -s AMCAL/tests -p test_two_action_ddqn_light_iat.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Two-action DDQN checks failed' }
    $runs = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs'
    if (-not $SourceRun) { $SourceRun = "$runs/light_iat_paironly_batch5_trainseed${Seed}/paironly_batch5" }
    $cache = "$runs/light_iat_minimax_diagnostics_seed${Seed}_n256"
    $teacher = "$runs/light_iat_minimax_query_batch5_seed${Seed}_v2_h16_c0.05_actorstd_natural"
    $actor = "$runs/light_iat_competitive_selector_seed${Seed}_p300_r2_v1_paper_reward"
    $common = @('--dataset','IAT','--light-cnn','--seed',"$Seed",'--device',$Device)
    if ($Action -ne 'Evaluate') {
        if (-not (Test-Path "$SourceRun/Models/final_enhancement.pth")) {
            throw "Completed Light IAT PairOnly source is required: $SourceRun. Supply -SourceRun for another completed Light IAT source."
        }
        Write-Host 'Light IAT preparation 1/3: independent TRAIN normalizer cache; no TEST scoring'
        & $python AMCAL/amcal_minimax_paironly_light_iat.py @common --action Diagnose --diagnostic-phase Collect --source-run $SourceRun --output $cache --diagnostic-rows 256 --horizon 16 --reference-size 32 --context-lr 0.00025 --replay-weight 0.1 --budget-fraction 0.2
        if ($LASTEXITCODE -ne 0) { throw 'Light IAT TRAIN cache failed' }
        Write-Host 'Light IAT preparation 2/3: healthy Context training, same six-epoch Natural procedure; frozen Light CNN'
        & $python AMCAL/amcal_minimax_paironly_light_iat.py @common --action Train --source-run $SourceRun --output $teacher --actor-train-cache $cache --actor-control natural --train-diagnostics --epochs 6 --horizon 16 --reference-size 32 --cost 0.05 --gain-weight 1 --forgetting-weight 1 --context-lr 0.00025 --query-lr 0.0001 --replay-weight 0.1 --budget-fraction 0.2
        if ($LASTEXITCODE -ne 0) { throw 'Light IAT healthy Context training failed' }
        Write-Host 'Light IAT preparation 3/3: matching compact paper-reward Actor; 300-row pilot'
        & $python AMCAL/amcal_competitive_selector_light_iat.py @common --action Train --teacher $teacher --output $actor --original-paper-reward --pilot-rows 300 --rounds 2
        if ($LASTEXITCODE -ne 0) { throw 'Light IAT parent Actor failed' }
    }
    Write-Host "Light IAT DDQN: $Action / $Epochs epochs / $PilotRows rows per domain / 20% cap / Context -> Frozen"
    & $python AMCAL/amcal_two_action_ddqn_light_iat.py @common --action $Action --teacher $teacher --actor-run $actor --pilot-rows $PilotRows --epochs $Epochs
    if ($LASTEXITCODE -ne 0) { throw 'Light IAT two-action DDQN failed' }
} finally { Pop-Location }
