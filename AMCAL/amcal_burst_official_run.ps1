param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('IAT','PS')][string]$Representation,
    [int]$Seed = 42,
    [double]$OnlineLR = 0.0001,
    [int]$ContextReplaySize = 0,
    [ValidateSet("absolute","relative")][string]$RoutingRewardMode = "absolute",
    [switch]$RequirePrepared,
    [switch]$EvaluateOnly
)
$ErrorActionPreference = 'Stop'
$dataset = "AMCAL/ANT_Burst_${Representation}_TEST/Dataset_official_v1"
$burst = "$dataset/AdvBurst_FTSC-$Representation"
$run = "AMCAL/ANT_Burst_${Representation}_TEST/protocol_runs/three_action_official_levels3_12_lr0p0001_seed$Seed"
if ($ContextReplaySize -gt 0 -or $OnlineLR -ne 0.0001) { $run += "_replay${ContextReplaySize}_lr$OnlineLR" }
if ($RoutingRewardMode -ne "absolute") { $run += "_routing$RoutingRewardMode" }
$cache = "AMCAL/ANT_Burst_TEST/official_source_cache/$Representation"
if (Test-Path "$dataset/manifest.json") {
    Write-Host "Using prepared $Representation data: $dataset"
} else {
    if ($RequirePrepared) { throw "Prepared data missing: $dataset ; downloads disabled." }
    python AMCAL/amcal_burst_prepare_official.py --representation $Representation --cache $cache --output $dataset
    if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed' }
}
if (-not $EvaluateOnly) {
python AMCAL/amcal_burst_official_protocol.py train --burst-dir $burst --burst-mapping-dir $burst --official-split-dir $dataset --output $run --seed $Seed --train-perturbations 0 3 12 --selection-cost 0.0001 --gain-weight 1 --retention-weight 1 --reference-batch-size 32 --reference-history-size 256 --budget-fraction 0.2 --train-online-lr $OnlineLR --context-replay-size $ContextReplaySize --context-current-weight 0.5 --routing-reward-mode $RoutingRewardMode --base-epochs 100 --context-epochs 100 --pretrain-epochs 20 --base-patience 15 --context-patience 15
if ($LASTEXITCODE -ne 0) { throw 'Training failed' }
} elseif (-not (Test-Path "$run/Models/adversarial_best_dqn.pth")) {
    throw "Training checkpoint missing: $run"
}
foreach ($mode in @('learned','random')) {
    foreach ($level in @(1,5,7,10,15,17,20)) {
        $stem = "AdvBurstTrainedBurst_size_${level}_Step_2000"
        python AMCAL/amcal_burst_official_protocol.py evaluate --burst-dir $burst --burst-mapping-dir $burst --official-split-dir $dataset --output $run --attack-file "$burst/$stem.csv" --row-map "$burst/${stem}_rows.csv" --selector-mode $mode --eval-epsilon 0.2 --routing-oracle --selection-cost 0.0001 --gain-weight 1 --retention-weight 1 --reference-batch-size 32 --reference-history-size 256 --seed $Seed --lr $OnlineLR --context-replay-size $ContextReplaySize --context-current-weight 0.5 --routing-reward-mode $RoutingRewardMode --budget-fraction 0.2 --stream-order shuffle --forgetting-diagnostics
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $Representation / $mode / level $level" }
    }
}
