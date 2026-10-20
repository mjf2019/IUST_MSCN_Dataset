param([double[]]$Alphas = @(0.5,0.6,0.7,0.9,1.0))
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$burst = "AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT"
$source = "AMCAL/ANT_Burst_IAT_TEST/protocol_runs/three_action_twostage_levels1_3_contextpre20_seed42"
foreach ($alpha in $Alphas) {
    $value = $alpha.ToString("0.0", [Globalization.CultureInfo]::InvariantCulture)
    $tag = $value.Replace(".", "p")
    $run = "AMCAL/ANT_Burst_IAT_TEST/protocol_runs/structured_bank3_alpha${tag}_lr0p0005_seed42"
    if (-not (Test-Path "$run/Models/structured_bank_ddqn.pth")) {
        if (Test-Path $run) { throw "Incomplete training directory: $run. Preserve it and choose a fresh output before retraining." }
        python AMCAL/amcal_burst_structured_bank_protocol.py train --source-run $source --burst-dir $burst --output $run --levels 0 1 3 --balanced-reward-reference --context-alpha $value --epochs 12 --patience 5 --router-warmup-episodes 3 --context-lr 0.0005 --budget-fraction 0.2 --query-schedule policy --update-policy learned --gamma 0.95 --selection-cost 0.0001 --gain-weight 1 --retention-weight 1 --train-epsilon 0.2 --eval-epsilon 0 --seed 42 --device cuda
        if ($LASTEXITCODE -ne 0) { throw "Training failed: alpha=$value" }
    }
    python AMCAL/amcal_burst_structured_bank_protocol.py evaluate --burst-dir $burst --output $run --levels 5 7 10 12 15 17 20 --balanced-reward-reference --context-alpha $value --context-lr 0.0005 --budget-fraction 0.2 --query-schedule policy --update-policy learned --gamma 0.95 --selection-cost 0.0001 --gain-weight 1 --retention-weight 1 --eval-epsilon 0.05 --modes learned --seed 42 --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: alpha=$value" }
}
