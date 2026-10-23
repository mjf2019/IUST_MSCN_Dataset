param(
    [ValidateSet('IAT','PS')][string]$Dataset = 'IAT',
    [ValidateSet('All','Train','Test')][string]$Action = 'All',
    [int]$Seed = 42,
    [int[]]$TrainLevels = @(0,1,3),
    [int[]]$Levels = @(5,7,10,12,15,17,20),
    [double]$Budget = 0.2,
    [string]$SourceRun = '',
    [string]$Output = '',
    [ValidateSet('river','skmultiflow')][string]$ADWINBackend = 'river'
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$datasetRoot = Join-Path $PSScriptRoot "ANT_Burst_${Dataset}_TEST"
if (-not $SourceRun) {
    $SourceRun = Join-Path $datasetRoot 'protocol_runs/structured_bank3_policy_lr0p0005_seed42'
}
if (-not $Output) {
    $trainTag = $TrainLevels -join '_'
    $budgetTag = [int][math]::Round(100*$Budget)
    $Output = Join-Path $datasetRoot "protocol_runs/ral_cd_train${trainTag}_budget${budgetTag}_seed${Seed}"
}
Push-Location $repoRoot
try {
    & python (Join-Path $PSScriptRoot 'amcal_ral_cd.py') --dataset $Dataset --action $Action `
        --source-run $SourceRun --output $Output --train-levels @TrainLevels --levels @Levels `
        --seed $Seed --budget-fraction $Budget --adwin-backend $ADWINBackend
    if ($LASTEXITCODE -ne 0) { throw "RAL-CD failed: $Dataset / $Action" }
} finally { Pop-Location }
