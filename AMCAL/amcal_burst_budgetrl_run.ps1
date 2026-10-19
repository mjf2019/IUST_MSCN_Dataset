param(
    [Parameter(Mandatory=$true)][ValidateSet('train','evaluate')][string]$Command,
    [ValidateSet('IAT','PS')][string]$Representation = 'IAT',
    [int]$TrainSeed = 42,
    [int]$Seed = 48,
    [double]$ContextLr = 0.0005,
    [double]$EvalEpsilon = 0.1,
    [int]$Epochs = 12,
    [int]$Patience = 5,
    [string]$SourceRun = '',
    [string]$OutputRun = '',
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$dataset = Join-Path $root "AMCAL/ANT_Burst_${Representation}_TEST/Dataset_official_v1"
$burst = Join-Path $dataset "AdvBurst_FTSC-$Representation"
$runs = Join-Path $root "AMCAL/ANT_Burst_${Representation}_TEST/protocol_runs"
$lr = $ContextLr.ToString('G', [Globalization.CultureInfo]::InvariantCulture)
$epsilon = $EvalEpsilon.ToString('G', [Globalization.CultureInfo]::InvariantCulture)
$lrTag = $lr.Replace('.', 'p')
if (-not $SourceRun) {
    $SourceRun = Join-Path $runs 'structured_bank3_policy_lr0p0005_seed42'
}
if (-not $OutputRun) {
    $OutputRun = Join-Path $runs "structured_bank3_budgetrl_v1_lr${lrTag}_seed$TrainSeed"
}
if (-not (Test-Path (Join-Path $burst 'AdvBurstTrainedBurst_size_0_Step_2000.csv'))) {
    throw "Prepared official data missing: $burst. This runner does not download data."
}
$pythonArgs = @(
    (Join-Path $PSScriptRoot 'amcal_burst_structured_bank_protocol.py'), $Command,
    '--burst-dir', $burst, '--output', $OutputRun,
    '--budget-aware-policy', '--query-schedule', 'policy', '--update-policy', 'learned',
    '--context-lr', $lr, '--budget-fraction', '0.2', '--gamma', '0.99',
    '--selection-cost', '0.0001', '--gain-weight', '1', '--retention-weight', '1',
    '--device', $Device
)
if ($Command -eq 'train') {
    if (-not (Test-Path (Join-Path $SourceRun 'Models/structured_bank_ddqn.pth'))) {
        throw "Trained structured source missing: $SourceRun. Use -SourceRun for an existing source run."
    }
    if (Test-Path $OutputRun) { throw "Output already exists; use a fresh -OutputRun: $OutputRun" }
    $pythonArgs += @('--source-run', $SourceRun, '--levels', '0', '1', '3',
        '--seed', "$TrainSeed", '--epochs', "$Epochs", '--patience', "$Patience",
        '--router-warmup-episodes', '1', '--train-epsilon', '0.2', '--eval-epsilon', '0')
} else {
    if (-not (Test-Path (Join-Path $OutputRun 'Models/structured_bank_ddqn.pth'))) {
        throw "Train this new budget policy first: $OutputRun"
    }
    $pythonArgs += @('--levels', '5', '7', '10', '12', '15', '17', '20',
        '--seed', "$Seed", '--eval-epsilon', $epsilon, '--modes', 'learned', 'random')
}
Write-Host "$Command | $Representation | run: $OutputRun"
& python @pythonArgs
if ($LASTEXITCODE -ne 0) { throw "Budget RL $Command failed: $Representation" }
