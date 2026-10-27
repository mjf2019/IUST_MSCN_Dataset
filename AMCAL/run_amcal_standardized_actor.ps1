param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('Natural','Balanced','Both')][string]$Control = 'Both',
    [ValidateSet('Train','Evaluate','All')][string]$Action = 'Train',
    [int]$Seed = 42,
    [int]$Epochs = 6,
    [ValidateRange(0,2147483647)][int]$PilotRows = 0,
    [ValidateRange(0,10)][double]$ActorLogitPenalty = 0,
    [string]$CacheRun = '',
    [string]$SourceRun = '',
    [switch]$UniformQueries,
    [switch]$UncertaintyQueries,
    [ValidateRange(1,99)][double]$UncertaintyPercentile = 75,
    [switch]$UniformTraining,
    [switch]$QueryUtilityTarget,
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
if ($UniformQueries -and $UncertaintyQueries) { throw 'Choose UniformQueries or UncertaintyQueries, not both.' }
if ($UncertaintyQueries -and $Action -ne 'Evaluate') { throw 'UncertaintyQueries requires Action Evaluate; no retraining.' }
if ($UniformQueries -and $Action -ne 'Evaluate') { throw 'UniformQueries requires Action Evaluate. No training is performed by this control.' }
if ($ActorLogitPenalty -gt 0 -and $Control -ne 'Natural') { throw 'ActorLogitPenalty is a separate Natural control; use Control Natural.' }
if ($QueryUtilityTarget -and ($Control -ne 'Natural' -or $ActorLogitPenalty -ne 0)) { throw 'QueryUtilityTarget requires Control Natural and zero ActorLogitPenalty.' }
$repo = Split-Path -Parent $PSScriptRoot
$culture = [Globalization.CultureInfo]::InvariantCulture
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
Push-Location $repo
try {
    & $python -c 'import torch, numpy, sklearn, joblib'
    if ($LASTEXITCODE -ne 0) { throw 'Required dependencies are missing. Nothing was installed.' }
    & $python -m unittest discover -s AMCAL/tests -p 'test_minimax*.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Actor experiment preflight checks failed.' }
    if (-not $CacheRun) {
        $CacheRun = "AMCAL/ANT_Burst_${Dataset}_TEST/protocol_runs/minimax_diagnostics_seed${Seed}_n256"
    }
    if ($Action -ne 'Evaluate' -and -not (Test-Path (Join-Path $CacheRun 'TRAIN/actor_cache.pt'))) {
        throw 'Local TRAIN/actor_cache.pt is required. JSON results alone cannot fit the normalizer.'
    }
    $controls = if ($Control -eq 'Both') { @('natural','balanced') } else { @($Control.ToLowerInvariant()) }
    foreach ($item in $controls) {
        if ($UniformQueries) { Write-Host "Uniform-query TEST: $Dataset / $item | exact 20% cap | no retraining" }
        if ($UncertaintyQueries) { Write-Host "Uncertainty-query TEST: $Dataset / $item | VAL q$UncertaintyPercentile | maximum 20% cap | no retraining" }
        $costText = if ($QueryUtilityTarget) { '0' } else { '0.05' }
        $output = "AMCAL/ANT_Burst_${Dataset}_TEST/protocol_runs/minimax_query_batch5_seed${Seed}_v2_h16_c${costText}_actorstd_${item}"
        if ($UniformTraining) {
            $output += '_trainuniform'
            Write-Host "Uniform TRAIN collection: $Dataset / $item | VAL uses Actor requests | no CNN retraining"
        }
        if ($PilotRows -gt 0) {
            $output += "_pilot${PilotRows}_ep${Epochs}"
            Write-Host "Small-data diagnostic: at most $PilotRows rows per TRAIN/VAL/TEST domain; $Epochs epochs"
        }
        if ($ActorLogitPenalty -gt 0) {
            $penaltyText = $ActorLogitPenalty.ToString($culture)
            $output += '_logitreg' + $penaltyText.Replace('.', 'p')
            Write-Host "TRAIN Actor logit penalty: $penaltyText | signed target unchanged | policy threshold 0.5"
        }
        if ($QueryUtilityTarget) {
            $output += '_utility'
            Write-Host 'Query target: G-F | cost 0 | no logit penalty | Context update unchanged'
        }
        $runArgs = @('AMCAL/amcal_minimax_paironly.py', '--dataset', $Dataset,
            '--action', $Action, '--seed', "$Seed", '--device', $Device,
            '--epochs', "$Epochs", '--horizon', '16', '--cost', $costText,
            '--gain-weight', '1', '--forgetting-weight', '1', '--reference-size', '32',
            '--budget-fraction', '0.2', '--context-lr', '0.00025', '--query-lr', '0.0001',
            '--replay-weight', '0.1', '--output', $output, '--actor-control', $item)
        if ($Action -ne 'Evaluate') { $runArgs += @('--actor-train-cache', $CacheRun, '--train-diagnostics') }
        if ($SourceRun) { $runArgs += @('--source-run', $SourceRun) }
        if ($UniformQueries) { $runArgs += @('--eval-query-schedule', 'uniform') }
        if ($UncertaintyQueries) { $runArgs += @('--eval-query-schedule', 'uncertainty', '--uncertainty-percentile', $UncertaintyPercentile.ToString($culture)) }
        if ($UniformTraining) { $runArgs += @('--train-query-schedule', 'uniform') }
        if ($PilotRows -gt 0) { $runArgs += @('--pilot-rows', "$PilotRows") }
        if ($ActorLogitPenalty -gt 0) { $runArgs += @('--actor-logit-penalty', $penaltyText) }
        if ($QueryUtilityTarget) { $runArgs += @('--query-target', 'update_utility') }
        & $python @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Standardized Actor failed: $Dataset / $item / $Action" }
    }
} finally { Pop-Location }



