param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [int]$Seed = 42,
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
Push-Location $repo
try {
    & $python -c 'import torch, numpy, sklearn, joblib'
    if ($LASTEXITCODE -ne 0) { throw 'Required dependencies are missing. Nothing was installed.' }
    & $python -m unittest discover -s AMCAL/tests -p 'test_paironly_context_only.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Context-only preflight failed.' }
    $stem = if ($Dataset -eq 'PS') { 'light_ps_paironly_batch5' } else { 'paironly_batch5' }
    $output = "AMCAL/ANT_Burst_${Dataset}_TEST/protocol_runs/${stem}_trainseed${Seed}/paironly_batch5"
    $burst = "AMCAL/ANT_Burst_${Dataset}_TEST/Dataset_official_v1/AdvBurst_FTSC-${Dataset}"
    if (-not (Test-Path (Join-Path $output 'Models/final_enhancement.pth'))) { throw "Missing local PairOnly checkpoint: $output" }
    Write-Host "Original PairOnly / $Dataset | Context -> frozen CNN | first 20% feedback | batch5 | no retraining"
    $runArgs = @('AMCAL/amcal_final_enhancement.py', 'evaluate', '--output', $output,
        '--burst-dir', $burst, '--levels', '5','7','10','12','15','17','20',
        '--seed', "$Seed", '--device', $Device, '--budget-fraction', '0.2',
        '--context-lr', '0.00025', '--context-count', '1', '--modes', 'learned',
        '--query-schedule', 'policy', '--update-policy', 'learned', '--paid-replay-mode', 'batch5',
        '--query-budget-bonus', '0', '--online-memory-ce-weight', '0.1',
        '--architecture-ablation', 'no_rl', '--context-only-frontloaded', '--endpoint-metrics',
        '--evaluation-tag', "paironly_contextonly_frontloaded_seed${Seed}", '--skip-existing-completed')
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "PairOnly Context-only evaluation failed: $Dataset" }
} finally { Pop-Location }
