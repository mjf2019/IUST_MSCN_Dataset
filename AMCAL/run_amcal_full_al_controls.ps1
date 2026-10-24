param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [ValidateSet('All','Random','Uncertainty')][string]$Control = 'All',
    [int]$Seed = 42,
    [ValidateSet('cuda','cpu','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    if ($Dataset -eq 'IAT') {
        $root = 'AMCAL/ANT_Burst_IAT_TEST'
        $model = 'paironly_batch5_trainseed42'
        $burst = 'AdvBurst_FTSC-IAT'
    } else {
        $root = 'AMCAL/ANT_Burst_PS_TEST'
        $model = 'light_ps_paironly_batch5_trainseed42'
        $burst = 'AdvBurst_FTSC-PS'
    }
    $controls = if ($Control -eq 'All') { @('Random','Uncertainty') } else { @($Control) }
    foreach ($item in $controls) {
        $mode = if ($item -eq 'Random') { 'random_al' } else { 'learned' }
        $tag = 'component_full_' + $item.ToLowerInvariant() + '_seed' + $Seed
        $runArgs = @(
            'AMCAL/amcal_final_enhancement.py', 'evaluate',
            '--output', "$root/protocol_runs/$model/paironly_batch5",
            '--burst-dir', "$root/Dataset_official_v1/$burst",
            '--levels', '5','7','10','12','15','17','20',
            '--seed', "$Seed", '--device', $Device,
            '--budget-fraction', '0.2', '--context-count', '1',
            '--context-lr', '0.00025', '--modes', $mode,
            '--query-schedule', 'policy', '--update-policy', 'learned',
            '--paid-replay-mode', 'batch5', '--query-budget-bonus', '0',
            '--online-memory-ce-weight', '0.1', '--endpoint-metrics',
            '--evaluation-tag', $tag, '--skip-existing-completed'
        )
        if ($item -eq 'Uncertainty') {
            $runArgs += @('--uncertainty-query','--uncertainty-routing')
        }
        & $python @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $Dataset / $item" }
    }
} finally {
    Pop-Location
}
