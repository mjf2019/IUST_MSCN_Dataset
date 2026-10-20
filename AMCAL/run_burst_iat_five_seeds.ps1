param(
    [ValidateSet('AMCAL','Baselines','All')][string]$Group = 'All',
    [string]$DatasetRepo = (Split-Path -Parent $PSScriptRoot),
    [int[]]$Seeds = @(42,48,54,62,72),
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$levels = @(5,7,10,12,15,17,20)
$base = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs'
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
$bank = "$base/structured_bank3_balancedref_lr0p0005_seed42"
$jobs = @(
    @{ Run="$base/baselines_levels1_3_seed42"; Methods=@('cnn','af','dfe') },
    @{ Run="$base/ral_paper_rf10_prefix_levels1_3_seed42"; Methods=@('ral') },
    @{ Run="$base/arf_levels1_3_shared_seed42"; Methods=@('arf') },
    @{ Run="$base/calmid_micfoal_levels1_3_seed42"; Methods=@('calmid','micfoal') }
)
function Find-Metrics([string]$Run,[string]$Method,[int]$Seed,[int]$Level,[bool]$Structured) {
    $prefix = if ($Structured) { "structured_${Method}_per${Level}_seed${Seed}_lr0.0005_learned_*_metrics.json" } else { "${Method}_per${Level}_seed${Seed}_*_metrics.json" }
    foreach ($file in @(Get-ChildItem -Path "$Run/Results/$prefix" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)) {
        try {
            $m = Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json
            $c = $m.execution_config
            if ($null -eq $c -or [int]$c.seed -ne $Seed -or [int]$m.samples -ne 1073 -or [int]$m.label_cap -ne 214) { continue }
            if ($null -eq $m.macro_f1) { continue }
            if ($c.PSObject.Properties.Name -contains 'test_scaler') { continue }
            if ($c.PSObject.Properties.Name -contains 'unseen_rows') { continue }
            if ($Structured) {
                if ($c.query_schedule -ne 'policy' -or $c.update_policy -ne 'learned' -or $c.reward_reference -ne 'class_balanced_macro_v1') { continue }
                if ([double]$c.context_lr -ne 0.0005 -or [double]$c.eval_epsilon -ne 0.05 -or [double]$c.budget_fraction -ne 0.2) { continue }
                if ([double]$c.selection_cost -ne 0.0001 -or [double]$c.gain_weight -ne 1 -or [double]$c.retention_weight -ne 1 -or [double]$c.gamma -ne 0.95) { continue }
                if ($c.bank_utility -or $c.budget_aware_policy -or ($c.PSObject.Properties.Name -contains 'query_prediction')) { continue }
            } else {
                if ([double]$c.budget -ne 0.2 -or [double]$c.af_online_lr -ne 0.00001) { continue }
                if ($Method -eq 'ral' -and ($c.ral_protocol_version -ne 2 -or $c.ral_budget_mode -ne 'prefix')) { continue }
            }
            return $file.FullName
        } catch { continue }
    }
    return $null
}
function Invoke-Evaluation([string[]]$EvalArgs) {
    & python @EvalArgs
    if ($LASTEXITCODE -ne 0) { throw ('Evaluation failed: ' + ($EvalArgs -join ' ')) }
}
Push-Location $DatasetRepo
try {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $bundle = "$base/paper_five_seeds_${Group}_$stamp"
    New-Item -ItemType Directory -Force "$bundle/metrics" | Out-Null
    Start-Transcript -Path "$bundle/console.log" | Out-Null
    try {
        if ($Group -in @('AMCAL','All')) {
            if (-not (Test-Path "$bank/Models/structured_bank_ddqn.pth")) { throw "Missing approved AMCAL checkpoint: $bank" }
            foreach ($seed in $Seeds) {
                $missing = $Force.IsPresent
                foreach ($level in $levels) { foreach ($mode in @('learned','random')) {
                    if (-not (Find-Metrics $bank $mode $seed $level $true)) { $missing = $true }
                }}
                if ($missing) {
                    Invoke-Evaluation -EvalArgs @('AMCAL/amcal_burst_structured_bank_protocol.py','evaluate','--burst-dir',$burst,'--output',$bank,'--levels','5','7','10','12','15','17','20','--balanced-reward-reference','--context-lr','0.0005','--budget-fraction','0.2','--query-schedule','policy','--update-policy','learned','--gamma','0.95','--selection-cost','0.0001','--gain-weight','1','--retention-weight','1','--eval-epsilon','0.05','--modes','learned','random','--seed',"$seed",'--device','cuda')
                } else { Write-Host "AMCAL seed $seed already complete; reusing matching metrics." }
                foreach ($level in $levels) { foreach ($mode in @('learned','random')) {
                    $file = Find-Metrics $bank $mode $seed $level $true
                    if (-not $file) { throw "Missing matching metrics: AMCAL $mode / seed $seed / level $level" }
                    Copy-Item -LiteralPath $file -Destination "$bundle/metrics/amcal_${mode}_seed${seed}_per${level}.json"
                }}
            }
        }
        if ($Group -in @('Baselines','All')) {
            foreach ($job in $jobs) {
                foreach ($method in $job.Methods) {
                    $methodDevice = if ($method -in @('cnn','af','dfe')) { 'cuda' } else { 'cpu' }
                    if (-not (Test-Path "$($job.Run)/Models/$method.joblib")) { throw "Missing trained baseline: $method in $($job.Run)" }
                    foreach ($seed in $Seeds) {
                        $missing = $Force.IsPresent
                        foreach ($level in $levels) { if (-not (Find-Metrics $job.Run $method $seed $level $false)) { $missing = $true } }
                        if ($missing) {
                            Invoke-Evaluation -EvalArgs @('AMCAL/amcal_burst_baselines.py','evaluate','--burst-dir',$burst,'--output',$job.Run,'--methods',$method,'--levels','5','7','10','12','15','17','20','--budget-fraction','0.2','--af-lr','0.00001','--ral-budget-mode','prefix','--seed',"$seed",'--device',$methodDevice)
                        } else { Write-Host "$method seed $seed already complete; reusing matching metrics." }
                        foreach ($level in $levels) {
                            $file = Find-Metrics $job.Run $method $seed $level $false
                            if (-not $file) { throw "Missing matching metrics: $method / seed $seed / level $level" }
                            Copy-Item -LiteralPath $file -Destination "$bundle/metrics/${method}_seed${seed}_per${level}.json"
                        }
                    }
                }
            }
        }
        @{ Group=$Group; Seeds=$Seeds; Levels=$levels; TrainingSeed=42; Protocol='Repeated online streams from fixed source checkpoints'; ContextLR=0.0005; Epsilon=0.05; Budget=0.2 } | ConvertTo-Json -Depth 4 | Set-Content "$bundle/manifest.json" -Encoding UTF8
    } finally { Stop-Transcript | Out-Null }
    Compress-Archive -Path "$bundle/*" -DestinationPath "$bundle.zip" -Force
    Write-Host "Evidence bundle ready: $bundle.zip"
} finally { Pop-Location }
