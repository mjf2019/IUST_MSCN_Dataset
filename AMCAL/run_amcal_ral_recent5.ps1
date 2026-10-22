param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [int]$Seed = 42
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $root = "AMCAL/ANT_Burst_${Dataset}_TEST"
    $sourceName = if ($Dataset -eq 'IAT') { 'ral_paper_rf10_prefix_levels1_3_seed42' } else { 'baselines_levels1_3_rf10_seed42' }
    $source = "$root/protocol_runs/$sourceName/Models"
    $output = "$root/protocol_runs/ral_recent5_seed$Seed"
    $models = Join-Path $output 'Models'
    foreach ($name in @('ral.joblib','protocol_preprocessing.joblib')) {
        $src = Join-Path $source $name
        if (-not (Test-Path $src)) { throw "Missing source checkpoint: $src" }
    }
    New-Item -ItemType Directory -Path $models -Force | Out-Null
    foreach ($name in @('ral.joblib','protocol_preprocessing.joblib')) {
        $src = Join-Path $source $name
        $dst = Join-Path $models $name
        if (Test-Path $dst) {
            if ((Get-FileHash $src).Hash -ne (Get-FileHash $dst).Hash) { throw "Checkpoint mismatch: $dst" }
        } else { Copy-Item -LiteralPath $src -Destination $dst }
    }
    & python AMCAL/amcal_burst_baselines.py evaluate --methods ral --output $output --burst-dir "$root/Dataset_official_v1/AdvBurst_FTSC-$Dataset" --levels 5 7 10 12 15 17 20 --seed $Seed --budget-fraction 0.2 --ral-budget-mode prefix --ral-update-window 5 --device cpu
    if ($LASTEXITCODE -ne 0) { throw "RAL recent5 failed: $Dataset" }
} finally { Pop-Location }
