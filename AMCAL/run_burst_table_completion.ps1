param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('IAT','PS')]
    [string]$Dataset,
    [Parameter(Mandatory=$true)]
    [ValidateSet('AF','RAL','AMCAL')]
    [string]$Method,
    [ValidateSet('auto','cpu','cuda')]
    [string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo 'venv/Scripts/python.exe'
if (-not (Test-Path $python)) { throw 'Project venv Python not found' }
Push-Location $repo
try {
    $root = "AMCAL/ANT_Burst_${Dataset}_TEST"
    $runs = "$root/protocol_runs"
    $burst = "$root/Dataset_official_v1/AdvBurst_FTSC-$Dataset"
    if ($Method -eq 'AF') {
        $source = "$runs/structured_bank3_policy_lr0p0005_seed42"
        foreach ($seed in @(42,48,54,62,72)) {
            Write-Host "AF CHECKPOINT20 ONLINE DATASET=$Dataset SEED=$seed | reuse completed checkpoints; build missing seed-specific AF only"
            & $python AMCAL/amcal_af_clean_reference.py --dataset $Dataset --seed $seed --device $Device --source-root $source --burst-dir $burst
            if ($LASTEXITCODE -ne 0) { throw "AF failed: $Dataset seed $seed" }
        }
        & $python AMCAL/amcal_collect_af_checkpoint20.py --dataset $Dataset
        if ($LASTEXITCODE -ne 0) { throw 'AF table collection failed' }
    } elseif ($Method -eq 'RAL') {
        Write-Host "RAL FULL DATASET=$Dataset | fixed seed42 source | five evaluation seeds | missing paired endpoints only"
        & $python AMCAL/amcal_ral_full_paired_endpoints.py --dataset $Dataset
        if ($LASTEXITCODE -ne 0) { throw "RAL endpoints failed: $Dataset" }
    } else {
        Write-Host "AMCAL DATASET=$Dataset | five existing trained checkpoints | epsilon=0.05 | independent target Gain | no retraining"
        & $python AMCAL/amcal_final_multiseed_independent_gain.py --dataset $Dataset --device $Device
        if ($LASTEXITCODE -ne 0) { throw "AMCAL independent Gain failed: $Dataset" }
    }
} finally {
    Pop-Location
}
