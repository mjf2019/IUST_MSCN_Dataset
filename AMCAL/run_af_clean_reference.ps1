param(
    [ValidateSet('All','SDNCampus','ISCX-TOR','IUST','IoT','IAT','PS')]
    [string]$Dataset = 'All',
    [int]$Seed = 42,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$SourceRoot = '',
    [string]$BurstDir = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = Join-Path $repo 'venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
if ($Dataset -eq 'All' -and ($SourceRoot -or $BurstDir -or $Output)) {
    throw 'Explicit paths require a single dataset.'
}
if ($Dataset -in @('IAT','PS') -and (-not $SourceRoot -or -not $BurstDir)) {
    throw 'Burst requires SourceRoot and BurstDir.'
}
$datasets = @($Dataset)
if ($Dataset -eq 'All') { $datasets = @('SDNCampus','ISCX-TOR','IUST','IoT') }
Push-Location $repo
try {
    foreach ($name in $datasets) {
        $arguments = @('AMCAL/amcal_af_clean_reference.py','--dataset',$name,'--seed',"$Seed",'--device',$Device)
        if ($SourceRoot) { $arguments += @('--source-root',$SourceRoot) }
        if ($BurstDir) { $arguments += @('--burst-dir',$BurstDir) }
        if ($Output) { $arguments += @('--output',$Output) }
        Write-Host "AF CHECKPOINT: $name | clean source25/class + TRAIN20% | reset clean model for every target | calibration20%"
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "AF clean reference failed: $name" }
    }
} finally { Pop-Location }
