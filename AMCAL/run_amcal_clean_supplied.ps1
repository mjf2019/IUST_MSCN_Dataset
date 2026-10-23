param(
    [ValidateSet('All','SDNCampus','ISCX-TOR','IUST','IoT')][string]$Dataset = 'All',
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [ValidateSet('AMCAL','AF','DFE')][string[]]$Methods = @('AMCAL','AF','DFE'),
    [int]$Seed = 42,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [switch]$AssumeHistoricalPoolOrder
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $datasets = if ($Dataset -eq 'All') { @('SDNCampus','ISCX-TOR','IUST','IoT') } else { @($Dataset) }
    foreach ($name in $datasets) {
        $arguments = @('--dataset',$name,'--action',$Action,'--seed',"$Seed",'--device',$Device,'--methods') + $Methods
        if ($AssumeHistoricalPoolOrder) { $arguments += '--assume-historical-pool-order' }
        & python AMCAL/amcal_clean_supplied.py @arguments
        if ($LASTEXITCODE -ne 0) { throw "Clean-only supplied-attack pipeline failed: $name / $Action" }
    }
} finally { Pop-Location }
