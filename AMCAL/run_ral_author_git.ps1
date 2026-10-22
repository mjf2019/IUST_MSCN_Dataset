param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [int]$Level = 0,
    [int]$Seed = 42,
    [ValidateRange(0.0,1.0)][double]$BudgetFraction = 0.2,
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $runArgs = @('--dataset',$Dataset,'--level',"$Level",'--seed',"$Seed",'--budget-fraction',$BudgetFraction.ToString('0.##########',[Globalization.CultureInfo]::InvariantCulture))
    if ($Output) { $runArgs += @('--output',$Output) }
    & python AMCAL/amcal_ral_author_protocol.py @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Author RAL run failed: $Dataset level $Level" }
} finally { Pop-Location }
