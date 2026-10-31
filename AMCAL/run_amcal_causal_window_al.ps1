param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [double]$Beta = -1,
    [int]$WindowSize = 5,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda'
)
$ErrorActionPreference='Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $python=Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python='python' }
    if ($Beta -eq -1) { $Beta=if ($Dataset -eq 'IAT') {26} else {55} }
    & $python -m unittest discover -s AMCAL/tests -p test_causal_window_query.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Causal-window tests failed' }
    $culture=[Globalization.CultureInfo]::InvariantCulture
    Write-Host "DATASET=$Dataset BETA=$Beta WINDOW=$WindowSize | all eligible TEST samples, all seven levels, budget20%"
    Write-Host 'Four methods: DDQNOriginal, UncertaintyWindow, RandomWindow, DDQNWindowRank'
    Write-Host 'No training. Window methods predict first and release feedback only at window close.'
    & $python AMCAL/amcal_causal_window_al.py --dataset $Dataset --beta $Beta.ToString($culture) --window-size $WindowSize --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Causal-window evaluation failed: $Dataset" }
} finally { Pop-Location }

