param(
    [Parameter(Mandatory=$true)][ValidateSet('IAT','PS')][string]$Dataset,
    [double]$Beta = -1,
    [ValidateSet('All','Train','Evaluate')][string]$Action = 'All',
    [int]$Seed = 42,
    [int]$TrainRows = 300,
    [int]$ValRows = 300,
    [int]$TestRows = 300,
    [int]$CNNEpochs = 5,
    [double]$CNNLR = 0.0001,
    [int]$Patience = 2,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'cuda',
    [string]$SelectorRun = '',
    [string]$ActorRun = '',
    [string]$Teacher = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    if ($Beta -eq -1) { $Beta = if ($Dataset -eq 'IAT') { 26 } else { 55 } }
    $python = Join-Path (Get-Location) 'venv/Scripts/python.exe'
    if (-not (Test-Path $python)) { $python = 'python' }
    & $python -m unittest discover -s AMCAL/tests -p test_post_selector_cnn13.py -v
    if ($LASTEXITCODE -ne 0) { throw 'CNN-only fine-tuning contract tests failed' }
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $arguments = @('AMCAL/amcal_post_selector_cnn13.py','--dataset',$Dataset,'--beta',$Beta.ToString($culture),
        '--action',$Action,'--seed',"$Seed",'--train-rows',"$TrainRows",'--val-rows',"$ValRows",
        '--test-rows',"$TestRows",'--cnn-epochs',"$CNNEpochs",'--cnn-lr',$CNNLR.ToString($culture),
        '--patience',"$Patience",'--device',$Device)
    if ($SelectorRun) { $arguments += @('--selector-run',$SelectorRun) }
    if ($ActorRun) { $arguments += @('--actor-run',$ActorRun) }
    if ($Teacher) { $arguments += @('--teacher',$Teacher) }
    if ($Output) { $arguments += @('--output',$Output) }
    Write-Host "DATASET=$Dataset BETA=$Beta | Current versus CNN13 | CNN-only additional TRAIN 1/3 | LABEL_BUDGET=20%"
    Write-Host 'Initial Context and DDQN stay identical. Both bundles are saved. TEST resets adaptation per variant/level.'
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Post-selector CNN fine-tuning failed: $Dataset / $Action" }
} finally { Pop-Location }
