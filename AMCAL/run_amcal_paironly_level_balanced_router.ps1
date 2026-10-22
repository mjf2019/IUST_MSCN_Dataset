param(
    [ValidateRange(0.0,1000000.0)][double]$ProtectedWeight = 0.4,
    [ValidateRange(0.0,1000000.0)][Nullable[double]]$NormalWeight = $null,
    [ValidateSet('Both','Independent','Sequential')][string]$EvaluationSet = 'Both',
    [int]$Seed = 42,
    [string]$EvaluationTag = '',
    [string]$OutputRun = 'AMCAL/ANT_Burst_IAT_TEST/protocol_runs/final_enhancement_query_credit_paironly_trainseed42',
    [ValidateSet('cpu','cuda','auto')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$burst = 'AMCAL/ANT_Burst_IAT_TEST/Dataset_official_v1/AdvBurst_FTSC-IAT'
function Invoke-AMCAL([string[]]$Arguments) {
    & python 'AMCAL/amcal_final_enhancement.py' @Arguments
    if ($LASTEXITCODE -ne 0) { throw "AMCAL paired-only level-balanced routing replay failed with exit code $LASTEXITCODE" }
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path "$OutputRun/Models/final_enhancement.pth" -PathType Leaf)) {
        throw "PairOnly checkpoint not found under $OutputRun/Models."
    }
    $common = @('evaluate','--output',$OutputRun,'--burst-dir',$burst,'--context-count','1',
        '--query-schedule','policy','--update-policy','learned','--modes','learned','--router-protected-replay','--router-level-balanced-history','--routing-oracle-audit','--skip-existing-completed',
        '--router-protected-weight',$ProtectedWeight.ToString('G',[System.Globalization.CultureInfo]::InvariantCulture),
        '--seed',"$Seed",'--device',$Device,'--eval-epsilon','0.05',
        '--budget-fraction','0.2','--context-lr','0.0005')
    if ($EvaluationTag) { $common += @('--evaluation-tag',$EvaluationTag) }
    if ($PSBoundParameters.ContainsKey('NormalWeight')) {
        $common += @('--router-normal-weight',([double]$NormalWeight).ToString('G',[System.Globalization.CultureInfo]::InvariantCulture))
    }
    if ($EvaluationSet -in @('Both','Independent')) {
        Invoke-AMCAL -Arguments $common
    }
    if ($EvaluationSet -in @('Both','Sequential')) {
        Invoke-AMCAL -Arguments ($common + @('--continual'))
    }
} finally { Pop-Location }
