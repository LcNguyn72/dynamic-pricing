param(
    [string]$PythonPath = "",
    [string]$OutputPath = "experiments/artifacts/comparison_v3",
    [int]$Workers = 6,
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto"
)
$ErrorActionPreference = "Stop"
$comparisonRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "../..")).Path
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $comparisonVenv = Join-Path $comparisonRoot "tmp/comparison_venv/Scripts/python.exe"
    if (Test-Path -LiteralPath $comparisonVenv) { $PythonPath = $comparisonVenv }
    else { $PythonPath = "python" }
}
if ($Workers -lt 1) { throw "Workers must be positive" }
Push-Location -LiteralPath $comparisonRoot
try {
    & $PythonPath -u -X utf8 -m experiments.comparison_v3 all --output $OutputPath --workers $Workers --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Comparison stopped with exit code $LASTEXITCODE; completed checkpoints are preserved." }
} finally {
    Pop-Location
}
