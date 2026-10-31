param(
    [string]$Server = '.\SQLEXPRESS'
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Push-Location $project
try {
    $env:PYTHONIOENCODING = 'utf-8'
    python '3_select_products_2y.py'
    if ($LASTEXITCODE -ne 0) { throw 'Product selection failed.' }

    sqlcmd -S $Server -E -b -i (Join-Path $PSScriptRoot 'schema.sql')
    if ($LASTEXITCODE -ne 0) { throw 'Database schema creation failed.' }

    $env:PRICING_SQL_SERVER = $Server
    python -m database.import_data
    if ($LASTEXITCODE -ne 0) { throw 'Data import failed.' }
} finally {
    Pop-Location
}
