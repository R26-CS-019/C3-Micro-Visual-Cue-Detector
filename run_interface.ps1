param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$componentPython = 'C:\Users\oshan\Desktop\MicroVisualPhishingDetector\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $componentPython)) { throw "Python environment not found: $componentPython" }
& $componentPython -u -B (Join-Path $PSScriptRoot 'interface\server.py') --project $PSScriptRoot --port $Port
exit $LASTEXITCODE
