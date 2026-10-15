param([Parameter(ValueFromRemainingArguments=$true)][string[]]$ResearchArguments)
$ErrorActionPreference = 'Stop'
$pythonPath = if ($env:PP2_PYTHON) { $env:PP2_PYTHON } else { 'C:\Users\oshan\Desktop\MicroVisualPhishingDetector\venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Set PP2_PYTHON to the Python executable in your TensorFlow environment.' }
& $pythonPath -u -B (Join-Path $PSScriptRoot 'research.py') @ResearchArguments
exit $LASTEXITCODE
