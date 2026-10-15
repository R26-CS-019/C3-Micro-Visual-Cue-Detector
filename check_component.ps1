param()
$ErrorActionPreference = 'Stop'
$taskPython = 'C:\Users\oshan\Desktop\MicroVisualPhishingDetector\venv\Scripts\python.exe'
$taskEntry = Join-Path $PSScriptRoot 'Micro Visual Cue Component\src\visual_component.py'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'The TensorFlow virtual environment was not found.' }
if (-not (Test-Path -LiteralPath $taskEntry)) { throw 'The visual component entry point was not found.' }
Write-Host 'Starting the full visual research component: EfficientNetV2B0 Models 1 and 2.'
& $taskPython -u -B $taskEntry
if ($LASTEXITCODE -ne 0) { throw 'Component startup or URL analysis failed; review the error above.' }
