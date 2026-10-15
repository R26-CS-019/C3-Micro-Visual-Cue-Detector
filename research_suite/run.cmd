@echo off
setlocal
if defined PP2_PYTHON (
  set "PP2_RESEARCH_PYTHON=%PP2_PYTHON%"
) else (
  set "PP2_RESEARCH_PYTHON=%~dp0..\.venv\Scripts\python.exe"
)
if not exist "%PP2_RESEARCH_PYTHON%" (
  echo Create the project .venv or set PP2_PYTHON to your TensorFlow Python executable.
  exit /b 1
)
"%PP2_RESEARCH_PYTHON%" -u -B "%~dp0research.py" %*
exit /b %ERRORLEVEL%
