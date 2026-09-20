# PowerShell launcher for Singapore AI Travel Planning Assistant
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $scriptDir "..\..\..\.venv\Scripts\python.exe"

Write-Host "Launching Singapore AI Travel Planning Assistant Web UI..." -ForegroundColor Cyan
& $venvPython -m streamlit run (Join-Path $scriptDir "app.py")
