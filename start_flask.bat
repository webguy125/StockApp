@echo off
cd /d C:\StockApp

echo Checking for running Flask servers...
powershell -Command "Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { Write-Host 'Stopping Python process:' $_.Id; Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak >nul

echo Starting Flask server...
venv\Scripts\python.exe backend\api_server.py
pause