@echo off
echo Stopping Flask server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Flask*" 2>nul
timeout /t 2 /nobreak >nul

echo Starting Flask server...
cd backend
start "Flask Server" ..\venv\Scripts\python.exe api_server.py

echo.
echo Flask server restarted!
echo Open: http://127.0.0.1:5000/ml-trading
pause
