@echo off
echo ========================================
echo  Scanner Scheduler (Nightly at Midnight)
echo ========================================
echo.
echo This will keep running and execute the
echo scanner every night at midnight UTC.
echo.
echo Press Ctrl+C to stop the scheduler
echo.
echo ========================================
echo.

cd /d "%~dp0"
cd agents

call ..\venv\Scripts\activate.bat

python schedule_scanner.py

pause
