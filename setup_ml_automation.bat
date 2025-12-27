@echo off
echo ========================================
echo SETUP FULLY AUTOMATED ML SYSTEM
echo ========================================
echo.
echo This will configure Windows to run the ML system automatically:
echo - Every day at 6:00 PM
echo - Runs in background (no windows)
echo - No manual work required!
echo.
echo ========================================
pause

echo.
echo Creating scheduled task...
echo.

schtasks /create /tn "ML Trading System - Daily Learning" /tr "%CD%\venv\Scripts\python.exe %CD%\backend\trading_system\automated_scheduler.py --now" /sc daily /st 18:00 /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo ✅ SUCCESS!
    echo ========================================
    echo.
    echo The ML system is now set to run automatically:
    echo - Time: 6:00 PM daily
    echo - What it does:
    echo   1. Scans market
    echo   2. Simulates trades
    echo   3. Checks outcomes
    echo   4. Learns and retrains
    echo.
    echo NO MORE CLICKING NEEDED!
    echo.
    echo To view/modify:
    echo - Open Task Scheduler
    echo - Find: "ML Trading System - Daily Learning"
    echo.
    echo To run manually:
    echo - Go to: http://127.0.0.1:5000/ml-trading
    echo - Click "Run Full Cycle" button
    echo.
    echo ========================================
) else (
    echo.
    echo ========================================
    echo ❌ ERROR
    echo ========================================
    echo.
    echo Could not create scheduled task.
    echo Try running this as Administrator.
    echo.
    echo Right-click setup_ml_automation.bat
    echo Select "Run as administrator"
    echo.
    echo ========================================
)

pause
