@echo off
echo ========================================
echo Testing ML Trading System
echo ========================================
echo.

echo Step 1: Testing imports...
venv\Scripts\python.exe test_ml_system.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo Import test FAILED!
    echo Fix errors above before continuing.
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo All imports successful!
echo.
echo Now you can:
echo 1. Start Flask: start_flask.bat
echo 2. Visit: http://127.0.0.1:5000/ml-trading
echo 3. Click "Run Scan"
echo ========================================
pause
