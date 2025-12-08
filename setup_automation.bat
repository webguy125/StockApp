@echo off
echo ========================================
echo  StockApp Scanner Automation Setup
echo ========================================
echo.
echo This will set up the scanner to run
echo automatically every night at midnight.
echo.
echo Checking for Administrator privileges...
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running as Administrator
    echo.
) else (
    echo [ERROR] Not running as Administrator!
    echo.
    echo Please right-click this file and select
    echo "Run as administrator"
    echo.
    pause
    exit /b 1
)

REM Run PowerShell script
echo Running PowerShell setup script...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0setup_automation.ps1"

exit /b 0
