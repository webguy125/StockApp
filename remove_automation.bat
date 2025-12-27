@echo off
echo ========================================
echo REMOVE AUTOMATION
echo ========================================
echo.
echo This will STOP the automatic ML system.
echo.
pause

schtasks /delete /tn "ML Trading System - Daily Learning" /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Automation removed successfully!
    echo.
    echo The ML system will no longer run automatically.
    echo You can still run it manually from the website.
    echo.
) else (
    echo.
    echo ⚠️ Task not found or already removed.
    echo.
)

pause
