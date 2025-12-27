@echo off
echo ========================================
echo FULLY AUTOMATED ML TRADING SYSTEM
echo ========================================
echo.
echo This will run EVERYTHING automatically:
echo.
echo DAILY (at 6 PM):
echo   1. Scan market (50+ stocks)
echo   2. Simulate trades on best signals
echo   3. Check outcomes of open positions
echo   4. Mark wins/losses
echo   5. Retrain model automatically
echo.
echo NO MANUAL WORK REQUIRED!
echo.
echo ========================================
echo.
pause

cd backend\trading_system
start "Automated ML System" ..\..\venv\Scripts\python.exe automated_scheduler.py

echo.
echo ========================================
echo Automated system started!
echo Check the new window for status.
echo.
echo To stop: Close the "Automated ML System" window
echo ========================================
pause
