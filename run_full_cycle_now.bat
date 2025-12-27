@echo off
echo ========================================
echo RUN FULL AUTOMATED CYCLE NOW
echo ========================================
echo.
echo This will run IMMEDIATELY (not wait for scheduled time):
echo   1. Scan market
echo   2. Simulate trades
echo   3. Check outcomes
echo   4. Retrain if ready
echo.
echo ========================================
echo.

cd backend\trading_system
..\..\venv\Scripts\python.exe automated_scheduler.py --now

echo.
echo ========================================
echo Full cycle complete!
echo ========================================
pause
