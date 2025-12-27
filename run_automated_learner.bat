@echo off
echo ========================================
echo Automated ML Learning System
echo ========================================
echo.
echo This runs ONE learning cycle:
echo 1. Checks outcomes of open positions
echo 2. Simulates new trades from latest signals
echo 3. Retrains model if enough data collected
echo.
echo ========================================
echo.

cd backend\trading_system
..\..\venv\Scripts\python.exe automated_learner.py

echo.
echo ========================================
echo Cycle complete!
echo View results at: http://127.0.0.1:5000/ml-trading
echo ========================================
pause
