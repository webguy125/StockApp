@echo off
echo ========================================
echo ML Trading System - Quick Start
echo ========================================
echo.

cd backend\trading_system
..\..\venv\Scripts\python.exe run_scan.py

echo.
echo ========================================
echo Scan complete!
echo View results at: http://127.0.0.1:5000/ml-trading
echo ========================================
pause
