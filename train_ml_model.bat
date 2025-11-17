@echo off
echo ============================================
echo Triad Trend Pulse ML Model Training
echo ============================================
echo.
echo This will train the ML model for pivot detection.
echo Expected time: 10-30 minutes
echo.
echo Assets: BTC-USD, ETH-USD, SOL-USD, AAPL, TSLA, NVDA, SPY, QQQ
echo Timeframes: 5min, 15min, 1h, 4h, 1d
echo.
pause

cd backend\ml
..\..\venv\Scripts\python.exe train_pivot_model.py

echo.
echo ============================================
echo Training Complete!
echo Model saved to: backend\ml\models\pivot_model.pth
echo ============================================
echo.
echo Press any key to close...
pause
