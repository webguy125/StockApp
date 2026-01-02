@echo off
REM ============================================================================
REM Step 1: Verify Backtest Completed Successfully
REM Run this FIRST after restart
REM ============================================================================

cd ..

echo.
echo ============================================================================
echo STEP 1: VERIFY BACKTEST COMPLETION
echo ============================================================================
echo.

REM Check if database exists and has data
echo [CHECK] Database file exists...
if exist "backend\backend\data\advanced_ml_system.db" (
    echo [OK] Database found
    dir "backend\backend\data\advanced_ml_system.db" | findstr "advanced_ml_system.db"
) else (
    echo [ERROR] Database not found!
    pause
    exit /b 1
)

echo.
echo [CHECK] Verifying label distribution...
venv\Scripts\python.exe -c "import sqlite3; conn = sqlite3.connect('backend/backend/data/advanced_ml_system.db'); cursor = conn.cursor(); cursor.execute('SELECT outcome, COUNT(*) FROM trades WHERE trade_type=\"backtest\" GROUP BY outcome'); rows = cursor.fetchall(); print('\nLabel distribution:'); [print(f'  {row[0]}: {row[1]}') for row in rows]; conn.close(); print('\n'); buy = [r[1] for r in rows if r[0]=='buy']; sell = [r[1] for r in rows if r[0]=='sell']; print('[OK] Fix worked!' if buy and sell else '[ERROR] Fix failed - no buy/sell labels!')"

echo.
echo ============================================================================
echo VERIFICATION COMPLETE
echo ============================================================================
echo.
echo If you see 'buy', 'hold', 'sell' labels above, the fix worked!
echo.
echo Next step: Run 2_train_models.bat
echo.
pause
