@echo off
echo ============================================================
echo FULL ML TRAINING - OVERNIGHT RUN
echo ============================================================
echo.
echo This will:
echo   - Process 82 symbols (6-9 hours)
echo   - Train 9 models (1-2 hours)
echo   - Total: 8-11 hours (perfect for overnight)
echo.
echo You will see progress in this window.
echo DO NOT CLOSE THIS WINDOW while training!
echo.
echo Press Ctrl+C if you need to stop (checkpoint saves progress)
echo ============================================================
echo.
pause

cd /d C:\StockApp\backend
call ..\venv\Scripts\activate.bat

echo.
echo Starting full training...
echo.

python ..\run_training_with_checkpoints.py

echo.
echo ============================================================
echo TRAINING COMPLETE!
echo ============================================================
echo.
echo Check the results above for test accuracy.
echo Goal: Test accuracy >= 90%%
echo.
pause
