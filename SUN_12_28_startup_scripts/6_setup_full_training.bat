@echo off
REM ============================================================================
REM Step 6: Set Up Full Production Training (510 symbols)
REM CRITICAL: This sets the flag for full training
REM ============================================================================

cd ..\backend\turbomode

echo.
echo ============================================================================
echo STEP 6: SETUP FULL PRODUCTION TRAINING
echo ============================================================================
echo.
echo WARNING: This will configure training for ALL 510 S&P 500 symbols
echo This takes approximately 36 HOURS to complete!
echo.
echo You should run this OVERNIGHT (Friday 6PM -> Sunday 6AM)
echo.
echo CRITICAL STEP: Setting USE_ALL_SYMBOLS = True
echo.
pause

REM Open the file in notepad so user can verify the change
notepad regenerate_training_data.py

echo.
echo ============================================================================
echo VERIFY THE CHANGE:
echo ============================================================================
echo.
echo Please verify that Line 68 shows:
echo   USE_ALL_SYMBOLS = True
echo.
echo If you made the change and saved, press any key to continue...
echo If not, edit the file and save, then run this script again.
echo.
pause

echo.
echo ============================================================================
echo STARTING FULL PRODUCTION TRAINING
echo ============================================================================
echo.
echo This will take ~36 hours. Progress will be saved to database.
echo You can safely close this window - the process will continue.
echo.
echo Start time: %date% %time%
echo Expected completion: Sunday morning (~6:00 AM)
echo.
pause

..\..\venv\Scripts\python.exe regenerate_training_data.py

echo.
echo ============================================================================
echo TRAINING COMPLETE
echo ============================================================================
echo.
pause
