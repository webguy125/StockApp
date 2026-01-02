@echo off
REM ============================================================================
REM Step 5: Analyze Test Scan Results
REM Verify the fix worked - check BUY/SELL distribution and confidence spread
REM ============================================================================

cd ..

echo.
echo ============================================================================
echo STEP 5: ANALYZE SCAN RESULTS
echo ============================================================================
echo.

venv\Scripts\python.exe analyze_signals.py

echo.
echo ============================================================================
echo KEY SUCCESS CRITERIA:
echo ============================================================================
echo.
echo [OK] FIX WORKED IF:
echo   - SELL signals exist (10-40%% of total)
echo   - Confidence spread greater than 5%%
echo   - BUY: 30-70%% (not 100%%)
echo.
echo [FAIL] FIX FAILED IF:
echo   - Still 100%% BUY or 100%% SELL
echo   - Confidence clustered (less than 1%% spread)
echo.
echo Next step: If fix worked, run 6_setup_full_training.bat
echo.
pause
