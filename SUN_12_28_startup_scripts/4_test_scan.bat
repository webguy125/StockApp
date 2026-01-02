@echo off
REM ============================================================================
REM Step 4: Run Test Scan with Fixed Models
REM Scans S&P 500 to verify BUY/SELL balance and confidence spread
REM ============================================================================

cd ..\backend\turbomode

echo.
echo ============================================================================
echo STEP 4: RUN TEST SCAN
echo ============================================================================
echo.
echo Running overnight_scanner.py with FIXED models...
echo This will scan all S&P 500 stocks and generate signals.
echo.
echo Expected results:
echo   - Mix of BUY and SELL signals (not 100%% BUY)
echo   - Varied confidence scores (60-99%%, not all 99%%)
echo   - Reasonable distribution by sector/market cap
echo.
pause

python overnight_scanner.py

echo.
echo ============================================================================
echo SCAN COMPLETE
echo ============================================================================
echo.
echo Next step: Run 5_analyze_results.bat to verify fix worked
echo.
pause
