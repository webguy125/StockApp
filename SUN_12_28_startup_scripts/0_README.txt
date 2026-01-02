================================================================================
TURBOMODE FIX - STARTUP SCRIPTS
Saturday December 28, 2025 - After 6 PM Restart
================================================================================

OVERVIEW:
These scripts automate the post-restart workflow to complete the TurboMode fix.
Run them in numerical order (1 -> 2 -> 3 -> 4 -> 5 -> 6).

================================================================================
WORKFLOW:
================================================================================

1. 1_verify_backtest.bat (~1 minute)
   - Checks if backtest completed successfully
   - Verifies labels are 'buy'/'hold'/'sell' (not 'win'/'loss')
   - Shows database statistics

2. 2_train_models.bat (~15 minutes)
   - Trains all 9 TurboMode models on corrected data
   - Random Forest, XGBoost, LightGBM, etc.
   - Creates meta-learner ensemble

3. 3_clear_bad_signals.bat (~5 seconds)
   - Deletes the 200 incorrect signals from turbomode.db
   - Removes all-BUY signals with 99% clustered confidence

4. 4_test_scan.bat (~5 minutes)
   - Runs full S&P 500 scan with FIXED models
   - Generates new signals with correct BUY/SELL balance

5. 5_analyze_results.bat (~1 minute)
   - Analyzes signal distribution
   - Verifies fix worked (SELL signals exist, confidence varies)
   - Shows detailed statistics

6. 6_setup_full_training.bat (~36 hours)
   - Sets USE_ALL_SYMBOLS = True
   - Runs full 510-symbol production training
   - Start Friday 6 PM, completes Sunday 6 AM

================================================================================
TOTAL TIME:
================================================================================

Steps 1-5: ~25 minutes
Step 6: ~36 hours (run overnight Friday -> Sunday)

================================================================================
SUCCESS CRITERIA:
================================================================================

After Step 5, you should see:
- SELL signals: 10-40% (not 0%)
- Confidence spread: > 5% range
- BUY signals: 30-70% (not 100%)

If these criteria are met, the fix WORKED!
Proceed to Step 6 for full production training.

================================================================================
IMPORTANT NOTES:
================================================================================

- Run scripts from SUN_12_28_startup_scripts directory
- Each script will pause for confirmation
- Scripts are numbered in execution order
- Read output carefully at each step
- If any step fails, STOP and investigate

================================================================================
FULL DOCUMENTATION:
================================================================================

See: C:\StockApp\TURBOMODE_FIX_SUMMARY_2025-12-28.md
For: Complete technical details, root cause analysis, and troubleshooting

================================================================================
QUICK REFERENCE:
================================================================================

Database: C:\StockApp\backend\backend\data\advanced_ml_system.db
Models: C:\StockApp\backend\data\turbomode_models\
Signals: C:\StockApp\backend\data\turbomode.db

================================================================================
END OF README
================================================================================
