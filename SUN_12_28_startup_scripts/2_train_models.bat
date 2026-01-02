@echo off
REM ============================================================================
REM Step 2: Train All 9 TurboMode Models
REM Takes ~15 minutes
REM ============================================================================

cd ..\backend\turbomode

echo.
echo ============================================================================
echo STEP 2: TRAIN TURBOMODE MODELS
echo ============================================================================
echo.
echo This will take approximately 15 minutes...
echo.
echo Training 9 models:
echo   1. Random Forest
echo   2. XGBoost
echo   3. LightGBM
echo   4. Extra Trees
echo   5. Gradient Boost
echo   6. Neural Network
echo   7. Logistic Regression
echo   8. SVM
echo   9. Meta-Learner (Ensemble)
echo.
pause

..\..\venv\Scripts\python.exe train_turbomode_models.py

echo.
echo ============================================================================
echo TRAINING COMPLETE
echo ============================================================================
echo.
echo Next step: Run 3_clear_bad_signals.bat
echo.
pause
