@echo off
echo ========================================
echo CHECKPOINT-ENABLED TRAINING
echo ========================================
echo.
echo This training script can be stopped and resumed at any time!
echo Progress is saved after each symbol and each model.
echo.
echo Press Ctrl+C to stop at any time.
echo Run this script again to resume from last checkpoint.
echo.
pause

python run_training_with_checkpoints.py

echo.
echo ========================================
echo Training session ended
echo ========================================
pause
