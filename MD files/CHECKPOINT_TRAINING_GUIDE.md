# Checkpoint Training System Guide

## Overview
Your training pipeline now has **automatic checkpoint saves** - you can restart Windows anytime without losing progress!

## Current Status
- **Started**: 2025-12-24 12:46 PM
- **Processing**: Symbol 1/82 (AAPL) in progress
- **Estimated Time**: 6.8-9.6 hours total
- **Progress Saved**: After each symbol completes

## How It Works

### Automatic Checkpoints
- ✅ **After each symbol**: Progress saved to `backend/data/checkpoints/training_checkpoint.json`
- ✅ **After each model**: Model training progress saved
- ✅ **Database persistence**: All feature data saved to `advanced_ml_system.db`
- ✅ **Resume capability**: Automatically skips completed symbols/models

### What's Saved
1. **Backtest Progress**
   - List of completed symbols
   - List of failed symbols (if any)
   - Total training samples collected

2. **Training Progress**
   - Which base models have been trained (0-8)
   - Whether meta-learner is trained
   - Model files saved to disk

3. **Evaluation Results**
   - Model performance metrics
   - Regime-specific results

## Commands You Can Use

### Check Progress Anytime
```bash
python check_checkpoint.py
```
Shows:
- Current phase
- Symbols completed/failed/remaining
- Models trained
- Total samples collected

### Resume After Restart
```bash
python run_training_with_checkpoints.py
```
OR
```bash
run_checkpoint_training.bat
```
Automatically:
- Loads last checkpoint
- Skips completed work
- Continues from where it stopped

### Start Fresh (if needed)
```bash
python reset_checkpoint.py
```
- Backs up current checkpoint
- Resets progress to zero
- Keeps database data (won't re-download data)

## Safe to Stop Anytime!

### You Can:
- ✅ Press **Ctrl+C** to stop training
- ✅ **Close terminal** - progress is saved
- ✅ **Restart Windows** for other programs
- ✅ Run `run_training_with_checkpoints.py` again to resume

### What Happens on Restart:
1. Script loads checkpoint file
2. Checks database for completed symbols
3. **Skips all completed work**
4. Continues from next pending symbol/model

## Progress Tracking

### Phase 1: Backtest (6-9 hours)
- Processes 82 symbols one at a time
- Checkpoint after EACH symbol
- If you stop at symbol 40, next run starts at symbol 41

### Phase 2: Load Data (< 1 minute)
- Loads all samples from database
- No checkpoint needed (very fast)

### Phase 3: Train Base Models (1-2 hours)
- Trains 8 models one at a time
- Checkpoint after EACH model
- If you stop after model 5, next run starts at model 6

### Phase 4: Train Meta-Learner (5-10 minutes)
- Single checkpoint after completion

### Phase 5-6: Evaluation (2-5 minutes)
- Single checkpoint with all results

## Current Training Run

**Status**: Running in background
**Process ID**: 777615
**Log Location**: Console output

**Estimated Timeline**:
- Symbol processing: ~5-7 minutes per symbol
- 82 symbols × 6 minutes = ~8.2 hours
- Model training: +1.5 hours
- **Total**: ~10 hours

## Example Scenarios

### Scenario 1: Need to Restart Windows After 3 Hours
- **Progress**: ~30 symbols completed (3 hours × 10 symbols/hour)
- **Action**: Press Ctrl+C, restart Windows, install other program
- **Resume**: Run `python run_training_with_checkpoints.py`
- **Result**: Starts at symbol 31, saves ~3 hours of work!

### Scenario 2: Training Crashes at Symbol 60
- **Progress**: 59 symbols complete, 23 remaining
- **Action**: Run `python run_training_with_checkpoints.py`
- **Result**: Starts at symbol 60, no data loss

### Scenario 3: Want to Check Progress
- **Action**: Run `python check_checkpoint.py` in new terminal
- **Result**: See current status without stopping training

## Files Created

### Core Scripts
- `run_training_with_checkpoints.py` - Main training script with checkpoints
- `backend/advanced_ml/training/checkpoint_manager.py` - Checkpoint logic

### Helper Scripts
- `check_checkpoint.py` - View current status
- `reset_checkpoint.py` - Start fresh
- `run_checkpoint_training.bat` - Windows launcher

### Data Files
- `backend/data/checkpoints/training_checkpoint.json` - Progress state
- `backend/data/advanced_ml_system.db` - All training data
- `backend/data/training_results_checkpoint.json` - Final results

## Tips

1. **Monitor Progress**: Run `check_checkpoint.py` periodically to see how many symbols are done

2. **Don't Worry About Interruptions**: Every completed symbol is saved - you can restart anytime

3. **Database is Safe**: Even if checkpoint file gets corrupted, all feature data is in the database

4. **Multiple Restarts OK**: You can start/stop as many times as needed

5. **Background Running**: Leave it running overnight, check in the morning!

## What's Running Now

Current training is processing all 82 symbols with 202 features each:
- **Technology**: 9 symbols (AAPL, MSFT, NVDA, GOOGL, META, PLTR, SNOW, CRWD, SMCI)
- **Financials**: 8 symbols (JPM, BAC, WFC, C, GS, MS, BLK, SCHW)
- **Healthcare**: 8 symbols
- **Consumer Discretionary**: 8 symbols
- **Communication Services**: 7 symbols
- **Industrials**: 8 symbols
- **Consumer Staples**: 7 symbols
- **Energy**: 7 symbols
- **Materials**: 7 symbols
- **Real Estate**: 7 symbols
- **Utilities**: 6 symbols

Each symbol generates 500-800 training samples with:
- **179 technical features**: Price, volume, indicators, patterns
- **23 event features**: SEC filings, news, sentiment, volatility

**Total Expected**: ~40,000-65,000 training samples

## Next Steps After Training Completes

1. Review results: `backend/data/training_results_checkpoint.json`
2. Run SHAP analysis: `python backend/advanced_ml/analysis/shap_analyzer.py`
3. Validate models: Promotion gate checks
4. Deploy to production

---

**You're all set!** Restart Windows whenever you need - your progress is safe! 🚀
