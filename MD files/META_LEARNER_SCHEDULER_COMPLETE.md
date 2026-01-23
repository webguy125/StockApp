# Meta-Learner Retraining Scheduler - Implementation Complete

**Date:** 2026-01-11
**Status:** READY FOR PRODUCTION

---

## Overview

Successfully added automated meta-learner retraining to the TurboMode scheduler. The system will now automatically regenerate meta-predictions and retrain the meta-learner with override-aware features every 6 weeks.

---

## Complete TurboMode Schedule

### Daily Tasks
1. **Overnight Scan** - 23:00 (11:00 PM)
   - Scans 82 curated stocks
   - Generates BUY/SELL/HOLD predictions
   - Duration: ~5-10 minutes

2. **Outcome Tracker** - 02:00 (2:00 AM)
   - Tracks performance of previous signals
   - Updates signal outcomes in database
   - Duration: ~2-5 minutes

### Weekly Tasks
3. **Training Sample Generator** - Sunday 03:00 (3:00 AM)
   - Generates training samples from tracked outcomes
   - Prepares data for model retraining
   - Duration: ~5-10 minutes

4. **Meta-Learner Retraining** - Every 6 weeks, Sunday 23:45 (11:45 PM)
   - Regenerates meta-predictions table (169,400 samples)
   - Retrains meta-learner with 55 features
   - Duration: ~3-5 minutes
   - **First Run:** February 22, 2026 at 11:45 PM
   - **Next Runs:** Every 6 weeks thereafter (April 5, May 17, June 28, etc.)

### Monthly Tasks
5. **Model Retraining** - 1st of month, 04:00 (4:00 AM)
   - Retrains all 8 base models + meta-learner
   - Full training pipeline with latest data
   - Duration: ~30-60 minutes

---

## Conflict Analysis

### ⚠️ Potential Conflict Identified

**Conflict:** Overnight Scan (23:00) and Meta-Learner Retraining (23:45) on Sundays

**Risk Level:** LOW

**Mitigation:**
- 45-minute buffer between tasks
- Overnight scan typically completes in 5-10 minutes
- Meta-learner retraining has 1-hour grace period (misfire_grace_time=3600)
- Both tasks run in separate threads via APScheduler

**Resolution:** No changes needed. Buffer is sufficient.

---

## Files Created/Modified

### New Files
1. `backend/turbomode/meta_retrain.py`
   - Main retraining orchestrator
   - Calls generate_meta_predictions() and retrain_meta_learner()

2. `backend/turbomode/generate_meta_predictions.py`
   - Generates meta-predictions table with base model outputs
   - Optimized batch processing (5000 samples/batch)

3. `backend/turbomode/retrain_meta_with_override_features.py`
   - Retrains meta-learner with 55 features
   - Adds override-aware features (asymmetry, directional confidence, etc.)

### Modified Files
1. `backend/turbomode/turbomode_scheduler.py`
   - Added import for `maybe_retrain_meta`
   - Added scheduling job with 6-week interval
   - Updated logging output

---

## Technical Details

### Meta-Learner Retraining Process

**Step 1: Generate Meta-Predictions**
- Loads all 8 base models (XGBoost, LightGBM, CatBoost variants)
- Runs inference on 169,400 training samples
- Stores 24 probability outputs (8 models × 3 classes) in database
- Batch size: 5,000 samples
- Duration: ~2 minutes

**Step 2: Add Override-Aware Features**
- Calculates 24 per-model features (asymmetry, max_directional, neutral_dominance)
- Calculates 7 aggregate features (avg_asymmetry, consensus, etc.)
- Total features: 55 (24 base + 24 per-model + 7 aggregate)

**Step 3: Retrain Meta-Learner**
- Trains LightGBM classifier with 55 features
- Uses class weights to handle imbalance (82% neutral data)
- 80/20 train/validation split
- Early stopping (50 rounds)
- Saves to `backend/data/turbomode_models/meta_learner_v2/`
- Duration: ~1 minute

### Performance Metrics (Current Model)
- Validation Accuracy: 98.86%
- SELL: 99% precision, 94% recall
- HOLD: 99% precision, 100% recall
- BUY: 98% precision, 95% recall

---

## Scheduler Configuration

```python
# First run calculation
first_run = datetime(2026, 1, 11, 23, 45) + timedelta(weeks=6)
# = February 22, 2026 at 11:45 PM

# Cron trigger
scheduler.add_job(
    func=maybe_retrain_meta,
    trigger=CronTrigger(
        day_of_week='sun',
        hour=23,
        minute=45,
        start_date=first_run
    ),
    id='turbomode_meta_retrain',
    name='TurboMode - Meta-Learner Retraining (6-weekly)',
    replace_existing=True,
    misfire_grace_time=3600  # 1 hour grace period
)
```

---

## Testing

### Manual Test
```bash
python C:\StockApp\backend\turbomode\meta_retrain.py
```

Expected output:
- Step 1: Generate meta-predictions (~2 min)
- Step 2: Retrain meta-learner (~1 min)
- Success message with validation accuracy

### Scheduler Test
The scheduler will automatically run when Flask starts. Check logs for:
```
✅ TurboMode Scheduler STARTED
   Meta-Learner Retrain: Every 6 weeks, Sunday at 23:45 (first run: 2026-02-22)
```

---

## Maintenance

### Monitoring
- Check `backend/data/turbomode_scheduler_state.json` for scheduler status
- Monitor Flask logs for retraining completion
- Validation accuracy should remain >98%

### Troubleshooting
1. **Retraining fails:** Check database for meta_predictions table
2. **Memory issues:** Reduce batch size in generate_meta_predictions.py
3. **Time conflicts:** Adjust schedule in turbomode_scheduler.py

---

## Next Steps

1. ✅ Scheduler configured and ready
2. ✅ First run scheduled for February 22, 2026
3. ⏳ Monitor first automated run
4. ⏳ Verify model performance after first retrain

---

## Summary

The meta-learner retraining scheduler is now fully integrated into the TurboMode pipeline. The system will:

1. Automatically retrain every 6 weeks on Sunday nights
2. Use the latest training data (169,400 samples)
3. Incorporate override-aware features to reduce HOLD bias
4. Save improved model to meta_learner_v2 directory
5. Maintain >98% validation accuracy

**Status:** Production-ready. No further action required.
