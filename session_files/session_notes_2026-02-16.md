SESSION STARTED AT: 2026-02-16 05:29

============================================
[2026-02-16 05:45] TRAINING FAILURE DIAGNOSIS - TIMEOUT ISSUE
============================================

## Issue Summary

Training Task 2 failed last night (2026-02-16 00:00) due to timeout after 2 hours.

## Evidence from Logs

**Scheduler Log (backend/logs/scheduler.log):**
```
[2026-02-16 00:00:00] scheduler - INFO - [ASYNC] Task 2 started in daemon thread
[2026-02-16 00:00:00] scheduler - INFO - [SCHEDULER] Job task_2 completed successfully
[2026-02-16 02:00:00] scheduler - ERROR - [TIMEOUT] Task 2: Task exceeded timeout of 120 minutes
```

## Root Cause

**1. Timeout Configuration is TOO SHORT**
   - Current timeout: 120 minutes (2 hours)
   - Actual training time needed: ~4-5 hours
   - Location: backend/scheduler_config.json line 69

**2. Data Volume Changed Since Last Training**
   - Last successful training: 2026-02-01 (old dataset, 61.9% HOLD labels)
   - New dataset: 1,638,941 samples with MFE/MAE logic (51.4% HOLD labels)
   - Training data was regenerated on 2026-02-15 (4 hours 15 minutes)
   - New data has different distribution, likely requires more training time

**3. Training Script Called**
   - unified_scheduler.py:251 imports `train_all_sectors_optimized_orchestrator`
   - This trains 66 models (11 sectors × 6 models)
   - Expected duration: 2-3 hours (ESTIMATE WAS WRONG)

## Timeline of Events

**2026-02-01 00:00:** Last successful training (old dataset)
- Duration: 1 hour 46 minutes (within 120-minute timeout)
- Dataset: Old close-to-close labels

**2026-02-15 14:35:** New MFE/MAE dataset generated
- 1,638,941 samples
- 4 hours 15 minutes to generate
- Label distribution: BUY 23.5%, SELL 25.1%, HOLD 51.4%

**2026-02-16 00:00:** Scheduled training started (Task 2)
- Scheduler triggered on time (Monday midnight)
- Training started in daemon thread

**2026-02-16 02:00:** Training KILLED by timeout
- Hit 120-minute hard limit
- Models partially trained or not saved
- Checkpoint file still shows old date: 2026-01-08

## Configuration Details

**File:** backend/scheduler_config.json
**Task ID:** 2
**Setting:** "timeout_minutes": 120

**Code Reference:** backend/unified_scheduler.py:1081-1113
```python
timeout_minutes = task_config.get('timeout_minutes', 60)
# ...
timeout_func = task_timeout(timeout_minutes)(task_func)
# ...
except TaskTimeoutError as e:
    last_error = f"Task exceeded timeout of {timeout_minutes} minutes"
```

## Impact Assessment

**Training Status:** FAILED - No new models trained
**Models in Use:** OLD (from 2026-01-08)
- Still using close-to-close labels
- Not using MFE/MAE path-dependent logic
- Train/serve mismatch active

**System Operation:** DEGRADED
- Scanner is operational but using outdated models
- Predictions based on old labeling methodology
- 14-day alignment incomplete

## Recommended Fix

**Option 1: Increase Timeout (RECOMMENDED)**
- Change timeout_minutes from 120 → 300 (5 hours)
- File: backend/scheduler_config.json line 69
- Safe buffer for 66-model training run

**Option 2: Manual Training Run**
- Run training manually outside scheduler
- Command: `python backend/turbomode/core_engine/train_all_sectors_optimized_orchestrator.py`
- Monitor duration to calibrate timeout
- Then update scheduler_config.json

**Option 3: Optimize Training Script**
- Reduce model complexity
- Use faster hyperparameters
- Add checkpointing/resume capability
- Requires code changes

## Status

**Diagnosis:** COMPLETE
**Fix Status:** PENDING USER DECISION
**No changes made** (as per user request)

============================================

[2026-02-16 14:03] SMOKE TEST STARTED - REAL_ESTATE SECTOR
============================================

**Objective:** Validate new MFE/MAE training pipeline before full 66-model run

**Command:**
```bash
python backend/turbomode/core_engine/train_all_sectors_optimized_orchestrator.py --smoke-test real_estate
```

**Test Scope:**
- Sector: real_estate (1 of 11 sectors)
- Models: 6 (5 base + 1 meta-learner)
- Duration: Expected 15-30 minutes
- Purpose: Validate training works with new MFE/MAE labels

**Training Configuration:**
- Label: 14-day MFE/MAE path-dependent (±5% threshold)
- Horizon: 14d (ENFORCED)
- Dataset: 1,638,941 samples generated 2026-02-15

**Status:** Running - preloading real_estate sector data...

============================================

