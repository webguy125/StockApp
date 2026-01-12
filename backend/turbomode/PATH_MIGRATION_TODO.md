# Path Migration TODO

## Goal
Migrate all scripts from hardcoded relative paths to centralized `paths.py` module

## Migration Pattern

**Before:**
```python
db_path = "backend/data/turbomode.db"  # ❌ Relative path
```

**After:**
```python
from backend.turbomode.paths import TURBOMODE_DB
db_path = str(TURBOMODE_DB)  # ✅ Absolute path
```

---

## Scripts to Migrate (by priority)

### High Priority (Core Production Scripts)
- [ ] `overnight_scanner.py` - Daily production scan
- [ ] `outcome_tracker.py` - Signal evaluation
- [ ] `adaptive_stock_ranker.py` - Top 10 rankings
- [ ] `training_sample_generator.py` - Training data generation
- [ ] `automated_retrainer.py` - Monthly model retraining
- [ ] `meta_retrain.py` - Meta-learner retraining

### Medium Priority (Scheduled/Automated)
- [ ] `turbomode_scheduler.py` - Job scheduler
- [ ] `generate_backtest_data.py` - Training data backtest
- [ ] `train_turbomode_models.py` - Model training
- [ ] `train_specialized_meta_learner.py` - Meta-learner training

### Low Priority (Utilities/Tools)
- [ ] `turbomode_backtest.py` - Backtest engine
- [ ] `database_schema.py` - Schema management
- [ ] `schema_guardrail.py` - Schema validation
- [ ] `turbomode_training_loader.py` - Training data loader
- [ ] `drift_monitor.py` - Model drift detection
- [ ] `backtest_generator.py` - Backtest generation
- [ ] `extract_features.py` - Feature extraction
- [ ] `options_data_collector.py` - Options data collection

### Very Low Priority (One-off/Diagnostic)
- [ ] `check_features_status.py`
- [ ] `create_turbomode_training_tables.py`
- [ ] `inspect_training_data.py`
- [ ] `inspect_training_logic.py`
- [ ] `migrate_add_entry_range.py`
- [ ] `migrate_signal_history.py`
- [ ] `training_orchestrator.py`
- [ ] `update_turbomode_schema.py`
- [ ] `generate_meta_predictions.py`
- [ ] `retrain_meta_with_override_features.py`

### Already Using Absolute Paths ✅
- [x] `populate_signal_history.py` - Uses relative path but not in production (can delete)

---

## Migration Strategy

1. **New scripts**: Always use `paths.py` from day one
2. **Existing scripts**: Migrate when touching the file for other reasons
3. **Critical scripts**: Migrate when doing major maintenance
4. **Utility scripts**: Migrate when convenient

---

## Benefits After Migration

- ✅ No more accidental database creation in wrong locations
- ✅ Scripts work from any directory
- ✅ Single source of truth for all paths
- ✅ Easier to refactor directory structure later
- ✅ Better error messages (absolute paths in errors)

---

## Notes

- Migration is non-breaking (old paths still work)
- Can test each migration independently
- No need to migrate all at once
- Delete this file when all scripts migrated
