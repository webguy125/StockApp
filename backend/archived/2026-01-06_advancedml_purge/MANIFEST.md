# TurboMode Purification Archive
## Date: 2026-01-06

This archive contains legacy code that was replaced during the TurboMode purification process.

## Archived Files

### train_turbomode_models_LEGACY.py
- **Original purpose**: Model training script
- **Reason for archival**: Contained AdvancedML contamination (AdvancedMLDatabase, HistoricalBacktest)
- **Replaced by**: Pure TurboMode train_turbomode_models.py
- **AdvancedML dependencies removed**:
  - `from advanced_ml.database.schema import AdvancedMLDatabase`
  - `from advanced_ml.backtesting.historical_backtest import HistoricalBacktest`
  - `db = AdvancedMLDatabase(db_path)`
  - `backtest = HistoricalBacktest(db_path)`
  - `X, y = backtest.prepare_training_data()`

## New Pure TurboMode Components

### turbomode_feature_extractor.py
- **Purpose**: Extract 179 technical features from price data
- **Dependencies**: Master Market Data API (read-only), GPUFeatureEngineer (computational utility)
- **NO AdvancedML contamination**: Uses only feature calculation libraries, not database/schema

### extract_features.py
- **Purpose**: Batch-populate entry_features_json for all 169,400 samples
- **Features**: Resume-safe, batch processing, progress tracking
- **NO AdvancedML contamination**: Pure TurboMode pipeline

### turbomode_training_loader.py
- **Purpose**: Load training data from turbomode.db
- **Replaces**: HistoricalBacktest.prepare_training_data()
- **NO AdvancedML contamination**: Reads directly from turbomode.db

### train_turbomode_models.py (REWRITTEN)
- **Purpose**: Train all 9 models (8 base + 1 meta-learner)
- **Changes**: Removed ALL AdvancedML database/backtest imports
- **Uses**: TurboModeTrainingDataLoader instead of HistoricalBacktest
- **NO AdvancedML contamination**: 100% pure TurboMode

## Purification Summary

**Total AdvancedML imports removed**: 2
- AdvancedMLDatabase
- HistoricalBacktest

**Total new TurboMode components created**: 3
- TurboModeFeatureExtractor
- TurboModeTrainingDataLoader
- FeatureExtractionPipeline (extract_features.py)

**Status**: ✅ PURIFICATION COMPLETE - TurboMode is now 100% autonomous
