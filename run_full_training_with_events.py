"""
Full Training Pipeline with Event Features
Runs complete training cycle: backtest → train → evaluate → save
Features: 202 total (179 technical + 23 event)
"""

import sys
sys.path.insert(0, 'backend')

from backend.advanced_ml.training.training_pipeline import TrainingPipeline
from backend.advanced_ml.config.core_symbols import CORE_SYMBOLS
from datetime import datetime
import json


def main():
    """Run full training pipeline with event features"""
    print("\n" + "=" * 70)
    print("FULL TRAINING PIPELINE - WITH EVENT FEATURES")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Features: 202 (179 technical + 23 event)")
    print(f"Models: 8 base models + meta-learner")
    print(f"Regime Processing: Enabled")
    print("=" * 70)

    # Initialize pipeline
    print("\n[INIT] Initializing training pipeline...")
    pipeline = TrainingPipeline()

    # Get ALL core symbols (82 total across all sectors and market caps)
    # Extract actual stock symbols from nested dictionary
    all_symbols = []
    for sector, market_caps in CORE_SYMBOLS.items():
        for cap_category, symbol_list in market_caps.items():
            all_symbols.extend(symbol_list)

    # Use ALL symbols for complete training dataset
    symbols = all_symbols  # All 82 symbols
    print(f"\n[SYMBOLS] Training on ALL {len(symbols)} symbols:")
    print(f"  Sectors: 11 (all GICS sectors)")
    print(f"  Market Caps: Large, Mid, Small (balanced)")
    print(f"\n  Symbol breakdown by sector:")

    # Show breakdown
    for sector, market_caps in CORE_SYMBOLS.items():
        sector_count = sum(len(symbols) for symbols in market_caps.values())
        print(f"    {sector}: {sector_count} symbols")

    print(f"\n  First 10 symbols: {', '.join(symbols[:10])}")
    print(f"  ... and {len(symbols)-10} more")

    # Phase 1: Historical Backtest
    print(f"\n{'=' * 70}")
    print("PHASE 1: HISTORICAL BACKTEST (FULL DATASET)")
    print(f"{'=' * 70}")
    print(f"This will generate training data with 202 features per sample")
    print(f"Symbols: {len(symbols)}")
    print(f"Estimated time: {len(symbols) * 5}-{len(symbols) * 7} minutes ({len(symbols) * 5 / 60:.1f}-{len(symbols) * 7 / 60:.1f} hours)")
    print(f"Features per sample: 202 (179 technical + 23 event)")
    print(f"Expected training samples: ~{len(symbols) * 500}-{len(symbols) * 800}")
    print(f"{'=' * 70}\n")

    backtest_results = pipeline.run_backtest(symbols, years=2)

    # Phase 2: Load Training Data
    print(f"\n{'=' * 70}")
    print("PHASE 2: LOAD TRAINING DATA")
    print(f"{'=' * 70}\n")

    X_train, X_test, y_train, y_test, sample_weight = pipeline.load_training_data(
        test_size=0.2,
        use_rare_event_archive=True,
        use_regime_processing=True
    )

    print(f"\n[DATA] Training samples: {len(X_train)}")
    print(f"[DATA] Test samples: {len(X_test)}")
    print(f"[DATA] Features per sample: {X_train.shape[1]}")
    print(f"[DATA] Expected: 202 features")

    # Verify feature count
    if X_train.shape[1] != 202:
        print(f"\n[WARNING] Feature count mismatch!")
        print(f"  Expected: 202")
        print(f"  Got: {X_train.shape[1]}")
        print(f"  Continuing anyway...")

    # Phase 3: Train All Models
    print(f"\n{'=' * 70}")
    print("PHASE 3: TRAIN ALL 8 BASE MODELS")
    print(f"{'=' * 70}")
    print(f"Training with regime weighting enabled")
    print(f"Estimated time: 2-3 hours")
    print(f"{'=' * 70}\n")

    pipeline.train_base_models(X_train, y_train, sample_weight=sample_weight)

    # Phase 4: Train Meta-Learner
    print(f"\n{'=' * 70}")
    print("PHASE 4: TRAIN META-LEARNER")
    print(f"{'=' * 70}\n")

    pipeline.train_meta_learner(X_test, y_test)

    # Phase 5: Evaluate Models
    print(f"\n{'=' * 70}")
    print("PHASE 5: EVALUATE MODELS")
    print(f"{'=' * 70}\n")

    results = pipeline.evaluate_models(X_test, y_test)

    # Phase 6: Create Regime Validation Sets
    print(f"\n{'=' * 70}")
    print("PHASE 6: REGIME-AWARE VALIDATION")
    print(f"{'=' * 70}\n")

    regime_validation = pipeline.create_regime_validation_sets(X_test, y_test)
    regime_results = pipeline.evaluate_models_per_regime(regime_validation)

    # Phase 7: Save Results
    print(f"\n{'=' * 70}")
    print("PHASE 7: SAVE RESULTS")
    print(f"{'=' * 70}\n")

    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'feature_count': X_train.shape[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'symbols': symbols,
        'backtest_results': backtest_results,
        'model_results': results,
        'regime_results': regime_results
    }

    # Save to file
    results_file = 'backend/data/training_results_with_events.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"[OK] Results saved to {results_file}")

    # Final Summary
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults Summary:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features: {X_train.shape[1]} (expected 202)")
    print(f"  - Models trained: 9 (8 base + 1 meta-learner)")
    print(f"  - Results saved: {results_file}")
    print(f"\nNext Steps:")
    print(f"  1. Run SHAP analysis")
    print(f"  2. Validate with promotion gate")
    print(f"  3. Deploy to production")
    print("=" * 70)

    return results_summary


if __name__ == "__main__":
    try:
        results = main()
        print("\n[SUCCESS] Full training pipeline completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
