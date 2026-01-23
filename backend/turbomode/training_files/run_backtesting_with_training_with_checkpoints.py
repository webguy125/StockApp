"""
Checkpoint-Enabled Full Training Pipeline
Resume-capable training with save points after each symbol

Features:
- Saves progress after each symbol in backtest
- Can resume after restart/crash
- Preserves all training data in database
- Automatic checkpoint management
"""

import sys
sys.path.insert(0, 'backend')

from backend.advanced_ml.training.training_pipeline import TrainingPipeline
from backend.advanced_ml.training.checkpoint_manager import CheckpointManager
from backend.advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from backend.advanced_ml.config.core_symbols import CORE_SYMBOLS
from datetime import datetime
import json
import numpy as np


def run_backtest_with_checkpoints(pipeline: TrainingPipeline, checkpoint: CheckpointManager,
                                  symbols: list, years: int = 2):
    """
    Run backtest with checkpoint after each symbol

    Args:
        pipeline: Training pipeline
        checkpoint: Checkpoint manager
        symbols: All symbols to process
        years: Years of history
    """
    print(f"\n{'=' * 70}")
    print("PHASE 1: HISTORICAL BACKTEST (WITH CHECKPOINTS)")
    print(f"{'=' * 70}")

    # Get remaining symbols (skip already completed)
    remaining_symbols = checkpoint.get_remaining_symbols(symbols)

    if len(remaining_symbols) == 0:
        print(f"\n[CHECKPOINT] All {len(symbols)} symbols already processed!")
        print(f"             Total samples: {checkpoint.state['backtest']['total_samples']}")
        checkpoint.mark_backtest_complete()
        return

    print(f"\nSymbols to process: {len(remaining_symbols)}/{len(symbols)}")
    print(f"Estimated time: {len(remaining_symbols) * 5}-{len(remaining_symbols) * 7} minutes")
    print(f"({len(remaining_symbols) * 5 / 60:.1f}-{len(remaining_symbols) * 7 / 60:.1f} hours)")
    print(f"{'=' * 70}\n")

    checkpoint.mark_backtest_start()

    # Process each symbol individually with checkpoints
    backtest = pipeline.backtest

    for i, symbol in enumerate(remaining_symbols, 1):
        print(f"\n[{i}/{len(remaining_symbols)}] Processing {symbol}...")

        try:
            # Run backtest for single symbol
            result = backtest.run_backtest([symbol], years=years, save_to_db=True)

            samples_added = result.get('total_trades', 0)
            checkpoint.mark_symbol_complete(symbol, samples_added)

            print(f"    [OK] {symbol} complete - {samples_added} samples added")

        except Exception as e:
            print(f"    [FAIL] {symbol} failed: {e}")
            checkpoint.mark_symbol_failed(symbol, str(e))
            continue

    # Mark backtest complete
    checkpoint.mark_backtest_complete()


def run_training_with_checkpoints(pipeline: TrainingPipeline, checkpoint: CheckpointManager,
                                  X_train, y_train, sample_weight):
    """
    Run model training with checkpoints after each model

    Args:
        pipeline: Training pipeline
        checkpoint: Checkpoint manager
        X_train: Training features
        y_train: Training labels
        sample_weight: Sample weights
    """
    print(f"\n{'=' * 70}")
    print("PHASE 3: TRAIN BASE MODELS (WITH CHECKPOINTS)")
    print(f"{'=' * 70}\n")

    models = [
        ('random_forest', pipeline.rf_model),
        ('xgboost', pipeline.xgb_model),
        ('lightgbm', pipeline.lgbm_model),
        ('extratrees', pipeline.et_model),
        ('gradientboost', pipeline.gb_model),
        ('neural_network', pipeline.nn_model),
        ('logistic_regression', pipeline.lr_model),
        ('svm', pipeline.svm_model)
    ]

    trained_models = checkpoint.state['training']['base_models_trained']

    for model_name, model in models:
        if model_name in trained_models:
            print(f"[SKIP] {model_name} already trained")
            continue

        print(f"\n[TRAINING] {model_name}...")
        start_time = datetime.now()

        try:
            # Train with sample weights
            model.train(X_train, y_train, sample_weight=sample_weight)

            # Save model to disk
            model.save()

            duration = (datetime.now() - start_time).total_seconds()
            print(f"    [OK] {model_name} trained in {duration:.1f}s")

            checkpoint.mark_model_trained(model_name)

        except Exception as e:
            print(f"    [FAIL] {model_name} failed: {e}")
            raise


def main():
    """Run full training pipeline with checkpoints"""
    print("\n" + "=" * 70)
    print("CHECKPOINT-ENABLED TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Features: 179 (technical only, no events)")
    print(f"Models: 8 base models + meta-learner")
    print(f"Regime Processing: DISABLED")
    print(f"\nYou can safely stop/restart this script at any time!")
    print("=" * 70)

    # Initialize checkpoint manager
    checkpoint = CheckpointManager()
    print(checkpoint.get_summary())

    # Initialize pipeline
    print("\n[INIT] Initializing training pipeline...")
    pipeline = TrainingPipeline()

    # Get ALL core symbols (82 total)
    all_symbols = []
    for sector, market_caps in CORE_SYMBOLS.items():
        for cap_category, symbol_list in market_caps.items():
            all_symbols.extend(symbol_list)

    symbols = all_symbols  # All 82 symbols

    print(f"\n[SYMBOLS] Training on ALL {len(symbols)} symbols:")
    print(f"  Sectors: 11 (all GICS sectors)")
    print(f"  Market Caps: Large, Mid, Small (balanced)")

    # PHASE 1: Historical Backtest with Checkpoints
    if checkpoint.state['current_phase'] in ['initialization', 'backtest']:
        run_backtest_with_checkpoints(pipeline, checkpoint, symbols, years=2)
    else:
        print(f"\n[SKIP] Backtest already complete ({checkpoint.state['backtest']['total_samples']} samples)")

    # PHASE 2: Load Training Data
    print(f"\n{'=' * 70}")
    print("PHASE 2: LOAD TRAINING DATA")
    print(f"{'=' * 70}\n")

    X_train, X_test, y_train, y_test, sample_weight = pipeline.load_training_data(
        test_size=0.2,
        use_rare_event_archive=False,  # DISABLED: Rare events had wrong feature dimensions
        use_regime_processing=False     # DISABLED: Was losing 40% of training data
    )

    print(f"\n[DATA] Training samples: {len(X_train)}")
    print(f"[DATA] Test samples: {len(X_test)}")
    print(f"[DATA] Features per sample: {X_train.shape[1]}")

    if X_train.shape[1] != 179:
        print(f"\n[WARNING] Expected 179 features (technical only), got {X_train.shape[1]}")

    # PHASE 3: Train Base Models with Checkpoints
    if len(checkpoint.state['training']['base_models_trained']) < 8:
        run_training_with_checkpoints(pipeline, checkpoint, X_train, y_train, sample_weight)
    else:
        print(f"\n[SKIP] All 8 base models already trained")
        # Load all trained models that weren't auto-loaded
        print(f"\n[LOADING] Ensuring all models are loaded...")
        models_to_check = [
            ('lightgbm', pipeline.lgbm_model),
            ('extratrees', pipeline.et_model),
            ('gradientboost', pipeline.gb_model),
            ('neural_network', pipeline.nn_model),
            ('logistic_regression', pipeline.lr_model),
            ('svm', pipeline.svm_model)
        ]
        for name, model in models_to_check:
            if not model.is_trained:
                print(f"  Loading {name}...")
                model.load()
        print(f"[OK] All models loaded")

    # PHASE 4: Train Meta-Learner
    if not checkpoint.state['training']['meta_learner_trained']:
        print(f"\n{'=' * 70}")
        print("PHASE 4: TRAIN META-LEARNER")
        print(f"{'=' * 70}\n")

        # FIXED: Train meta-learner on TRAINING data, not test data!
        pipeline.train_meta_learner(X_train, y_train)
        # Save meta-learner
        pipeline.meta_learner.save()
        checkpoint.mark_meta_learner_trained()
    else:
        print(f"\n[SKIP] Meta-learner already trained")
        # Load meta-learner if not already loaded
        if not pipeline.meta_learner.is_trained:
            print(f"[LOADING] Loading meta-learner...")
            pipeline.meta_learner.load()

    # PHASE 5: Evaluate Models
    if not checkpoint.state['evaluation']['completed']:
        print(f"\n{'=' * 70}")
        print("PHASE 5: EVALUATE MODELS")
        print(f"{'=' * 70}\n")

        results = pipeline.evaluate_models(X_test, y_test)

        # PHASE 6: Regime Validation (SKIPPED - method not implemented)
        # print(f"\n{'=' * 70}")
        # print("PHASE 6: REGIME-AWARE VALIDATION")
        # print(f"{'=' * 70}\n")
        # regime_validation = pipeline.create_regime_validation_sets(X_test, y_test)
        # regime_results = pipeline.evaluate_models_per_regime(regime_validation)

        regime_results = {}  # Initialize for later use
        checkpoint.mark_evaluation_complete({
            'model_results': results,
            'regime_results': regime_results
        })
    else:
        print(f"\n[SKIP] Evaluation already complete")
        results = checkpoint.state['evaluation']['results'].get('model_results', {})
        regime_results = checkpoint.state['evaluation']['results'].get('regime_results', {})

    # PHASE 7: Save Results
    print(f"\n{'=' * 70}")
    print("PHASE 7: SAVE RESULTS")
    print(f"{'=' * 70}\n")

    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'feature_count': X_train.shape[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'symbols': symbols,
        'checkpoint_state': checkpoint.state,
        'model_results': results,
        'regime_results': regime_results
    }

    results_file = 'backend/data/training_results_checkpoint.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"[OK] Results saved to {results_file}")

    # Final Summary
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(checkpoint.get_summary())
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Models trained: 9 (8 base + meta-learner)")
    print(f"  - Results: {results_file}")
    print("=" * 70)

    return results_summary


if __name__ == "__main__":
    try:
        results = main()
        print("\n[SUCCESS] Full training pipeline completed!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training stopped by user")
        print("Progress saved! Run this script again to resume from checkpoint.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nProgress saved! Run this script again to resume from checkpoint.")
        sys.exit(1)
