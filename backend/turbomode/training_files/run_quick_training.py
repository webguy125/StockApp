"""
Quick Training Script - 20 Symbols for Fast Validation
Use this to quickly verify model performance before running full 82-symbol training
"""
import sys
sys.path.insert(0, 'backend')

from backend.advanced_ml.training.training_pipeline import TrainingPipeline
from backend.advanced_ml.training.checkpoint_manager import CheckpointManager
from backend.advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from datetime import datetime
import json
import numpy as np


def run_backtest_quick(pipeline: TrainingPipeline, symbols: list, years: int = 2):
    """Run backtest on limited symbols for quick validation"""
    print(f"\n{'=' * 70}")
    print("QUICK BACKTEST (20 SYMBOLS)")
    print(f"{'=' * 70}")
    print(f"\nSymbols to process: {len(symbols)}")
    print(f"Estimated time: {len(symbols) * 5}-{len(symbols) * 7} minutes")
    print(f"({len(symbols) * 5 / 60:.1f}-{len(symbols) * 7 / 60:.1f} hours)")
    print(f"{'=' * 70}\n")

    backtest = pipeline.backtest

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

        try:
            result = backtest.run_backtest([symbol], years=years, save_to_db=True)
            samples_added = result.get('total_trades', 0)
            print(f"    [OK] {symbol} complete - {samples_added} samples added")
        except Exception as e:
            print(f"    [FAIL] {symbol} failed: {e}")
            continue


def main():
    """Run quick training pipeline on 20 symbols"""
    print("\n" + "=" * 70)
    print("QUICK TRAINING PIPELINE (20 SYMBOLS)")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Features: 179 (technical only, no events)")
    print(f"Models: 8 base models + meta-learner")
    print(f"Regime Processing: DISABLED")
    print(f"\nPurpose: Fast validation of baseline performance")
    print("=" * 70)

    # Initialize pipeline
    print("\n[INIT] Initializing training pipeline...")
    pipeline = TrainingPipeline()

    # Select 20 diverse symbols (mix of sectors and market caps)
    quick_symbols = [
        # Tech (Large Cap)
        'AAPL', 'MSFT', 'NVDA', 'GOOGL',
        # Tech (Mid/Small)
        'PLTR', 'SNOW', 'CRWD',
        # Financial
        'JPM', 'BAC', 'GS',
        # Healthcare
        'JNJ', 'UNH', 'LLY',
        # Consumer
        'AMZN', 'TSLA', 'HD',
        # Energy
        'XOM', 'CVX',
        # Industrials
        'CAT', 'GE'
    ]

    print(f"\n[SYMBOLS] Quick training on {len(quick_symbols)} diverse symbols:")
    print(f"  Sectors: Tech, Finance, Healthcare, Consumer, Energy, Industrials")
    print(f"  Market Caps: Large, Mid, Small (balanced)")

    # PHASE 1: Historical Backtest
    run_backtest_quick(pipeline, quick_symbols, years=2)

    # PHASE 2: Load Training Data
    print(f"\n{'=' * 70}")
    print("PHASE 2: LOAD TRAINING DATA")
    print(f"{'=' * 70}\n")

    result = pipeline.load_training_data(
        test_size=0.2,
        use_rare_event_archive=False,  # DISABLED
        use_regime_processing=False     # DISABLED
    )

    # Handle different return values (with/without sample_weight)
    if len(result) == 5:
        X_train, X_test, y_train, y_test, sample_weight = result
    else:
        X_train, X_test, y_train, y_test = result
        sample_weight = None  # No sample weighting

    print(f"\n[DATA] Training samples: {len(X_train)}")
    print(f"[DATA] Test samples: {len(X_test)}")
    print(f"[DATA] Features per sample: {X_train.shape[1]}")

    if X_train.shape[1] != 179:
        print(f"\n[WARNING] Expected 179 features (technical only), got {X_train.shape[1]}")
        print(f"[WARNING] This may indicate event features are still enabled!")
        return

    # PHASE 3: Train Base Models (just the best ones for speed)
    print(f"\n{'=' * 70}")
    print("PHASE 3: TRAIN KEY MODELS (QUICK)")
    print(f"{'=' * 70}\n")

    # Train only the 3 best models for quick validation
    models_to_train = [
        ('XGBoost', pipeline.xgb_model),
        ('Random Forest', pipeline.rf_model),
        ('LightGBM', pipeline.lgbm_model)
    ]

    for name, model in models_to_train:
        print(f"\n[TRAINING] {name}...")
        start = datetime.now()
        model.train(X_train, y_train, sample_weight=sample_weight)
        model.save()
        duration = (datetime.now() - start).total_seconds()
        print(f"    [OK] {name} trained in {duration:.1f}s")

    # PHASE 4: Evaluate Models
    print(f"\n{'=' * 70}")
    print("PHASE 4: EVALUATE MODELS")
    print(f"{'=' * 70}\n")

    from sklearn.metrics import accuracy_score, classification_report

    results = {}

    for name, model in models_to_train:
        print(f"\n[EVALUATING] {name}...")

        # Convert test data to feature dicts
        feature_dicts = [{f'feature_{j}': float(X_test[i, j]) for j in range(X_test.shape[1])}
                        for i in range(len(X_test))]

        # Get predictions
        predictions = model.predict_batch(feature_dicts)

        # Extract labels
        label_map = {'buy': 0, 'hold': 1, 'sell': 2}
        pred_labels = []
        for pred in predictions:
            p = pred['prediction']
            if isinstance(p, (int, np.integer)):
                pred_labels.append(p)
            else:
                pred_labels.append(label_map[p])

        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, pred_labels)

        # Get training accuracy
        train_feature_dicts = [{f'feature_{j}': float(X_train[i, j]) for j in range(X_train.shape[1])}
                               for i in range(min(1000, len(X_train)))]  # Sample 1000 for speed
        train_preds = model.predict_batch(train_feature_dicts)
        train_labels = []
        for pred in train_preds:
            p = pred['prediction']
            if isinstance(p, (int, np.integer)):
                train_labels.append(p)
            else:
                train_labels.append(label_map[p])
        train_accuracy = accuracy_score(y_train[:len(train_labels)], train_labels)

        results[name] = {
            'test_accuracy': test_accuracy,
            'train_accuracy': train_accuracy,
            'gap': train_accuracy - test_accuracy
        }

        print(f"\n  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Gap:               {results[name]['gap']:.4f} ({results[name]['gap']*100:.2f}%)")

        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, pred_labels,
                                   target_names=['buy', 'hold', 'sell'],
                                   digits=4))

    # Final Summary
    print(f"\n{'=' * 70}")
    print("QUICK TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults Summary:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Symbols: {len(quick_symbols)}")

    print(f"\nModel Performance:")
    for name, result in results.items():
        status = "✓ EXCELLENT" if result['test_accuracy'] > 0.85 else \
                 "✓ GOOD" if result['test_accuracy'] > 0.75 else \
                 "~ OK" if result['test_accuracy'] > 0.65 else \
                 "✗ NEEDS WORK"
        print(f"  {name:15} Test: {result['test_accuracy']*100:5.2f}%  Gap: {result['gap']*100:5.2f}%  {status}")

    # Save results
    results_file = 'backend/data/quick_training_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'symbols': quick_symbols,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'results': results
        }, f, indent=2, default=str)

    print(f"\n  Results saved: {results_file}")
    print("=" * 70)

    # Check if we're back at baseline
    best_test_acc = max([r['test_accuracy'] for r in results.values()])
    print(f"\n{'=' * 70}")
    if best_test_acc >= 0.90:
        print("✓ SUCCESS! Test accuracy ≥ 90% - BASELINE RESTORED!")
    elif best_test_acc >= 0.80:
        print("✓ GOOD! Test accuracy ≥ 80% - Close to baseline")
    elif best_test_acc >= 0.70:
        print("~ ACCEPTABLE - Test accuracy ≥ 70% - Improvements needed")
    else:
        print("✗ PROBLEM - Test accuracy < 70% - Further debugging required")
    print("=" * 70)

    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n[SUCCESS] Quick training pipeline completed!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
