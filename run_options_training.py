"""
Options-Optimized Training Script
Tailored for short-term options trading (puts/calls)

Key Differences from Stock Trading:
- Shorter hold period (5-7 days vs 14 days)
- Higher profit targets (25%+ vs 10%)
- Tighter stop losses (15% vs 5%)
- Focus on volatility and momentum
"""
import sys
sys.path.insert(0, 'backend')

from backend.advanced_ml.training.training_pipeline import TrainingPipeline
from backend.advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from datetime import datetime
import json
import numpy as np


def run_options_backtest(pipeline: TrainingPipeline, symbols: list,
                         hold_days: int = 5,
                         profit_target: float = 25.0,
                         loss_limit: float = 15.0,
                         years: int = 2):
    """
    Run backtest optimized for options trading

    Args:
        hold_days: Days to hold position (5-7 recommended for options)
        profit_target: Target profit % (25-50% for options)
        loss_limit: Maximum loss % (15-20% for options)
    """
    print(f"\n{'=' * 70}")
    print(f"OPTIONS-OPTIMIZED BACKTEST ({len(symbols)} SYMBOLS)")
    print(f"{'=' * 70}")
    print(f"\nOptions Trading Parameters:")
    print(f"  Hold Period: {hold_days} days (quick moves for options)")
    print(f"  Profit Target: +{profit_target}% (options can move big)")
    print(f"  Stop Loss: -{loss_limit}% (protect from theta decay)")
    print(f"\nEstimated time: {len(symbols) * 5}-{len(symbols) * 7} minutes")
    print(f"({len(symbols) * 5 / 60:.1f}-{len(symbols) * 7 / 60:.1f} hours)")
    print(f"{'=' * 70}\n")

    # Modify backtest parameters for options
    backtest = pipeline.backtest
    original_hold = backtest.hold_period
    original_win = backtest.win_threshold
    original_loss = backtest.loss_threshold

    # Set options-specific parameters
    backtest.hold_period = hold_days
    backtest.win_threshold = profit_target
    backtest.loss_threshold = loss_limit

    try:
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")

            try:
                result = backtest.run_backtest([symbol], years=years, save_to_db=True)
                samples_added = result.get('total_trades', 0)

                # Show label distribution for this symbol
                if 'label_distribution' in result:
                    dist = result['label_distribution']
                    buy_pct = dist.get('buy', 0) / samples_added * 100 if samples_added > 0 else 0
                    sell_pct = dist.get('sell', 0) / samples_added * 100 if samples_added > 0 else 0
                    print(f"    [OK] {symbol} complete - {samples_added} samples")
                    print(f"         Buy: {buy_pct:.1f}% | Sell: {sell_pct:.1f}% (good for options!)")
                else:
                    print(f"    [OK] {symbol} complete - {samples_added} samples added")

            except Exception as e:
                print(f"    [FAIL] {symbol} failed: {e}")
                continue
    finally:
        # Restore original parameters
        backtest.hold_period = original_hold
        backtest.win_threshold = original_win
        backtest.loss_threshold = original_loss


def main(hold_days=5, profit_target=25.0, loss_limit=15.0):
    """Run options-optimized training pipeline"""
    print("\n" + "=" * 70)
    print(f"OPTIONS-OPTIMIZED TRAINING PIPELINE")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Features: 179 (technical only, no events)")
    print(f"Models: 3 best models (XGBoost, RF, LightGBM)")
    print(f"Regime Processing: DISABLED")
    print(f"\nOptions Trading Focus:")
    print(f"  Hold Period: {hold_days} days")
    print(f"  Profit Target: +{profit_target}%")
    print(f"  Stop Loss: -{loss_limit}%")
    print("=" * 70)

    # Initialize pipeline
    print("\n[INIT] Initializing training pipeline...")
    pipeline = TrainingPipeline()

    # Same 20 diverse symbols as quick training
    symbols = [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL',  # Tech Large
        'PLTR', 'SNOW', 'CRWD',  # Tech Mid/Small (volatile - great for options!)
        'JPM', 'BAC', 'GS',  # Finance
        'JNJ', 'UNH', 'LLY',  # Healthcare
        'AMZN', 'TSLA', 'HD',  # Consumer
        'XOM', 'CVX',  # Energy
        'CAT', 'GE'  # Industrials
    ]

    print(f"\n[SYMBOLS] Options training on {len(symbols)} symbols")
    print(f"  Focus: High-liquidity stocks with active options markets")

    # PHASE 1: Options-Optimized Backtest
    run_options_backtest(pipeline, symbols,
                        hold_days=hold_days,
                        profit_target=profit_target,
                        loss_limit=loss_limit,
                        years=2)

    # PHASE 2: Load Training Data
    print(f"\n{'=' * 70}")
    print("PHASE 2: LOAD TRAINING DATA")
    print(f"{'=' * 70}\n")

    X_train, X_test, y_train, y_test, sample_weight = pipeline.load_training_data(
        test_size=0.2,
        use_rare_event_archive=False,
        use_regime_processing=False
    )

    print(f"\n[DATA] Training samples: {len(X_train)}")
    print(f"[DATA] Test samples: {len(X_test)}")
    print(f"[DATA] Features per sample: {X_train.shape[1]}")

    # Show label distribution
    from collections import Counter
    train_dist = Counter(y_train)
    total = len(y_train)
    print(f"\n[LABELS] Training distribution:")
    print(f"  Buy (0):  {train_dist[0]:5d} ({train_dist[0]/total*100:5.1f}%)")
    print(f"  Hold (1): {train_dist[1]:5d} ({train_dist[1]/total*100:5.1f}%)")
    print(f"  Sell (2): {train_dist[2]:5d} ({train_dist[2]/total*100:5.1f}%)")

    if X_train.shape[1] != 179:
        print(f"\n[WARNING] Expected 179 features, got {X_train.shape[1]}")
        return

    # PHASE 3: Train Models
    print(f"\n{'=' * 70}")
    print("PHASE 3: TRAIN MODELS (OPTIONS-OPTIMIZED)")
    print(f"{'=' * 70}\n")

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

    # PHASE 4: Evaluate
    print(f"\n{'=' * 70}")
    print("PHASE 4: EVALUATE FOR OPTIONS TRADING")
    print(f"{'=' * 70}\n")

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    results = {}

    for name, model in models_to_train:
        print(f"\n[EVALUATING] {name}...")

        # Test predictions
        feature_dicts = [{f'feature_{j}': float(X_test[i, j]) for j in range(X_test.shape[1])}
                        for i in range(len(X_test))]
        predictions = model.predict_batch(feature_dicts)

        label_map = {'buy': 0, 'hold': 1, 'sell': 2}
        pred_labels = []
        for pred in predictions:
            p = pred['prediction']
            if isinstance(p, (int, np.integer)):
                pred_labels.append(p)
            else:
                pred_labels.append(label_map[p])

        test_accuracy = accuracy_score(y_test, pred_labels)

        # Training accuracy (sample for speed)
        train_sample_size = min(1000, len(X_train))
        train_feature_dicts = [{f'feature_{j}': float(X_train[i, j]) for j in range(X_train.shape[1])}
                               for i in range(train_sample_size)]
        train_preds = model.predict_batch(train_feature_dicts)
        train_labels = []
        for pred in train_preds:
            p = pred['prediction']
            if isinstance(p, (int, np.integer)):
                train_labels.append(p)
            else:
                train_labels.append(label_map[p])
        train_accuracy = accuracy_score(y_train[:train_sample_size], train_labels)

        # Calculate per-class accuracy (important for options!)
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, pred_labels, labels=[0, 1, 2], zero_division=0
        )

        results[name] = {
            'test_accuracy': test_accuracy,
            'train_accuracy': train_accuracy,
            'gap': train_accuracy - test_accuracy,
            'buy_precision': precision[0],
            'buy_recall': recall[0],
            'sell_precision': precision[2],
            'sell_recall': recall[2]
        }

        print(f"\n  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Gap:               {results[name]['gap']:.4f} ({results[name]['gap']*100:.2f}%)")

        print(f"\n  OPTIONS TRADING METRICS (most important!):")
        print(f"    Buy Signal Precision:  {precision[0]:.4f} (when it says buy, is it right?)")
        print(f"    Buy Signal Recall:     {recall[0]:.4f} (does it catch good buys?)")
        print(f"    Sell Signal Precision: {precision[2]:.4f} (when it says sell, is it right?)")
        print(f"    Sell Signal Recall:    {recall[2]:.4f} (does it catch good sells?)")

        # Confusion matrix
        cm = confusion_matrix(y_test, pred_labels, labels=[0, 1, 2])
        print(f"\n  Confusion Matrix:")
        print(f"           Pred: Buy  Hold  Sell")
        print(f"    True Buy:  {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
        print(f"    True Hold: {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
        print(f"    True Sell: {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")

    # Final Summary
    print(f"\n{'=' * 70}")
    print("OPTIONS TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  - Hold Period: {hold_days} days")
    print(f"  - Profit Target: +{profit_target}%")
    print(f"  - Stop Loss: -{loss_limit}%")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features: {X_train.shape[1]}")

    print(f"\nModel Performance:")
    for name, result in results.items():
        status = "✓ EXCELLENT" if result['test_accuracy'] > 0.85 else \
                 "✓ GOOD" if result['test_accuracy'] > 0.75 else \
                 "~ OK" if result['test_accuracy'] > 0.65 else \
                 "✗ NEEDS WORK"
        print(f"  {name:15} Test: {result['test_accuracy']*100:5.2f}%  Gap: {result['gap']*100:5.2f}%  {status}")
        print(f"                 Buy Precision: {result['buy_precision']*100:5.2f}%  Sell Precision: {result['sell_precision']*100:5.2f}%")

    # Save results
    results_file = f'backend/data/options_training_{hold_days}d_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'hold_days': hold_days,
                'profit_target': profit_target,
                'loss_limit': loss_limit
            },
            'symbols': symbols,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'results': results
        }, f, indent=2, default=str)

    print(f"\n  Results saved: {results_file}")
    print("=" * 70)

    # Check accuracy
    best_test_acc = max([r['test_accuracy'] for r in results.values()])
    print(f"\n{'=' * 70}")
    print("OPTIONS TRADING ACCURACY CHECK:")
    if best_test_acc >= 0.90:
        print("✓ EXCELLENT! Test accuracy ≥ 90% - Ready for options trading!")
    elif best_test_acc >= 0.80:
        print("✓ GOOD! Test accuracy ≥ 80% - Strong signal for options")
    elif best_test_acc >= 0.70:
        print("~ ACCEPTABLE - Test accuracy ≥ 70% - Use with caution")
    else:
        print("✗ RISKY - Test accuracy < 70% - Not recommended for real trading")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Options-optimized ML training')
    parser.add_argument('--hold-days', type=int, default=5,
                       help='Hold period in days (default: 5)')
    parser.add_argument('--profit-target', type=float, default=25.0,
                       help='Profit target % (default: 25.0)')
    parser.add_argument('--loss-limit', type=float, default=15.0,
                       help='Stop loss % (default: 15.0)')

    args = parser.parse_args()

    try:
        results = main(
            hold_days=args.hold_days,
            profit_target=args.profit_target,
            loss_limit=args.loss_limit
        )
        print("\n[SUCCESS] Options training pipeline completed!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
