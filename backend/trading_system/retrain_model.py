"""
Retrain ML Model
Uses tracked trade outcomes to improve predictions
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_system.core.trade_tracker import TradeTracker
from trading_system.models.simple_trading_model import SimpleTradingModel


def retrain_from_trades():
    """Retrain model using completed trades"""
    tracker = TradeTracker()
    model = SimpleTradingModel()

    print("\n" + "=" * 60)
    print("ML MODEL RETRAINING")
    print("=" * 60 + "\n")

    # Get completed trades
    trades = tracker.get_trades(status='win', limit=1000)
    trades.extend(tracker.get_trades(status='loss', limit=1000))

    if len(trades) < 10:
        print(f"⚠️ Not enough trades for training (need at least 10, have {len(trades)})")
        print("Record more trades and try again.")
        return

    print(f"Found {len(trades)} completed trades")

    # Extract features and labels
    features_list = []
    labels_list = []

    for trade in trades:
        # Skip if no features stored
        if not trade.get('features'):
            continue

        try:
            import json
            features = json.loads(trade['features'])

            # Convert to numpy array
            features_array = np.array(features)

            # Label: 2=BUY (win), 0=SELL (loss), 1=HOLD (neutral)
            if trade['outcome'] == 'win':
                label = 2  # BUY signal was correct
            else:
                label = 0  # Should have been SELL

            features_list.append(features_array)
            labels_list.append(label)

        except Exception as e:
            print(f"⚠️ Skipping trade {trade['symbol']}: {e}")
            continue

    if len(features_list) < 10:
        print(f"⚠️ Not enough valid training data (need at least 10, have {len(features_list)})")
        print("Make sure trades have feature data attached.")
        return

    # Convert to numpy arrays
    X = np.vstack(features_list)
    y = np.array(labels_list)

    print(f"Training data: {len(X)} samples")
    print(f"Wins: {np.sum(y == 2)}, Losses: {np.sum(y == 0)}")

    # Train model
    print("\n🎓 Training model...")
    metrics = model.train(X, y)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training Accuracy:   {metrics['train_accuracy']:.2%}")
    print(f"Validation Accuracy: {metrics['val_accuracy']:.2%}")
    print(f"Training Samples:    {metrics['train_samples']}")
    print(f"Validation Samples:  {metrics['val_samples']}")
    print("=" * 60 + "\n")

    print("✅ Model saved! Future scans will use the improved model.")
    print("Run a new scan to see better predictions!")


if __name__ == '__main__':
    try:
        retrain_from_trades()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
