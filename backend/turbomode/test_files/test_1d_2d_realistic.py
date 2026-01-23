"""
Quick test with REALISTIC thresholds (3% instead of 10%)
"""
import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader

# Test symbols: AAPL (technology), JNJ (healthcare)
TEST_SYMBOLS = ['AAPL', 'JNJ']

# More realistic thresholds for 1-2 day moves
REALISTIC_THRESHOLDS = {
    "buy": 0.03,   # 3% gain
    "sell": -0.03  # 3% loss
}

if __name__ == "__main__":
    print("\n" + "="*80)
    print("REALISTIC THRESHOLD TEST: 1D/2D WITH 3% THRESHOLDS")
    print("="*80)
    print(f"Symbols: {TEST_SYMBOLS}")
    print(f"Thresholds: BUY >= {REALISTIC_THRESHOLDS['buy']:.1%}, SELL <= {REALISTIC_THRESHOLDS['sell']:.1%}")
    print()

    loader = TurboModeTrainingDataLoader()

    # Test 1-day horizon
    print("\n" + "="*80)
    print("1-DAY HORIZON (3% thresholds)")
    print("="*80)

    X, y = loader.load_training_data(
        return_split=False,
        symbols_filter=TEST_SYMBOLS,
        horizon_days=1,
        thresholds=REALISTIC_THRESHOLDS
    )

    print(f"\n[SUCCESS] 1-day horizon:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Label distribution:")
    for label in [0, 1, 2]:
        count = np.sum(y == label)
        label_name = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[label]
        print(f"    {label_name}: {count:,} ({count/len(y)*100:.1f}%)")

    # Test 2-day horizon
    print("\n" + "="*80)
    print("2-DAY HORIZON (3% thresholds)")
    print("="*80)

    X, y = loader.load_training_data(
        return_split=False,
        symbols_filter=TEST_SYMBOLS,
        horizon_days=2,
        thresholds=REALISTIC_THRESHOLDS
    )

    print(f"\n[SUCCESS] 2-day horizon:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Label distribution:")
    for label in [0, 1, 2]:
        count = np.sum(y == label)
        label_name = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[label]
        print(f"    {label_name}: {count:,} ({count/len(y)*100:.1f}%)")

    print("\n" + "="*80)
    print("TEST COMPLETE - DATA LOADING WORKS!")
    print("="*80)
    print("\nReady to test full GPU training on 2 symbols...")
