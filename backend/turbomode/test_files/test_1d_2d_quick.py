"""
Quick test of 1D/2D training with just 2 symbols (AAPL, JNJ)
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader

# Test symbols: AAPL (technology), JNJ (healthcare)
TEST_SYMBOLS = ['AAPL', 'JNJ']

SECTOR_THRESHOLDS_V1 = {
    "buy": 0.10,
    "sell": -0.10
}

if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUICK TEST: 1D/2D TRAINING WITH 2 SYMBOLS")
    print("="*80)
    print(f"Symbols: {TEST_SYMBOLS}")
    print()

    loader = TurboModeTrainingDataLoader()

    # Test 1-day horizon
    print("\n" + "="*80)
    print("TEST 1: LOADING DATA FOR 1-DAY HORIZON")
    print("="*80)

    try:
        X, y = loader.load_training_data(
            return_split=False,
            symbols_filter=TEST_SYMBOLS,
            horizon_days=1,
            thresholds=SECTOR_THRESHOLDS_V1
        )

        print(f"\n[SUCCESS] 1-day horizon:")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Label distribution:")
        import numpy as np
        for label in [0, 1, 2]:
            count = np.sum(y == label)
            label_name = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[label]
            print(f"    {label_name}: {count:,} ({count/len(y)*100:.1f}%)")

    except Exception as e:
        print(f"[ERROR] 1-day horizon failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2-day horizon
    print("\n" + "="*80)
    print("TEST 2: LOADING DATA FOR 2-DAY HORIZON")
    print("="*80)

    try:
        X, y = loader.load_training_data(
            return_split=False,
            symbols_filter=TEST_SYMBOLS,
            horizon_days=2,
            thresholds=SECTOR_THRESHOLDS_V1
        )

        print(f"\n[SUCCESS] 2-day horizon:")
        print(f"  Samples: {len(X):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Label distribution:")
        for label in [0, 1, 2]:
            count = np.sum(y == label)
            label_name = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[label]
            print(f"    {label_name}: {count:,} ({count/len(y)*100:.1f}%)")

    except Exception as e:
        print(f"[ERROR] 2-day horizon failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("QUICK TEST COMPLETE")
    print("="*80)
