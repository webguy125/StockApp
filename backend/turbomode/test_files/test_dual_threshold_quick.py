"""
Quick Test: Phase 1.5 Dual-Threshold Training
Tests both 5% and 10% threshold training for a single sector (Technology, 1D horizon only)

Expected runtime: ~5-6 minutes (2-3 min per threshold)
"""

import sys
import os
import time
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader
from backend.turbomode.training_symbols import TRAINING_SYMBOLS
from backend.turbomode.train_turbomode_models_fastmode import train_single_sector_worker_fastmode

# Test configuration
TEST_SECTOR = 'technology'
TEST_HORIZON = 1  # 1-day only

# PHASE 1.5: Dual-threshold configuration
THRESHOLDS = {
    "5pct": {
        "value": 0.05,
        "save_dir": "trained_5pct"
    },
    "10pct": {
        "value": 0.10,
        "save_dir": "trained_10pct"
    }
}

def get_symbols_by_sector(sector: str):
    """Get all symbols for a sector."""
    if sector not in TRAINING_SYMBOLS:
        raise ValueError(f"Unknown sector: {sector}")

    sector_data = TRAINING_SYMBOLS[sector]
    symbols = []

    if 'large_cap' in sector_data:
        symbols.extend(sector_data['large_cap'])
    if 'mid_cap' in sector_data:
        symbols.extend(sector_data['mid_cap'])
    if 'small_cap' in sector_data:
        symbols.extend(sector_data['small_cap'])

    return symbols


def test_dual_threshold_training():
    """
    Test dual-threshold training for Technology sector (1D horizon only)
    """
    print("\n" + "=" * 80)
    print("PHASE 1.5 - DUAL-THRESHOLD QUICK TEST")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Sector: {TEST_SECTOR}")
    print(f"Test Horizon: {TEST_HORIZON}D")
    print(f"Thresholds: 5% and 10%")
    print("=" * 80)

    global_start = time.time()
    loader = TurboModeTrainingDataLoader()

    all_results = {}

    # Get symbols for test sector
    sector_symbols = get_symbols_by_sector(TEST_SECTOR)
    print(f"\n[{TEST_SECTOR.upper()}] {len(sector_symbols)} symbols")

    # PHASE 1.5: Loop over thresholds
    for threshold_name, threshold_config in THRESHOLDS.items():
        threshold_value = threshold_config["value"]
        save_dir_name = threshold_config["save_dir"]

        print(f"\n{'=' * 80}")
        print(f"THRESHOLD: {threshold_name.upper()} ({threshold_value*100:.0f}%)")
        print(f"Save Directory: backend/turbomode/models/{save_dir_name}/")
        print(f"{'=' * 80}")

        try:
            # PHASE 1.5: Define threshold-specific labels
            sector_thresholds = {
                "buy": threshold_value,
                "sell": -threshold_value
            }

            # Load training data with threshold-specific labels
            print(f"[{threshold_name.upper()}] Loading data (threshold: {threshold_value*100:.0f}%)...")
            X_train, y_train, X_val, y_val = loader.load_training_data(
                symbols_filter=sector_symbols,
                return_split=True,
                test_size=0.2,
                horizon_days=TEST_HORIZON,
                thresholds=sector_thresholds
            )

            print(f"[{threshold_name.upper()}] Train: {len(X_train):,}, Val: {len(X_val):,}")
            print(f"[{threshold_name.upper()}] Label distribution:")
            import numpy as np
            unique, counts = np.unique(y_train, return_counts=True)
            for label, count in zip(unique, counts):
                label_name = ['SELL', 'HOLD', 'BUY'][label]
                pct = count / len(y_train) * 100
                print(f"  {label_name}: {count:,} ({pct:.1f}%)")

            # PHASE 1.5: Compute threshold-specific save directory
            base_save_dir = os.path.join(project_root, 'backend', 'turbomode', 'models', save_dir_name)

            # Train sector with Fast Mode (threshold-specific save path)
            result = train_single_sector_worker_fastmode(
                sector=TEST_SECTOR,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                horizon_days=TEST_HORIZON,
                save_models=True,
                save_dir=base_save_dir  # PHASE 1.5: Threshold-specific directory
            )

            all_results[threshold_name] = result

            print(f"\n[{threshold_name.upper()}] COMPLETE")
            print(f"  Meta Accuracy: {result['meta_accuracy']:.4f}")
            print(f"  Training Time: {result['total_time']/60:.1f} min")
            print(f"  Saved to: {base_save_dir}/{TEST_SECTOR}/{TEST_HORIZON}d/")

        except Exception as e:
            print(f"\n[ERROR] [{threshold_name.upper()}] Training failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[threshold_name] = {
                'status': 'failed',
                'error': str(e)
            }

    # Summary
    global_time = time.time() - global_start

    print("\n" + "=" * 80)
    print("PHASE 1.5 - DUAL-THRESHOLD QUICK TEST COMPLETE")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {global_time/60:.1f} minutes")
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    for threshold_name, result in all_results.items():
        if result.get('status') == 'failed':
            print(f"  {threshold_name.upper():6s}: FAILED - {result.get('error')}")
        else:
            acc = result['meta_accuracy']
            time_min = result['total_time'] / 60
            print(f"  {threshold_name.upper():6s}: Accuracy: {acc:.4f}  Time: {time_min:.1f} min")

    print("\n" + "=" * 80)
    print("MODEL DIRECTORIES:")
    print("=" * 80)
    print(f"  5% models:  C:\\StockApp\\backend\\turbomode\\models\\trained_5pct\\{TEST_SECTOR}\\1d\\")
    print(f"  10% models: C:\\StockApp\\backend\\turbomode\\models\\trained_10pct\\{TEST_SECTOR}\\1d\\")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = test_dual_threshold_training()
