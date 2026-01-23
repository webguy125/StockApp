
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Full Production Training - Fast Mode (Phase 1.5 - Dual-Threshold)
Train all 11 sectors for multiple horizons (1D, 2D, 5D) with BOTH 5% and 10% thresholds

PHASE 1.5 ENHANCEMENT:
- Trains TWO complete model universes: 5% threshold and 10% threshold
- Each universe has separate labels, datasets, and trained models
- NO cross-contamination between thresholds
- Preserves all Phase 1 logic and model architectures

Expected Runtime:
- 11 sectors × 3 horizons × 2 thresholds × ~3 min = ~180-200 minutes total
- 5% models saved to: backend/turbomode/models/trained_5pct/
- 10% models saved to: backend/turbomode/models/trained_10pct/
"""

import sys
import os
import time
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.turbomode.core_engine.turbomode_training_loader import TurboModeTrainingDataLoader
from backend.turbomode.core_engine.training_symbols import TRAINING_SYMBOLS
from backend.turbomode.core_engine.train_turbomode_models_fastmode import train_single_sector_worker_fastmode

# Training configuration
HORIZONS = [1, 2, 5]  # 1-day, 2-day, 5-day prediction horizons

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

# All 11 sectors
ALL_SECTORS = [
    'technology',
    'financials',
    'healthcare',
    'consumer_discretionary',
    'communication_services',
    'industrials',
    'consumer_staples',
    'energy',
    'materials',
    'real_estate',
    'utilities'
]

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


def train_all_sectors_all_horizons():
    """
    PHASE 1.5 - Dual-Threshold Production Training:
    - 2 thresholds (5%, 10%)
    - 11 sectors per threshold
    - 3 horizons (1D, 2D, 5D) per sector
    - 66 total training runs (33 per threshold)
    """
    print("\n" + "=" * 80)
    print("PHASE 1.5 - DUAL-THRESHOLD PRODUCTION TRAINING")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Thresholds: {len(THRESHOLDS)}")
    print(f"Sectors: {len(ALL_SECTORS)}")
    print(f"Horizons: {HORIZONS}")
    print(f"Total Runs: {len(THRESHOLDS) * len(ALL_SECTORS) * len(HORIZONS)}")
    print("=" * 80)

    global_start = time.time()
    loader = TurboModeTrainingDataLoader()

    all_results = {}

    # PHASE 1.5: Loop over thresholds
    for threshold_name, threshold_config in THRESHOLDS.items():
        threshold_value = threshold_config["value"]
        save_dir_name = threshold_config["save_dir"]

        print(f"\n{'=' * 80}")
        print(f"THRESHOLD: {threshold_name.upper()} ({threshold_value*100:.0f}%)")
        print(f"Save Directory: backend/turbomode/models/{save_dir_name}/")
        print(f"{'=' * 80}")

        threshold_results = {}

        for horizon_days in HORIZONS:
            print(f"\n{'=' * 80}")
            print(f"THRESHOLD: {threshold_name.upper()} | HORIZON: {horizon_days}D")
            print(f"{'=' * 80}")

            horizon_results = {}

            for sector in ALL_SECTORS:
                try:
                    # Get symbols for sector
                    sector_symbols = get_symbols_by_sector(sector)
                    print(f"\n[{threshold_name.upper()}] [{sector.upper()}] {len(sector_symbols)} symbols")

                    # PHASE 1.5: Define threshold-specific labels
                    sector_thresholds = {
                        "buy": threshold_value,
                        "sell": -threshold_value
                    }

                    # Load training data with threshold-specific labels
                    print(f"[{threshold_name.upper()}] [{sector.upper()}] Loading data for {horizon_days}d horizon (threshold: {threshold_value*100:.0f}%)...")
                    X_train, y_train, X_val, y_val = loader.load_training_data(
                        symbols_filter=sector_symbols,
                        return_split=True,
                        test_size=0.2,
                        horizon_days=horizon_days,
                        thresholds=sector_thresholds
                    )

                    print(f"[{threshold_name.upper()}] [{sector.upper()}] Train: {len(X_train):,}, Val: {len(X_val):,}")

                    # PHASE 1.5: Compute threshold-specific save directory
                    base_save_dir = os.path.join(project_root, 'backend', 'turbomode', 'models', save_dir_name)

                    # Train sector with Fast Mode (threshold-specific save path)
                    result = train_single_sector_worker_fastmode(
                        sector=sector,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        horizon_days=horizon_days,
                        save_models=True,
                        save_dir=base_save_dir  # PHASE 1.5: Threshold-specific directory
                    )

                    horizon_results[sector] = result

                    print(f"\n[{threshold_name.upper()}] [{sector.upper()}] COMPLETE")
                    print(f"  Meta Accuracy: {result['meta_accuracy']:.4f}")
                    print(f"  Training Time: {result['total_time']/60:.1f} min")
                    print(f"  Saved to: {base_save_dir}/{sector}/{horizon_days}d/")

                except Exception as e:
                    print(f"\n[ERROR] [{threshold_name.upper()}] {sector.upper()} - {horizon_days}d failed: {e}")
                    horizon_results[sector] = {
                        'status': 'failed',
                        'error': str(e)
                    }

            threshold_results[f'{horizon_days}d'] = horizon_results

        all_results[threshold_name] = threshold_results

    # Summary
    global_time = time.time() - global_start

    print("\n" + "=" * 80)
    print("PHASE 1.5 - DUAL-THRESHOLD TRAINING COMPLETE")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {global_time/3600:.1f} hours ({global_time/60:.1f} minutes)")
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for threshold_name, threshold_results in all_results.items():
        print(f"\n{'=' * 80}")
        print(f"THRESHOLD: {threshold_name.upper()}")
        print(f"{'=' * 80}")

        for horizon, horizon_results in threshold_results.items():
            print(f"\n{horizon.upper()} HORIZON:")
            print("-" * 80)

            successful = 0
            failed = 0
            total_accuracy = 0

            for sector, result in horizon_results.items():
                if result.get('status') == 'completed':
                    successful += 1
                    acc = result['meta_accuracy']
                    total_accuracy += acc
                    time_min = result['total_time'] / 60
                    print(f"  {sector:25s} ✓ Accuracy: {acc:.4f}  Time: {time_min:.1f} min")
                else:
                    failed += 1
                    print(f"  {sector:25s} ✗ FAILED: {result.get('error', 'Unknown error')}")

            if successful > 0:
                avg_accuracy = total_accuracy / successful
                print(f"\n  Successful: {successful}/{len(horizon_results)}")
                print(f"  Average Accuracy: {avg_accuracy:.4f}")
            if failed > 0:
                print(f"  Failed: {failed}/{len(horizon_results)}")

    print("\n" + "=" * 80)
    print("MODEL DIRECTORIES:")
    print("=" * 80)
    print(f"  5% models:  C:\\StockApp\\backend\\turbomode\\models\\trained_5pct\\")
    print(f"  10% models: C:\\StockApp\\backend\\turbomode\\models\\trained_10pct\\")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    results = train_all_sectors_all_horizons()
