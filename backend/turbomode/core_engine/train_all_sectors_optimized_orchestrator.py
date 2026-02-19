
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Ensemble Training Orchestrator
Trains 6 models per sector: 5 base models + 1 MetaLearner (66 models total)

ARCHITECTURE:
- Single label: 14-day MFE/MAE path-dependent labels (±6% threshold)
- 6 models per sector: 11 sectors × 6 models = 66 models total
- 5 base models: LightGBM-GPU, CatBoost-GPU, XGBoost-Hist-GPU, XGBoost-Linear, RandomForest
- 1 MetaLearner: LogisticRegression (trained on stacked out-of-fold predictions)
- Directory structure: models/trained/<sector>/<model_name>.pkl

PERFORMANCE:
- Training time: ~4-5 hours for all 11 sectors (14-day labels)
- Fast ensemble architecture with GPU acceleration

Author: TurboMode Optimization Team
Date: 2026-01-21
Updated: 2026-02-16 (14-day enforcement)
"""

# TURBOMODE 14-DAY HORIZON ENFORCEMENT
TURBOMODE_HORIZON = '14d'
HORIZON_DAYS = 14
BUY_THRESHOLD = 0.06   # +6% over 14 days
SELL_THRESHOLD = -0.06  # -6% over 14 days

import os
import time
from datetime import datetime
import numpy as np
import multiprocessing

# Enable spawn method for safe multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from backend.turbomode.core_engine.sector_batch_trainer import run_sector_training
from backend.turbomode.core_engine.training_symbols import TRAINING_SYMBOLS
from backend.turbomode.core_engine.train_sector_models import train_sector_ensemble


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


def train_single_sector(args):
    """
    Train a single sector in isolation (used for multiprocessing).

    Args:
        args: Tuple of (sector_name, preloaded_data_tuple)

    Returns:
        Dictionary with training results for the sector
    """
    sector, preloaded = args
    X_sector, labels_dict, trade_ids = preloaded

    sector_start = time.time()
    print(f"[WORKER {os.getpid()}] Starting sector: {sector}")
    print(f"[WORKER {os.getpid()}] Using preloaded data for {sector}...")

    try:
        print(f"[WORKER {os.getpid()}] Preloaded data verified for {sector} | X={len(X_sector)} samples")

        if len(X_sector) == 0:
            return {'sector': sector, 'status': 'failed', 'error': 'No training data', 'total_time': 0}

        # Build label vector
        y_sector = np.array([labels_dict[tid] for tid in trade_ids], dtype=np.int32)

        print(f"[WORKER {os.getpid()}] Training ensemble for {sector}...")

        # Train ensemble
        ensemble_paths = train_sector_ensemble(sector, X_sector, y_sector)

        total_time = time.time() - sector_start
        print(f"[WORKER {os.getpid()}] Completed sector: {sector} | {len(ensemble_paths)} models saved | {total_time:.1f}s")

        return {
            'sector': sector,
            'status': 'completed',
            'n_models': 6,
            'model_paths': ensemble_paths,
            'n_samples': X_sector.shape[0],
            'total_time': total_time
        }

    except Exception as e:
        total_time = time.time() - sector_start
        return {'sector': sector, 'status': 'failed', 'error': str(e), 'total_time': total_time}


def train_all_sectors_optimized():
    """
    ENSEMBLE TRAINING ORCHESTRATOR

    Architecture:
        for each sector:
            load data ONCE → train 5 base models + 1 MetaLearner → save 6 models

    Performance:
        - 11 sectors × 6 models each = 66 models total
        - Training time: ~4-5 hours
        - Single label: 14-day MFE/MAE path-dependent (±6% threshold)
    """
    # ENFORCE 14-DAY HORIZON
    assert TURBOMODE_HORIZON == '14d', 'TurboMode must use 14d horizon only'

    # EXPLICIT START LOGGING
    start_time = datetime.now()
    global_start = time.time()
    print(f"[TRAIN] TurboMode training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "=" * 80)
    print("ENSEMBLE TRAINING (5 BASE MODELS + META-LEARNER)")
    print("=" * 80)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sectors: {len(ALL_SECTORS)}")
    print(f"Models per sector: 6 (5 base + 1 meta-learner)")
    print(f"Label: 14-day MFE/MAE path-dependent (±6% threshold)")
    print(f"Horizon: {TURBOMODE_HORIZON} ({HORIZON_DAYS} days - ENFORCED)")
    print(f"Thresholds: BUY >= +{BUY_THRESHOLD*100:.0f}%, SELL <= {SELL_THRESHOLD*100:.0f}%")
    print(f"Total models: {len(ALL_SECTORS) * 6}")
    print("=" * 80)
    print()

    # Database and save directory
    backend_dir = str(project_root / "backend")
    db_path = os.path.join(backend_dir, "data", "turbomode.db")
    save_dir = os.path.join(backend_dir, "turbomode", "models", "trained")

    all_results = {}

    # PRELOAD: Load all sector data in main process before multiprocessing
    print(f"[OPTIMIZE] Preloading all sector data in main process...")
    from backend.turbomode.core_engine.sector_batch_trainer import load_sector_data_once

    preloaded_data = {}
    for sector in ALL_SECTORS:
        sector_symbols = get_symbols_by_sector(sector)
        print(f"[PRELOAD] Loading {sector}...")
        preloaded_data[sector] = load_sector_data_once(
            db_path,
            sector_symbols,
            horizon_days=HORIZON_DAYS,
            buy_threshold=BUY_THRESHOLD,
            sell_threshold=SELL_THRESHOLD
        )
    print(f"[OPTIMIZE] Preloading complete - {len(preloaded_data)} sectors loaded")
    print()

    # MULTIPROCESSING: Train sectors in parallel
    print(f"[OPTIMIZE] Sector parallelization enabled - training {len(ALL_SECTORS)} sectors in parallel")
    print(f"[OPTIMIZE] Using multiprocessing pool with {min(len(ALL_SECTORS), 4)} processes")
    print()

    with multiprocessing.Pool(processes=min(len(ALL_SECTORS), 4)) as pool:
        async_result = pool.map_async(train_single_sector, [(sector, preloaded_data[sector]) for sector in ALL_SECTORS])

        # GLOBAL PROGRESS TICKER
        print("[STATUS] Workers running... monitoring progress...")

        while not async_result.ready():
            elapsed = time.time() - global_start
            print(f"[STATUS] Elapsed: {elapsed/60:.1f} min | Training in progress... | {min(len(ALL_SECTORS), 4)} workers active")
            async_result.wait(30)  # Wait up to 30 seconds

        sector_results = async_result.get()
        print(f"[STATUS] All sectors completed!")

    # Convert list of results to dictionary
    for result in sector_results:
        sector = result['sector']
        all_results[sector] = result

    # Final summary
    total_time = time.time() - global_start
    end_time = datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()

    # EXPLICIT COMPLETION LOGGING
    print(f"[TRAIN] TurboMode training completed successfully at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, duration: {duration_seconds:.1f} seconds ({duration_seconds/3600:.2f} hours)")

    print("\n" + "=" * 80)
    print("SINGLE-MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print()

    # Results summary
    successful = sum(1 for r in all_results.values() if r.get('status') == 'completed')
    failed = sum(1 for r in all_results.values() if r.get('status') == 'failed')

    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Sectors processed: {len(ALL_SECTORS)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print()

    print("Per-Sector Results:")
    print("-" * 80)
    for sector in ALL_SECTORS:
        result = all_results.get(sector, {})
        status = result.get('status', 'unknown')

        if status == 'completed':
            time_min = result['total_time'] / 60
            print(f"  {sector:30s} [OK] Completed in {time_min:.1f} min")
        else:
            error = result.get('error', 'Unknown error')
            print(f"  {sector:30s} [FAILED]: {error}")

    print()
    print("=" * 80)
    print("MODEL DIRECTORY:")
    print("=" * 80)
    print(f"  Location: {save_dir}")
    print(f"  Structure: models/trained/<sector>/<model_name>.pkl")
    print(f"  Models per sector: 6 (5 base + 1 meta-learner)")
    print(f"  Total models: {len(ALL_SECTORS) * 6} (66 models)")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    import sys

    # SMOKE TEST MODE: python train_all_sectors_optimized_orchestrator.py --smoke-test [sector]
    if len(sys.argv) > 1 and sys.argv[1] == '--smoke-test':
        # Determine which sector to test
        test_sector = sys.argv[2] if len(sys.argv) > 2 else 'technology'

        print(f"[SMOKE TEST MODE] Training SINGLE sector ({test_sector}) for validation")

        # Temporarily override ALL_SECTORS for smoke test
        original_sectors = ALL_SECTORS.copy()
        ALL_SECTORS.clear()
        ALL_SECTORS.append(test_sector)

        results = train_all_sectors_optimized()

        # Restore original sectors
        ALL_SECTORS.clear()
        ALL_SECTORS.extend(original_sectors)

        print("\n[SMOKE TEST COMPLETE]")
    else:
        # FULL PRODUCTION RUN
        results = train_all_sectors_optimized()
