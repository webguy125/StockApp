
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Ensemble Training Orchestrator
Trains 6 models per sector: 5 base models + 1 MetaLearner (66 models total)

ARCHITECTURE:
- Single label: label_1d_5pct (1-day horizon, 5% threshold)
- 6 models per sector: 11 sectors × 6 models = 66 models total
- 5 base models: LightGBM-GPU, CatBoost-GPU, XGBoost-Hist-GPU, XGBoost-Linear, RandomForest
- 1 MetaLearner: LogisticRegression (trained on stacked out-of-fold predictions)
- Directory structure: models/trained/<sector>/<model_name>.pkl

PERFORMANCE:
- Training time: ~60-90 minutes for all 11 sectors
- Fast ensemble architecture with GPU acceleration

Author: TurboMode Optimization Team
Date: 2026-01-21
"""

import os
import time
from datetime import datetime
import numpy as np

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


def train_all_sectors_optimized():
    """
    ENSEMBLE TRAINING ORCHESTRATOR

    Architecture:
        for each sector:
            load data ONCE → train 5 base models + 1 MetaLearner → save 6 models

    Performance:
        - 11 sectors × 6 models each = 66 models total
        - Training time: ~60-90 minutes
        - Single label: label_1d_5pct (1-day horizon, 5% threshold)
    """
    print("\n" + "=" * 80)
    print("ENSEMBLE TRAINING (5 BASE MODELS + META-LEARNER)")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sectors: {len(ALL_SECTORS)}")
    print(f"Models per sector: 6 (5 base + 1 meta-learner)")
    print(f"Label: label_1d_5pct (1-day horizon, 5% threshold)")
    print(f"Total models: {len(ALL_SECTORS) * 6}")
    print("=" * 80)
    print()

    global_start = time.time()

    # Database and save directory
    backend_dir = str(project_root / "backend")
    db_path = os.path.join(backend_dir, "data", "turbomode.db")
    save_dir = os.path.join(backend_dir, "turbomode", "models", "trained")

    all_results = {}

    # SECTOR LOOP (trains 6 models per sector)
    for i, sector in enumerate(ALL_SECTORS, 1):
        sector_symbols = get_symbols_by_sector(sector)

        print("\n" + "=" * 80)
        print(f"[{i}/{len(ALL_SECTORS)}] SECTOR: {sector.upper()}")
        print(f"Symbols: {len(sector_symbols)}")
        print("=" * 80)
        print()

        sector_start = time.time()

        try:
            # Load sector data using existing infrastructure
            from backend.turbomode.core_engine.sector_batch_trainer import load_sector_data_once

            print("Loading sector data...")

            # Load all sector data (features + labels)
            X_sector, labels_dict, trade_ids = load_sector_data_once(db_path, sector_symbols)

            if len(X_sector) == 0:
                raise ValueError(f"No training data available for {sector}")

            # Build label vector
            y_sector = np.array([labels_dict[tid] for tid in trade_ids], dtype=np.int32)

            print(f"Training data ready: X shape={X_sector.shape}, y shape={y_sector.shape}")
            print(f"Label distribution: BUY={np.sum(y_sector==2)}, SELL={np.sum(y_sector==0)}, HOLD={np.sum(y_sector==1)}")

            # Train ensemble (5 base models + MetaLearner)
            print("Training ensemble models...")
            ensemble_paths = train_sector_ensemble(sector, X_sector, y_sector)

            sector_time = time.time() - sector_start

            all_results[sector] = {
                'status': 'completed',
                'total_time': sector_time,
                'n_models': 6,
                'model_paths': ensemble_paths,
                'n_samples': X_sector.shape[0]
            }

            print(f"\n[{i}/{len(ALL_SECTORS)}] {sector.upper()} COMPLETE [OK]")
            print(f"Time: {sector_time/60:.1f} minutes")
            print(f"Models saved: {len(ensemble_paths)}")

        except Exception as e:
            import traceback
            print(f"\n[{i}/{len(ALL_SECTORS)}] {sector.upper()} FAILED [X]")
            print(f"Exception: {e}")
            traceback.print_exc()
            all_results[sector] = {'status': 'failed', 'error': str(e)}

        # Progress update
        elapsed = time.time() - global_start
        sectors_done = i
        sectors_remaining = len(ALL_SECTORS) - i
        avg_time_per_sector = elapsed / sectors_done
        estimated_remaining = avg_time_per_sector * sectors_remaining

        print(f"\n[PROGRESS] {sectors_done}/{len(ALL_SECTORS)} sectors complete")
        print(f"[PROGRESS] Elapsed: {elapsed/60:.1f} min | Estimated remaining: {estimated_remaining/60:.1f} min")
        print()

    # Final summary
    total_time = time.time() - global_start

    print("\n" + "=" * 80)
    print("SINGLE-MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    results = train_all_sectors_optimized()
