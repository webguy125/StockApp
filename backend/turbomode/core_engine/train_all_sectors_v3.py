
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
TurboMode v3.1.0 Training Orchestrator

MODES:
1. python train_all_sectors_v3.py       → Train all sectors with pruned features
2. python train_all_sectors_v3.py prune → Generate new pruned feature lists and exit

ARCHITECTURE:
- Single LightGBM model per sector (11 sectors = 11 models)
- 14-day horizon, ±6% thresholds
- 187 canonical features (with auto-pruning to 40-80 per sector)
- Real options data from options_universe.db (802K rows)
- SHAP logging, waterfall plots, cross-sector aggregation

NO ENVIRONMENT VARIABLES - All features enabled by default

Author: TurboMode Core Engine
Version: 3.1.0
Date: 2026-02-18
"""

# TURBOMODE CANONICAL CONFIGURATION
TURBOMODE_VERSION = '3.1.0'
HORIZON_DAYS = 14
BUY_THRESHOLD = 0.06   # +6% over 14 days
SELL_THRESHOLD = -0.06  # -6% over 14 days

# V3.1.0 FEATURES (Always enabled, no environment variables)
ENABLE_SHAP_LOGGING = True
ENABLE_ADAPTIVE_PRUNING = True
ENABLE_SHAP_WATERFALL = True
ENABLE_ADVANCED_SIGNALS = True
ENABLE_FEATURE_PRUNING = True  # Load existing pruned features in train mode

# SHAP Configuration
SHAP_SAMPLE_SIZE = 2000
SHAP_RANDOM_STATE = 42
SHAP_WATERFALL_MAX_PLOTS = 5

# Adaptive Pruning Configuration
ADAPTIVE_PRUNING_MIN = 40
ADAPTIVE_PRUNING_MAX = 80
ADAPTIVE_PRUNING_THRESHOLD = 0.92

import os
import time
from datetime import datetime
import multiprocessing

# Set environment variables for downstream modules
os.environ['ENABLE_SHAP_LOGGING'] = 'true'
os.environ['ENABLE_ADAPTIVE_PRUNING'] = 'true'
os.environ['ENABLE_SHAP_WATERFALL'] = 'true'
os.environ['ENABLE_ADVANCED_SIGNALS'] = 'true'
os.environ['ENABLE_FEATURE_PRUNING'] = 'true'
os.environ['SHAP_SAMPLE_SIZE'] = str(SHAP_SAMPLE_SIZE)
os.environ['SHAP_RANDOM_STATE'] = str(SHAP_RANDOM_STATE)
os.environ['SHAP_WATERFALL_MAX_PLOTS'] = str(SHAP_WATERFALL_MAX_PLOTS)
os.environ['ADAPTIVE_PRUNING_MIN'] = str(ADAPTIVE_PRUNING_MIN)
os.environ['ADAPTIVE_PRUNING_MAX'] = str(ADAPTIVE_PRUNING_MAX)
os.environ['ADAPTIVE_PRUNING_THRESHOLD'] = str(ADAPTIVE_PRUNING_THRESHOLD)

from backend.turbomode.core_engine.sector_batch_trainer import load_sector_data_once, run_sector_training
from backend.turbomode.core_engine.training_symbols import TRAINING_SYMBOLS

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


def train_mode():
    """
    TRAIN MODE: Train all 11 sectors using existing pruned feature lists.

    Behavior:
    - Load canonical 187 features
    - Load pruned feature lists (if available)
    - Apply pruned features during training
    - Use real options data (802K rows)
    - Generate SHAP logs and waterfall plots
    - Train all 11 sector models
    """
    start_time = datetime.now()
    global_start = time.time()

    print("\n" + "=" * 80)
    print(f"TURBOMODE v{TURBOMODE_VERSION} - TRAIN MODE")
    print("=" * 80)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: TRAIN (using existing pruned features)")
    print(f"Sectors: {len(ALL_SECTORS)}")
    print(f"Models: 11 (1 LightGBM per sector)")
    print(f"Label: 14-day MFE/MAE path-dependent (±{BUY_THRESHOLD*100:.0f}% threshold)")
    print(f"Horizon: {HORIZON_DAYS} days (ENFORCED)")
    print()
    print("V3.1.0 Features (always enabled):")
    print(f"  - SHAP Logging: {ENABLE_SHAP_LOGGING}")
    print(f"  - Adaptive Pruning: {ENABLE_ADAPTIVE_PRUNING}")
    print(f"  - SHAP Waterfall: {ENABLE_SHAP_WATERFALL}")
    print(f"  - Advanced Signals: {ENABLE_ADVANCED_SIGNALS}")
    print(f"  - Feature Pruning: {ENABLE_FEATURE_PRUNING}")
    print(f"  - Real Options Data: options_universe.db (802K rows)")
    print("=" * 80)
    print()

    # Database and save directory
    backend_dir = str(project_root / "backend")
    db_path = os.path.join(backend_dir, "data", "turbomode.db")
    save_dir = os.path.join(backend_dir, "turbomode", "models", "trained")

    # Train all sectors sequentially
    all_results = {}
    for sector in ALL_SECTORS:
        sector_symbols = get_symbols_by_sector(sector)

        result = run_sector_training(
            sector_name=sector,
            sector_symbols=sector_symbols,
            db_path=db_path,
            save_dir=save_dir
        )

        all_results[sector] = result

    # Final summary
    total_time = time.time() - global_start
    end_time = datetime.now()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Sectors Trained: {len([r for r in all_results.values() if r.get('status') == 'completed'])}/{len(ALL_SECTORS)}")
    print()

    # Run cross-sector SHAP aggregation
    if ENABLE_SHAP_LOGGING:
        print("=" * 80)
        print("RUNNING CROSS-SECTOR SHAP AGGREGATION")
        print("=" * 80)
        try:
            from backend.turbomode.core_engine.shap_analysis import run_cross_sector_analysis
            run_cross_sector_analysis(save_dir, ALL_SECTORS)
        except Exception as e:
            print(f"[WARN] Cross-sector aggregation failed: {e}")
        print()

    print(f"Models saved to: {save_dir}")
    print("=" * 80)


def prune_mode():
    """
    PRUNE MODE: Generate new sector-specific pruned feature lists and exit.

    Behavior:
    - Load canonical 187 features (NO pruning applied)
    - Train models on full feature set
    - Generate SHAP summaries
    - Compute adaptive pruned feature lists per sector
    - Save pruned_features.json for each sector
    - EXIT without keeping the models
    """
    start_time = datetime.now()
    global_start = time.time()

    print("\n" + "=" * 80)
    print(f"TURBOMODE v{TURBOMODE_VERSION} - PRUNE MODE")
    print("=" * 80)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: PRUNE (generate new pruned feature lists)")
    print(f"Sectors: {len(ALL_SECTORS)}")
    print(f"Features: 187 (full canonical set)")
    print()
    print("This will:")
    print("  1. Train models on FULL 187 features")
    print("  2. Generate SHAP summaries per sector")
    print("  3. Compute adaptive pruned feature lists (40-80 features)")
    print("  4. Save pruned_features.json for each sector")
    print("  5. EXIT (models will be discarded)")
    print("=" * 80)
    print()

    # Temporarily disable feature pruning load (train on full set)
    os.environ['ENABLE_FEATURE_PRUNING'] = 'false'

    # Database and save directory
    backend_dir = str(project_root / "backend")
    db_path = os.path.join(backend_dir, "data", "turbomode.db")
    save_dir = os.path.join(backend_dir, "turbomode", "models", "trained")

    # Train all sectors to generate pruned lists
    for sector in ALL_SECTORS:
        sector_symbols = get_symbols_by_sector(sector)

        print(f"\n[PRUNE] Processing sector: {sector}")
        result = run_sector_training(
            sector_name=sector,
            sector_symbols=sector_symbols,
            db_path=db_path,
            save_dir=save_dir
        )

        if result.get('status') == 'completed':
            print(f"[PRUNE] {sector}: Pruned features saved")
        else:
            print(f"[PRUNE] {sector}: FAILED - {result.get('error', 'Unknown error')}")

    # Final summary
    total_time = time.time() - global_start
    end_time = datetime.now()

    print("\n" + "=" * 80)
    print("PRUNING COMPLETE")
    print("=" * 80)
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print()
    print("Pruned feature lists saved to:")
    print(f"  {save_dir}/{{sector}}/lightgbm/pruned_features.json")
    print()
    print("Next step: Run training mode to use these pruned features:")
    print(f"  python {__file__}")
    print("=" * 80)


def main():
    """Main entry point with mode detection."""
    import sys

    # Detect mode from command line arguments
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'prune':
        prune_mode()
    else:
        train_mode()


if __name__ == '__main__':
    main()
