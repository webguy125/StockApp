"""
1D/2D Horizon Training Entry Point - GPU System

Trains 11 sector-specific models for BOTH 1-day and 2-day horizons.
Uses the full GPU training system (8 XGBoost variants + LightGBM + CatBoost + Meta-learner).

Models saved to:
- models/trained/{sector}/1d/  (1-day horizon)
- models/trained/{sector}/2d/  (2-day horizon)
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.turbomode.train_turbomode_models import train_all_sectors_parallel

if __name__ == "__main__":
    SECTOR_THRESHOLDS_V1 = {
        "buy": 0.10,
        "sell": -0.10
    }

    print("\n" + "="*80)
    print("1D/2D HORIZON TRAINING - GPU SYSTEM")
    print("="*80)
    print()
    print("This script trains models for TWO horizons:")
    print("  1. 1-day horizon (1d)")
    print("  2. 2-day horizon (2d)")
    print()
    print("Each horizon gets:")
    print("  - 11 sector-specific model sets")
    print("  - 8 base models per sector (XGBoost variants, LightGBM, CatBoost)")
    print("  - 1 meta-learner per sector")
    print()
    print("="*80)
    print()

    # Train 1-day horizon
    print("\n\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "1-DAY HORIZON TRAINING" + " " * 31 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()

    train_all_sectors_parallel(max_workers=1, horizon_days=1, thresholds=SECTOR_THRESHOLDS_V1)

    # Train 2-day horizon
    print("\n\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "2-DAY HORIZON TRAINING" + " " * 31 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()

    train_all_sectors_parallel(max_workers=1, horizon_days=2, thresholds=SECTOR_THRESHOLDS_V1)

    print("\n\n")
    print("="*80)
    print("1D/2D TRAINING COMPLETE")
    print("="*80)
    print()
    print("Models saved to:")
    print("  - models/trained/{sector}/1d/")
    print("  - models/trained/{sector}/2d/")
    print()
    print("="*80)
