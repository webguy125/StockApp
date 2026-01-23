"""
Single Sector Test: Technology Sector - 1D/2D Horizons
Tests the full GPU training pipeline with canonical candles integration
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.turbomode.train_turbomode_models import train_all_sectors_parallel

if __name__ == "__main__":
    SECTOR_THRESHOLDS = {
        "buy": 0.10,
        "sell": -0.10
    }

    print("=" * 80)
    print("SINGLE SECTOR TEST: TECHNOLOGY - 1D/2D HORIZONS")
    print("=" * 80)
    print()
    print("Training pipeline:")
    print("  - Sector: technology")
    print("  - Horizons: 1d, 2d")
    print("  - Models per horizon: 8 base + 1 meta = 9 models")
    print("  - Total models: 18 (9 x 2 horizons)")
    print("  - Data source: Canonical candles table")
    print()
    print("=" * 80)
    print()

    # Train 1-day horizon for technology sector
    print("\n[1D HORIZON] Training technology sector...")
    train_all_sectors_parallel(
        max_workers=1,
        horizon_days=1,
        thresholds=SECTOR_THRESHOLDS,
        sectors_filter=['technology']
    )

    # Train 2-day horizon for technology sector
    print("\n[2D HORIZON] Training technology sector...")
    train_all_sectors_parallel(
        max_workers=1,
        horizon_days=2,
        thresholds=SECTOR_THRESHOLDS,
        sectors_filter=['technology']
    )

    print("\n" + "=" * 80)
    print("SINGLE SECTOR TEST COMPLETE")
    print("=" * 80)
    print()
    print("Models saved to:")
    print("  - backend/turbomode/models/trained/technology/1d/")
    print("  - backend/turbomode/models/trained/technology/2d/")
    print()
