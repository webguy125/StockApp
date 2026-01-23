"""
Single Sector Test: Technology - 1D/2D Horizons
Loads data once, trains technology sector for both horizons
"""
import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader
from backend.turbomode.training_symbols import get_symbols_by_sector
from backend.turbomode.train_turbomode_models import train_single_sector_worker

SECTOR_THRESHOLDS = {
    "buy": 0.10,
    "sell": -0.10
}

print("=" * 80)
print("SINGLE SECTOR TEST: TECHNOLOGY - 1D/2D HORIZONS")
print("=" * 80)
print()

# Train 1-day horizon
print("[1D HORIZON] Loading data and training technology sector...")
loader = TurboModeTrainingDataLoader()

tech_symbols = get_symbols_by_sector('technology')
print(f"Technology symbols: {len(tech_symbols)}")

X_train, y_train, X_val, y_val = loader.load_training_data(
    symbols_filter=tech_symbols,
    return_split=True,
    test_size=0.2,
    horizon_days=1,
    thresholds=SECTOR_THRESHOLDS
)

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
result_1d = train_single_sector_worker('technology', X_train, y_train, X_val, y_val, horizon_days=1)
print(f"1D Result: {result_1d['status']}, Meta Accuracy: {result_1d.get('meta_accuracy', 0):.4f}")

# Train 2-day horizon
print("\n[2D HORIZON] Loading data and training technology sector...")
X_train, y_train, X_val, y_val = loader.load_training_data(
    symbols_filter=tech_symbols,
    return_split=True,
    test_size=0.2,
    horizon_days=2,
    thresholds=SECTOR_THRESHOLDS
)

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")
result_2d = train_single_sector_worker('technology', X_train, y_train, X_val, y_val, horizon_days=2)
print(f"2D Result: {result_2d['status']}, Meta Accuracy: {result_2d.get('meta_accuracy', 0):.4f}")

print("\n" + "=" * 80)
print("SINGLE SECTOR TEST COMPLETE")
print("=" * 80)
