"""
Quick test script to verify sector training works with one sector
Tests the fix for Windows multiprocessing issue
"""
import sys
import os
from pathlib import Path
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader
from backend.turbomode.training_symbols import TRAINING_SYMBOLS
from backend.turbomode.train_sector_models_parallel import (
    get_sector_symbols,
    train_single_sector_worker
)

print("=" * 80)
print("TESTING SINGLE SECTOR TRAINING (TECHNOLOGY)")
print("=" * 80)

# Load data
print("\n[1/4] Loading training data...")
loader = TurboModeTrainingDataLoader()
X_all, y_all, symbols_all = loader.load_training_data(return_split=False)
print(f"Loaded {len(X_all):,} samples")

# Filter for technology sector
print("\n[2/4] Filtering for technology sector...")
sector = 'technology'
sector_symbols = get_sector_symbols(sector)
print(f"Technology sector has {len(sector_symbols)} symbols")

mask = np.isin(symbols_all, sector_symbols)
X_sector = X_all[mask]
y_sector = y_all[mask]
print(f"Technology sector has {len(X_sector):,} samples")

# Split train/val
print("\n[3/4] Splitting into train/val...")
split_idx = int(0.8 * len(X_sector))
X_train = X_sector[:split_idx]
y_train = y_sector[:split_idx]
X_val = X_sector[split_idx:]
y_val = y_sector[split_idx:]
print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

# Train sector
print("\n[4/4] Training technology sector...")
print("=" * 80)
result = train_single_sector_worker(sector, X_train, y_train, X_val, y_val)
print("=" * 80)

print("\nRESULT:")
print(f"Status: {result['status']}")
if result['status'] == 'completed':
    print(f"Base models: {result['num_base_models']}")
    print(f"Meta accuracy: {result['meta_accuracy']:.4f}")
    print(f"Training time: {result['training_time_minutes']:.1f} min")
    print("\n[SUCCESS] Technology sector trained successfully!")
else:
    print(f"Details: {result}")
    print("\n[FAILED] Technology sector training failed")
