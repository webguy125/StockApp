"""
Fast Mode Test: Technology Sector - 1D/2D Horizons

Tests the new sklearn-style Fast Mode training pipeline.
NO wrapper classes, NO BASE_MODELS list, just direct sklearn usage.
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader
from backend.turbomode.training_symbols import get_symbols_by_sector
from backend.turbomode.train_turbomode_models_fastmode import train_single_sector_worker_fastmode

SECTOR_THRESHOLDS = {
    "buy": 0.10,
    "sell": -0.10
}

print("=" * 80)
print("FAST MODE TEST: TECHNOLOGY SECTOR - 1D/2D HORIZONS")
print("=" * 80)
print()

# Get technology symbols
tech_symbols = get_symbols_by_sector('technology')
print(f"Technology symbols: {len(tech_symbols)}")

# Initialize loader
loader = TurboModeTrainingDataLoader()

# ============================================================================
# 1D HORIZON
# ============================================================================
print("\n[1D HORIZON] Loading data and training technology sector...")

X_train, y_train, X_val, y_val = loader.load_training_data(
    symbols_filter=tech_symbols,
    return_split=True,
    test_size=0.2,
    horizon_days=1,
    thresholds=SECTOR_THRESHOLDS
)

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

result_1d = train_single_sector_worker_fastmode(
    'technology',
    X_train,
    y_train,
    X_val,
    y_val,
    horizon_days=1,
    save_models=True
)

print(f"\n1D Result: {result_1d['status']}")
print(f"Meta Accuracy: {result_1d['meta_accuracy']:.4f}")
print(f"Training Time: {result_1d['total_time']/60:.1f} minutes")

# ============================================================================
# 2D HORIZON
# ============================================================================
print("\n[2D HORIZON] Loading data and training technology sector...")

X_train, y_train, X_val, y_val = loader.load_training_data(
    symbols_filter=tech_symbols,
    return_split=True,
    test_size=0.2,
    horizon_days=2,
    thresholds=SECTOR_THRESHOLDS
)

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

result_2d = train_single_sector_worker_fastmode(
    'technology',
    X_train,
    y_train,
    X_val,
    y_val,
    horizon_days=2,
    save_models=True
)

print(f"\n2D Result: {result_2d['status']}")
print(f"Meta Accuracy: {result_2d['meta_accuracy']:.4f}")
print(f"Training Time: {result_2d['total_time']/60:.1f} minutes")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FAST MODE TEST COMPLETE")
print("=" * 80)
print(f"\n1D Horizon:")
print(f"  Training Time: {result_1d['total_time']/60:.1f} min")
print(f"  Meta Accuracy: {result_1d['meta_accuracy']:.4f}")
print(f"\n2D Horizon:")
print(f"  Training Time: {result_2d['total_time']/60:.1f} min")
print(f"  Meta Accuracy: {result_2d['meta_accuracy']:.4f}")
print(f"\nTotal Time: {(result_1d['total_time'] + result_2d['total_time'])/60:.1f} minutes")
print("\nBase Model Accuracies (1D):")
for model_name, metrics in result_1d['base_models'].items():
    print(f"  {model_name}: {metrics['val_accuracy']:.4f} ({metrics['training_time']:.1f}s)")
print("\nBase Model Accuracies (2D):")
for model_name, metrics in result_2d['base_models'].items():
    print(f"  {model_name}: {metrics['val_accuracy']:.4f} ({metrics['training_time']:.1f}s)")
