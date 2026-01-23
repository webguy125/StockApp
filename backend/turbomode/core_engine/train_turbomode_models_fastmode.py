
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Single-Model Training Pipeline — One LightGBM model per sector

ARCHITECTURE: Single-model-per-sector (1d/5% only)
- Trains exactly ONE model per sector (11 sectors total)
- Uses only label_1d_5pct (1-day horizon, 5% threshold)
- No ensemble, no meta-learner, no multi-model architecture
- Flat directory structure: models/<sector>/model.pkl

Model:
- LightGBM (LGBMClassifier with GPU acceleration)

Key differences from ensemble architecture:
- NO 5-model ensemble
- NO meta-learner stacking
- NO multi-horizon/threshold support
- Just one trained model per sector
- Simple, deterministic, fast training workflow
- Training time: ~45-60 minutes for all 11 sectors

This is the ONLY training pipeline for TurboMode.
"""

import sys
import os
import numpy as np
import time
import pickle
import json
from typing import Dict, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def save_fastmode_models(model, sector: str, save_dir: str = None):
    """
    Save single Fast Mode model to disk.

    ARCHITECTURE: Single-model-per-sector (1d/5% only)

    Args:
        model: Trained sklearn-style model
        sector: Sector name
        save_dir: Base directory for saving models (default: backend/turbomode/models/trained)
    """
    if save_dir is None:
        save_dir = os.path.join(project_root, 'backend', 'turbomode', 'models', 'trained')

    # Flat directory structure: models/<sector>/model.pkl
    sector_dir = os.path.join(save_dir, sector)
    os.makedirs(sector_dir, exist_ok=True)

    # Save single model
    model_path = os.path.join(sector_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = {
        'sector': sector,
        'horizon_days': 1,
        'threshold_pct': 5,
        'label': 'label_1d_5pct',
        'architecture': 'single_model',
        'model_type': type(model).__name__,
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    metadata_path = os.path.join(sector_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Model saved to {sector_dir}/model.pkl")


def load_fastmode_models(sector: str, load_dir: str = None):
    """
    Load single Fast Mode model from disk.

    ARCHITECTURE: Single-model-per-sector (1d/5% only)

    Args:
        sector: Sector name
        load_dir: Base directory for loading models

    Returns:
        Loaded sklearn-style model
    """
    if load_dir is None:
        load_dir = os.path.join(project_root, 'backend', 'turbomode', 'models', 'trained')

    # Flat directory structure: models/<sector>/model.pkl
    sector_dir = os.path.join(load_dir, sector)

    if not os.path.exists(sector_dir):
        raise ValueError(f"Model directory not found: {sector_dir}")

    # Load single model
    model_path = os.path.join(sector_dir, 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load metadata (optional, for info)
    metadata_path = os.path.join(sector_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            print(f"[OK] Model loaded from {sector_dir}/model.pkl ({metadata.get('model_type', 'unknown')})")
    else:
        print(f"[OK] Model loaded from {sector_dir}/model.pkl")

    return model


def train_single_sector_worker_fastmode(
    sector: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    horizon_days: int = 1,
    save_models: bool = True,
    save_dir: str = None
) -> Dict:
    """
    Train exactly ONE model per sector using LightGBM.

    ARCHITECTURE: Single-model-per-sector (1d/5% only)
    - No ensemble
    - No meta-learner
    - Just one LightGBM classifier per sector

    Args:
        sector: Sector name (e.g., 'technology', 'healthcare')
        X_train: Training features (N, 179)
        y_train: Training labels (N,) - 0=SELL, 1=HOLD, 2=BUY
        X_val: Validation features (M, 179)
        y_val: Validation labels (M,)
        horizon_days: Prediction horizon (always 1 for single-model architecture)
        save_models: Whether to save trained model to disk
        save_dir: Base directory for saving model (flat structure)

    Returns:
        Dictionary with training results
    """
    print(f"\n[{sector.upper()}] Starting single-model training...")
    print(f"[{sector.upper()}] Data: {len(X_train):,} train, {len(X_val):,} val")
    print(f"[{sector.upper()}] Label: label_1d_5pct")

    sector_start = time.time()

    # ========================================================================
    # SINGLE MODEL ARCHITECTURE - LightGBM only
    # ========================================================================
    model = LGBMClassifier(
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    # ========================================================================

    # Train single model
    model_start = time.time()
    model.fit(X_train, y_train)

    # Get accuracy
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    model_time = time.time() - model_start

    print(f"[{sector}] LightGBM: train_acc={train_acc:.4f}, val_acc={val_acc:.4f} ({model_time:.1f}s)")

    # Save model if requested (flat directory structure)
    if save_models:
        save_fastmode_models(model, sector, save_dir=save_dir)

    sector_time = time.time() - sector_start
    print(f"[{sector.upper()}] COMPLETE - {sector_time/60:.1f} min")

    return {
        'status': 'completed',
        'sector': sector,
        'horizon_days': horizon_days,
        'model_type': 'LightGBM',
        'train_accuracy': train_acc,
        'accuracy': val_acc,
        'total_time': sector_time,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'model': model
    }


def train_all_sectors_fastmode(
    sectors_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    horizon_days: int = 1
) -> Dict[str, Dict]:
    """
    Train all sectors in Fast Mode.

    Args:
        sectors_data: Dictionary mapping sector names to (X_train, y_train, X_val, y_val)
        horizon_days: Prediction horizon

    Returns:
        Dictionary mapping sector names to training results
    """
    print("\n" + "=" * 80)
    print(f"FAST MODE TRAINING - {len(sectors_data)} SECTORS - {horizon_days}D HORIZON")
    print("=" * 80)

    all_results = {}

    for sector, (X_train, y_train, X_val, y_val) in sectors_data.items():
        result = train_single_sector_worker_fastmode(
            sector, X_train, y_train, X_val, y_val, horizon_days
        )
        all_results[sector] = result

    print("\n" + "=" * 80)
    print("FAST MODE TRAINING COMPLETE")
    print("=" * 80)

    return all_results
