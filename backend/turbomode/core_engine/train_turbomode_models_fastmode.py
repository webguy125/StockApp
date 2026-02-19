
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Single-Model Training Pipeline — One LightGBM model per sector

ARCHITECTURE: Single-model-per-sector (14d/±6%)
- Trains exactly ONE model per sector (11 sectors total)
- Uses label_14d_swing (14-day horizon, ±6% threshold)
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
Updated: 2026-02-18 (14-day ±6% alignment)
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

# SHAP imports (optional, fail gracefully if not installed)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] SHAP not installed. Feature importance logging disabled.")

# Configuration flags
ENABLE_SHAP_LOGGING = os.getenv('ENABLE_SHAP_LOGGING', 'false').lower() == 'true'
ENABLE_FEATURE_PRUNING = os.getenv('ENABLE_FEATURE_PRUNING', 'false').lower() == 'true'
ENABLE_SHAP_WATERFALL = os.getenv('ENABLE_SHAP_WATERFALL', 'false').lower() == 'true'
ENABLE_ADAPTIVE_PRUNING = os.getenv('ENABLE_ADAPTIVE_PRUNING', 'false').lower() == 'true'
SHAP_SAMPLE_SIZE = int(os.getenv('SHAP_SAMPLE_SIZE', '2000'))
SHAP_RANDOM_STATE = int(os.getenv('SHAP_RANDOM_STATE', '42'))
PRUNING_TOP_N = int(os.getenv('PRUNING_TOP_N', '60'))
SHAP_WATERFALL_MAX_PLOTS = int(os.getenv('SHAP_WATERFALL_MAX_PLOTS', '5'))
ADAPTIVE_PRUNING_MIN = int(os.getenv('ADAPTIVE_PRUNING_MIN', '40'))
ADAPTIVE_PRUNING_MAX = int(os.getenv('ADAPTIVE_PRUNING_MAX', '80'))
ADAPTIVE_PRUNING_THRESHOLD = float(os.getenv('ADAPTIVE_PRUNING_THRESHOLD', '0.92'))


def compute_shap_values(model, X_sample: np.ndarray, feature_names: List[str]) -> Dict:
    """
    Compute SHAP values for a trained model using TreeExplainer.

    Args:
        model: Trained tree-based model (LightGBM, XGBoost, CatBoost, RandomForest)
        X_sample: Sample data for SHAP computation (N, n_features)
        feature_names: List of feature names

    Returns:
        Dictionary with SHAP values, feature names, and summary statistics
    """
    if not SHAP_AVAILABLE:
        return None

    try:
        print(f"  [SHAP] Computing SHAP values for {len(X_sample):,} samples...")
        start_time = time.time()

        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For multiclass, shap_values is a list of arrays (one per class)
        # We'll compute mean |SHAP| across all classes
        if isinstance(shap_values, list):
            # Stack arrays and compute mean absolute values across classes
            stacked = np.stack([np.abs(sv) for sv in shap_values], axis=0)
            abs_shap = np.mean(stacked, axis=0)
        else:
            abs_shap = np.abs(shap_values)

        # Compute mean |SHAP| per feature (average across samples)
        mean_abs_shap = np.mean(abs_shap, axis=0)

        # Ensure we have the right number of features
        if len(mean_abs_shap) != len(feature_names):
            print(f"  [SHAP] Warning: Feature count mismatch ({len(mean_abs_shap)} vs {len(feature_names)})")
            # Pad or truncate as needed
            if len(mean_abs_shap) < len(feature_names):
                mean_abs_shap = np.pad(mean_abs_shap, (0, len(feature_names) - len(mean_abs_shap)))
            else:
                mean_abs_shap = mean_abs_shap[:len(feature_names)]

        # Create feature importance ranking
        feature_importance = {}
        for i, feature_name in enumerate(feature_names):
            val = mean_abs_shap[i]
            # Handle numpy scalars and arrays
            if isinstance(val, np.ndarray):
                val = val.item() if val.size == 1 else float(val.flat[0])
            feature_importance[feature_name] = float(val)

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        elapsed = time.time() - start_time
        print(f"  [SHAP] Computation complete in {elapsed:.2f}s")
        print(f"  [SHAP] Top 10 features by mean |SHAP|:")
        for i, (fname, importance) in enumerate(sorted_features[:10], 1):
            print(f"    {i}. {fname}: {importance:.6f}")

        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'sorted_features': sorted_features,
            'mean_abs_shap': mean_abs_shap.tolist()
        }

    except Exception as e:
        print(f"  [SHAP] Error computing SHAP values: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_shap_results(shap_results: Dict, sector: str, model_name: str, save_dir: str):
    """
    Save SHAP results to disk.

    Args:
        shap_results: Dictionary from compute_shap_values
        sector: Sector name
        model_name: Model identifier (e.g., 'lightgbm')
        save_dir: Base directory for saving
    """
    if shap_results is None:
        return

    # Create SHAP directory
    shap_dir = os.path.join(save_dir, sector, model_name, 'shap')
    os.makedirs(shap_dir, exist_ok=True)

    # Save SHAP values as numpy array
    shap_values_path = os.path.join(shap_dir, 'shap_values.npy')
    np.save(shap_values_path, shap_results['shap_values'])

    # Save feature names
    feature_names_path = os.path.join(shap_dir, 'feature_names.json')
    with open(feature_names_path, 'w') as f:
        json.dump(list(shap_results['feature_importance'].keys()), f, indent=2)

    # Save SHAP summary (mean |SHAP| per feature)
    shap_summary_path = os.path.join(shap_dir, 'shap_summary.json')
    with open(shap_summary_path, 'w') as f:
        json.dump({
            'feature_importance': shap_results['feature_importance'],
            'top_features': [
                {'name': name, 'importance': imp}
                for name, imp in shap_results['sorted_features'][:50]
            ]
        }, f, indent=2)

    print(f"  [SHAP] Results saved to {shap_dir}/")


def load_pruned_features(sector: str, model_name: str = 'lightgbm', save_dir: str = None) -> List[int]:
    """
    Load pruned feature indices from disk.

    Args:
        sector: Sector name
        model_name: Model identifier
        save_dir: Base directory for loading

    Returns:
        List of feature indices to keep, or None if pruning not enabled
    """
    if not ENABLE_FEATURE_PRUNING:
        return None

    if save_dir is None:
        save_dir = os.path.join(project_root, 'backend', 'turbomode', 'models', 'trained')

    pruned_path = os.path.join(save_dir, sector, model_name, 'pruned_features.json')

    if not os.path.exists(pruned_path):
        print(f"  [PRUNE] No pruned features found at {pruned_path}")
        return None

    with open(pruned_path, 'r') as f:
        pruned_data = json.load(f)

    feature_indices = pruned_data.get('feature_indices', None)
    print(f"  [PRUNE] Loaded {len(feature_indices)} pruned features")

    return feature_indices


def save_pruned_features(shap_results: Dict, sector: str, model_name: str, top_n: int, save_dir: str):
    """
    Save pruned feature list based on SHAP importance.

    Args:
        shap_results: Dictionary from compute_shap_values
        sector: Sector name
        model_name: Model identifier
        top_n: Number of top features to keep
        save_dir: Base directory for saving
    """
    if shap_results is None:
        return

    from backend.turbomode.core_engine.feature_list import FEATURE_INDEX

    # Get top N features by SHAP importance
    top_features = [name for name, _ in shap_results['sorted_features'][:top_n]]

    # Convert to indices
    feature_indices = [FEATURE_INDEX[fname] for fname in top_features if fname in FEATURE_INDEX]

    # Create model directory
    model_dir = os.path.join(save_dir, sector, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save pruned features
    pruned_path = os.path.join(model_dir, 'pruned_features.json')
    with open(pruned_path, 'w') as f:
        json.dump({
            'top_n': top_n,
            'feature_names': top_features,
            'feature_indices': feature_indices,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)

    print(f"  [PRUNE] Saved {len(feature_indices)} pruned features to {pruned_path}")


def compute_adaptive_pruning_threshold(shap_results: Dict, min_features: int, max_features: int, cumulative_threshold: float) -> int:
    """
    Compute optimal number of features based on cumulative SHAP importance.

    Args:
        shap_results: Dictionary from compute_shap_values
        min_features: Minimum number of features to keep
        max_features: Maximum number of features to keep
        cumulative_threshold: Cumulative importance threshold (e.g., 0.92 for 92%)

    Returns:
        Optimal number of features to keep
    """
    if shap_results is None:
        return max_features

    # Get sorted feature importance
    sorted_features = shap_results['sorted_features']
    importance_values = [imp for _, imp in sorted_features]

    # Normalize to sum to 1
    total_importance = sum(importance_values)
    if total_importance == 0:
        return max_features

    normalized = [imp / total_importance for imp in importance_values]

    # Compute cumulative sum
    cumsum = 0.0
    for i, imp in enumerate(normalized):
        cumsum += imp
        if cumsum >= cumulative_threshold:
            optimal = i + 1
            # Clamp to min/max
            return max(min_features, min(max_features, optimal))

    # If threshold not reached, return max
    return max_features


def generate_shap_waterfall_plots(
    model,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    feature_names: List[str],
    sector: str,
    model_name: str,
    save_dir: str,
    max_plots: int = 5
):
    """
    Generate SHAP waterfall plots for top predictions.

    Args:
        model: Trained model
        X_sample: Sample data
        y_sample: Sample labels
        feature_names: List of feature names
        sector: Sector name
        model_name: Model identifier
        save_dir: Base directory for saving plots
        max_plots: Maximum number of plots to generate
    """
    if not SHAP_AVAILABLE:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        print(f"  [SHAP] Generating waterfall plots (max {max_plots})...")

        # Create explainer
        explainer = shap.TreeExplainer(model)

        # Get predictions and confidence
        probs = model.predict_proba(X_sample)
        max_probs = np.max(probs, axis=1)

        # Select top-N samples by confidence
        top_indices = np.argsort(max_probs)[-max_plots:][::-1]

        # Create waterfall directory
        waterfall_dir = os.path.join(save_dir, sector, model_name, 'shap', 'waterfall')
        os.makedirs(waterfall_dir, exist_ok=True)

        # Generate plots
        for i, sample_idx in enumerate(top_indices):
            X_instance = X_sample[sample_idx:sample_idx+1]
            shap_values = explainer(X_instance)

            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0, :, 1], max_display=15, show=False)

            # Save plot
            plot_path = os.path.join(waterfall_dir, f'waterfall_{i+1}_conf_{max_probs[sample_idx]:.3f}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

        print(f"  [SHAP] Saved {len(top_indices)} waterfall plots to {waterfall_dir}/")

    except Exception as e:
        print(f"  [SHAP] Error generating waterfall plots: {e}")


def save_fastmode_models(model, sector: str, save_dir: str = None, horizon_days: int = 14):
    """
    Save single Fast Mode model to disk.

    ARCHITECTURE: Single-model-per-sector (14d/±6%)

    Args:
        model: Trained sklearn-style model
        sector: Sector name
        save_dir: Base directory for saving models (default: backend/turbomode/models/trained)
        horizon_days: Trading horizon in days (default: 14)
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
        'horizon_days': horizon_days,
        'threshold_pct': 6,
        'label': 'label_14d_swing',
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

    ARCHITECTURE: Single-model-per-sector (14d/±6%)

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
    horizon_days: int = 14,
    save_models: bool = True,
    save_dir: str = None
) -> Dict:
    """
    Train exactly ONE model per sector using LightGBM.

    ARCHITECTURE: Single-model-per-sector (14d/±6%)
    - No ensemble
    - No meta-learner
    - Just one LightGBM classifier per sector
    - Optional SHAP logging and feature pruning

    Args:
        sector: Sector name (e.g., 'technology', 'healthcare')
        X_train: Training features (N, 179)
        y_train: Training labels (N,) - 0=SELL, 1=HOLD, 2=BUY
        X_val: Validation features (M, 179)
        y_val: Validation labels (M,)
        horizon_days: Prediction horizon (default: 14 days)
        save_models: Whether to save trained model to disk
        save_dir: Base directory for saving model (flat structure)

    Returns:
        Dictionary with training results
    """
    from backend.turbomode.core_engine.feature_list import FEATURE_LIST

    print(f"\n[{sector.upper()}] Starting single-model training...")
    print(f"[{sector.upper()}] Data: {len(X_train):,} train, {len(X_val):,} val")
    print(f"[{sector.upper()}] Horizon: {horizon_days}d, Label: label_14d_swing")

    if ENABLE_SHAP_LOGGING and SHAP_AVAILABLE:
        print(f"[{sector.upper()}] SHAP logging: ENABLED")
    if ENABLE_FEATURE_PRUNING:
        print(f"[{sector.upper()}] Feature pruning: ENABLED (top {PRUNING_TOP_N} features)")

    sector_start = time.time()

    # ========================================================================
    # AUTO-FEATURE PRUNING: Load and apply pruned features if available
    # ========================================================================
    pruned_indices = None
    original_feature_count = X_train.shape[1]
    pruned_feature_names = None

    if ENABLE_FEATURE_PRUNING:
        pruned_indices = load_pruned_features(sector, 'lightgbm', save_dir)

        if pruned_indices is not None and len(pruned_indices) > 0:
            print(f"[{sector.upper()}] Applying feature pruning: {len(pruned_indices)}/{original_feature_count} features")

            # Apply feature mask
            X_train = X_train[:, pruned_indices]
            X_val = X_val[:, pruned_indices]

            # Update feature names for SHAP
            pruned_feature_names = [FEATURE_LIST[i] for i in pruned_indices]
        else:
            print(f"[{sector.upper()}] No pruned features found - using all {original_feature_count} features")

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

    # ========================================================================
    # POST-TRAINING: SHAP LOGGING
    # ========================================================================
    shap_results = None
    if ENABLE_SHAP_LOGGING and SHAP_AVAILABLE and save_models:
        print(f"[{sector}] Computing SHAP values...")

        # Deterministic sampling for SHAP computation
        sample_size = min(SHAP_SAMPLE_SIZE, len(X_train))
        np.random.seed(SHAP_RANDOM_STATE)
        sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_sample = X_train[sample_indices]

        # Use pruned feature names if pruning was applied, otherwise use full list
        feature_names_for_shap = pruned_feature_names if pruned_feature_names is not None else FEATURE_LIST

        # Compute SHAP values
        shap_results = compute_shap_values(model, X_sample, feature_names_for_shap)

        # Save SHAP results
        if shap_results is not None and save_dir is not None:
            save_shap_results(shap_results, sector, 'lightgbm', save_dir)

            # Adaptive pruning: compute optimal number of features
            if ENABLE_ADAPTIVE_PRUNING:
                adaptive_n = compute_adaptive_pruning_threshold(
                    shap_results,
                    ADAPTIVE_PRUNING_MIN,
                    ADAPTIVE_PRUNING_MAX,
                    ADAPTIVE_PRUNING_THRESHOLD
                )
                print(f"  [ADAPTIVE] Optimal feature count: {adaptive_n} (cumulative {ADAPTIVE_PRUNING_THRESHOLD*100:.0f}%)")
                save_pruned_features(shap_results, sector, 'lightgbm', adaptive_n, save_dir)
            elif ENABLE_FEATURE_PRUNING:
                # Standard pruning: fixed top-N
                save_pruned_features(shap_results, sector, 'lightgbm', PRUNING_TOP_N, save_dir)

            # Generate SHAP waterfall plots
            if ENABLE_SHAP_WATERFALL:
                generate_shap_waterfall_plots(
                    model,
                    X_sample,
                    y_train[sample_indices],
                    feature_names_for_shap,
                    sector,
                    'lightgbm',
                    save_dir,
                    SHAP_WATERFALL_MAX_PLOTS
                )

    # Save model if requested (flat directory structure)
    if save_models:
        save_fastmode_models(model, sector, save_dir=save_dir, horizon_days=horizon_days)

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
        'model': model,
        'shap_enabled': ENABLE_SHAP_LOGGING and SHAP_AVAILABLE,
        'pruning_enabled': ENABLE_FEATURE_PRUNING
    }


def train_all_sectors_fastmode(
    sectors_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    horizon_days: int = 14
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
