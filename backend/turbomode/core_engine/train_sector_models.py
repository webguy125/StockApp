
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Per-Sector Fast Ensemble Training Pipeline

Trains 6 models per sector:
- 5 fast base models (3 GPU, 2 CPU)
- 1 MetaLearner (LogisticRegression on stacked predictions)

CLASS SEMANTICS (CORRECTED):
- Index 0: SELL (go short, open bearish position)
- Index 1: HOLD (do nothing, no new position)
- Index 2: BUY (go long, open bullish position)

NOTE: This matches the training labels in sector_batch_trainer.py (0=SELL, 1=HOLD, 2=BUY)

Uses out-of-fold predictions to prevent overfitting in MetaLearner.
"""

import numpy as np
import pickle
import os
from typing import Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import logging

# Import model registry
from backend.turbomode.core_engine.model_registry import (
    MODELS_BASE_DIR,
    BASE_MODELS,
    META_LEARNER,
    get_model_path
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_lightgbm_gpu(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    """
    Train LightGBM with GPU acceleration.

    CLASS SEMANTICS: [0=SELL, 1=HOLD, 2=BUY] (matches sector_batch_trainer.py)
    """
    logger.info("  Training LightGBM (GPU)...")

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        device='gpu',
        gpu_platform_id=0,
        gpu_device_id=0,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    model.fit(X, y)
    logger.info("    ✓ LightGBM trained")
    return model


def train_catboost_gpu(X: np.ndarray, y: np.ndarray) -> CatBoostClassifier:
    """
    Train CatBoost with GPU acceleration.

    CLASS SEMANTICS: [0=SELL, 1=HOLD, 2=BUY] (matches sector_batch_trainer.py)
    """
    logger.info("  Training CatBoost (GPU)...")

    model = CatBoostClassifier(
        loss_function='MultiClass',
        classes_count=3,
        task_type='GPU',
        devices='0',
        iterations=200,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False
    )

    model.fit(X, y)
    logger.info("    ✓ CatBoost trained")
    return model


def train_xgb_hist_gpu(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    """
    Train XGBoost with GPU hist method.

    CLASS SEMANTICS: [0=SELL, 1=HOLD, 2=BUY] (matches sector_batch_trainer.py)
    """
    logger.info("  Training XGBoost-Hist (GPU)...")

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        tree_method='hist',
        device='cuda',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    model.fit(X, y)
    logger.info("    ✓ XGBoost-Hist trained")
    return model


def train_xgb_linear(X: np.ndarray, y: np.ndarray) -> xgb.XGBClassifier:
    """
    Train XGBoost with linear booster (CPU).

    CLASS SEMANTICS: [0=SELL, 1=HOLD, 2=BUY] (matches sector_batch_trainer.py)
    """
    logger.info("  Training XGBoost-Linear (CPU)...")

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        booster='gblinear',
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )

    model.fit(X, y)
    logger.info("    ✓ XGBoost-Linear trained")
    return model


def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Train RandomForest classifier (CPU).

    CLASS SEMANTICS: [0=SELL, 1=HOLD, 2=BUY] (matches sector_batch_trainer.py)
    """
    logger.info("  Training RandomForest (CPU)...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    model.fit(X, y)
    logger.info("    ✓ RandomForest trained")
    return model


def generate_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Generate out-of-fold predictions for MetaLearner stacking.

    Uses K-fold cross-validation to prevent overfitting:
    - Split data into K folds
    - For each fold: train on K-1 folds, predict on holdout fold
    - Concatenate all holdout predictions to get full training set predictions

    Args:
        X: Feature matrix (N, n_features)
        y: Target labels (N,) - [0=SELL, 1=HOLD, 2=BUY] (matches sector_batch_trainer.py)
        n_splits: Number of CV folds (default 5)

    Returns:
        oof_predictions: Stacked predictions (N, 15) - 5 models x 3 classes each
        base_models: Dict of fully-trained base models on complete dataset
    """
    logger.info(f"  Generating out-of-fold predictions ({n_splits}-fold CV)...")

    n_samples = X.shape[0]
    n_classes = 3
    n_models = len(BASE_MODELS)

    # Initialize OOF prediction array (N, 15) - 5 models x 3 classes
    oof_predictions = np.zeros((n_samples, n_models * n_classes))

    # K-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"    Fold {fold_idx + 1}/{n_splits}...")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Train each base model on this fold
        fold_models = [
            train_lightgbm_gpu(X_train_fold, y_train_fold),
            train_catboost_gpu(X_train_fold, y_train_fold),
            train_xgb_hist_gpu(X_train_fold, y_train_fold),
            train_xgb_linear(X_train_fold, y_train_fold),
            train_random_forest(X_train_fold, y_train_fold)
        ]

        # Predict on validation fold
        for model_idx, model in enumerate(fold_models):
            probs = model.predict_proba(X_val_fold)  # shape: (n_val, 3)

            # Store in OOF array
            start_col = model_idx * n_classes
            end_col = start_col + n_classes
            oof_predictions[val_idx, start_col:end_col] = probs

    logger.info("    ✓ Out-of-fold predictions generated")

    # Now train final base models on FULL dataset
    logger.info("  Training final base models on full dataset...")
    base_models = {
        'lightgbm_gpu': train_lightgbm_gpu(X, y),
        'catboost_gpu': train_catboost_gpu(X, y),
        'xgb_hist_gpu': train_xgb_hist_gpu(X, y),
        'xgb_linear': train_xgb_linear(X, y),
        'random_forest': train_random_forest(X, y)
    }
    logger.info("    ✓ Final base models trained")

    return oof_predictions, base_models


def train_meta_learner(
    oof_predictions: np.ndarray,
    y: np.ndarray
) -> LogisticRegression:
    """
    Train MetaLearner on stacked out-of-fold predictions.

    Args:
        oof_predictions: Stacked OOF predictions (N, 15)
        y: Target labels (N,) - [0=SELL, 1=HOLD, 2=BUY] (matches sector_batch_trainer.py)

    Returns:
        Trained LogisticRegression meta-learner
    """
    logger.info("  Training MetaLearner (LogisticRegression)...")

    meta_learner = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        verbose=0
    )

    meta_learner.fit(oof_predictions, y)
    logger.info("    ✓ MetaLearner trained")

    return meta_learner


def train_sector_ensemble(
    sector: str,
    X: np.ndarray,
    y: np.ndarray
) -> Dict[str, str]:
    """
    Train complete ensemble for a sector (5 base models + MetaLearner).

    Args:
        sector: Sector name (e.g., 'technology')
        X: Feature matrix (N, n_features)
        y: Target labels (N,) - [0=SELL, 1=HOLD, 2=BUY] (matches sector_batch_trainer.py)

    Returns:
        Dictionary with paths to saved models
    """
    logger.info(f"[TRAIN] Starting ensemble training for {sector}")
    logger.info(f"  Training samples: {X.shape[0]}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Classes: BUY={np.sum(y==0)}, SELL={np.sum(y==1)}, HOLD={np.sum(y==2)}")

    # Create sector directory
    sector_dir = os.path.join(MODELS_BASE_DIR, sector)
    os.makedirs(sector_dir, exist_ok=True)
    logger.info(f"  Model directory: {sector_dir}")

    # Step 1: Generate OOF predictions and train base models
    oof_predictions, base_models = generate_oof_predictions(X, y, n_splits=5)

    # Step 2: Train MetaLearner on OOF predictions
    meta_learner = train_meta_learner(oof_predictions, y)

    # Step 3: Save all models
    logger.info("  Saving models...")
    saved_paths = {}

    # Save base models
    for model_name in BASE_MODELS:
        model_path = get_model_path(sector, model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(base_models[model_name], f)
        saved_paths[model_name] = model_path
        logger.info(f"    ✓ Saved {model_name}")

    # Save MetaLearner
    meta_path = get_model_path(sector, META_LEARNER)
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_learner, f)
    saved_paths[META_LEARNER] = meta_path
    logger.info(f"    ✓ Saved {META_LEARNER}")

    logger.info(f"[SUCCESS] Ensemble training completed for {sector}")
    logger.info(f"  Total models saved: {len(saved_paths)}")

    return saved_paths


if __name__ == '__main__':
    # Test with synthetic data
    logger.info("Testing ensemble training with synthetic data...")

    np.random.seed(42)
    n_samples = 1000
    n_features = 179

    # Generate synthetic features and labels
    X_test = np.random.randn(n_samples, n_features)
    y_test = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.2, 0.5])

    # Train ensemble for test sector
    test_sector = 'technology'
    saved_paths = train_sector_ensemble(test_sector, X_test, y_test)

    logger.info("\nSaved model paths:")
    for model_name, path in saved_paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        logger.info(f"  {exists} {model_name}: {path}")
