"""
TurboMode Training Script - Purified Architecture (Genesis Rebuild)

This script trains all 9 TurboMode models (8 base + 1 meta) using RAW features only.
NO StandardScaler, NO preprocessing, NO contamination.

Architecture:
- 8 Base Models: 6 XGBoost variants + LightGBM + CatBoost (NNs disabled)
- 1 Meta-Learner: Combines base model predictions (probabilities)
- Feature Flow: RAW 179 features → Base Models → Probabilities → Meta-Learner

PROHIBITIONS:
- NO StandardScaler anywhere
- NO scaler.pkl files
- NO preprocessing transforms
- NO advanced_ml imports
- NO legacy wrappers
- NO feature pipelines
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

# ============================================================================
# PURIFIED MODEL IMPORTS - backend.turbomode.models ONLY
# ============================================================================

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader

# Tree-based models (scale-invariant, accept RAW features)
from backend.turbomode.models.xgboost_model import XGBoostModel
from backend.turbomode.models.xgboost_et_model import XGBoostETModel
from backend.turbomode.models.xgboost_hist_model import XGBoostHistModel
from backend.turbomode.models.xgboost_dart_model import XGBoostDARTModel
from backend.turbomode.models.xgboost_gblinear_model import XGBoostGBLinearModel
from backend.turbomode.models.xgboost_approx_model import XGBoostApproxModel
from backend.turbomode.models.lightgbm_model import LightGBMModel
from backend.turbomode.models.catboost_model import CatBoostModel

# Meta-learner (accepts RAW probability vectors)
from backend.turbomode.models.meta_learner import MetaLearner

# Neural network models (use INTERNAL BatchNorm, accept RAW features)
from backend.turbomode.models.tc_nn_model import TurboCoreNNWrapper

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_BASE_PATH = 'backend/data/turbomode_models'

# Ensure model directory exists
os.makedirs(MODEL_BASE_PATH, exist_ok=True)

# Model registry (order matters for deterministic execution)
# NOTE: Neural networks (tc_nn_lstm, tc_nn_gru) disabled due to architectural mismatch
#       - LSTM/GRU expect sequential data, but TurboMode uses tabular features
#       - Training resulted in 11.7% and 8.9% val accuracy (worse than 33% random)
#       - 8 tree models provide excellent performance (best: 87.2% val accuracy)
BASE_MODELS = [
    ('xgboost', XGBoostModel, 'xgboost'),
    ('xgboost_et', XGBoostETModel, 'xgboost_et'),
    ('lightgbm', LightGBMModel, 'lightgbm'),
    ('catboost', CatBoostModel, 'catboost'),
    ('xgboost_hist', XGBoostHistModel, 'xgboost_hist'),
    ('xgboost_dart', XGBoostDARTModel, 'xgboost_dart'),
    ('xgboost_gblinear', XGBoostGBLinearModel, 'xgboost_gblinear'),
    ('xgboost_approx', XGBoostApproxModel, 'xgboost_approx'),
    # ('tc_nn_lstm', TurboCoreNNWrapper, 'tc_nn_lstm'),  # DISABLED - architectural mismatch
    # ('tc_nn_gru', TurboCoreNNWrapper, 'tc_nn_gru')     # DISABLED - architectural mismatch
]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load RAW training data from TurboMode database.

    Returns:
        X_train, y_train, X_val, y_val (all RAW, NO scaling)
    """
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)

    loader = TurboModeTrainingDataLoader()

    # Load all data (RAW features, 179 dimensions)
    X_all, y_all = loader.load_training_data()

    print(f"[OK] Loaded {len(X_all):,} samples with {X_all.shape[1]} RAW features")
    print(f"   Feature shape: {X_all.shape}")
    print(f"   Label shape: {y_all.shape}")
    print(f"   Label distribution: {np.bincount(y_all)}")

    # Split into train/val (80/20)
    n_train = int(len(X_all) * 0.8)
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val = X_all[n_train:]
    y_val = y_all[n_train:]

    print(f"[OK] Train: {len(X_train):,} samples")
    print(f"[OK] Val:   {len(X_val):,} samples")

    return X_train, y_train, X_val, y_val


# ============================================================================
# BASE MODEL TRAINING
# ============================================================================

def train_base_model(model_name: str, model_class, model_path: str,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> object:
    """
    Train a single base model on RAW features.

    Args:
        model_name: Model identifier
        model_class: Model class constructor
        model_path: Path to save model
        X_train, y_train: Training data (RAW features)
        X_val, y_val: Validation data (RAW features)

    Returns:
        Trained model instance
    """
    print(f"\n{'-'*80}")
    print(f"Training: {model_name.upper()}")
    print(f"{'-'*80}")

    # Initialize model
    full_path = os.path.join(MODEL_BASE_PATH, model_path)

    # Handle neural network models differently
    if 'tc_nn' in model_name:
        recurrent_type = 'lstm' if 'lstm' in model_name else 'gru'
        model = model_class(
            input_dim=X_train.shape[1],
            num_classes=3,  # Force 3-class mode
            recurrent_type=recurrent_type,
            model_name=model_name
        )
    else:
        model = model_class(model_path=full_path, use_gpu=True)

    # Train on RAW features (NO SCALING)
    print(f"Training on RAW features: {X_train.shape}")

    # Neural networks use fit(), others use train()
    if 'tc_nn' in model_name:
        model.fit(X_train, y_train, X_val, y_val, epochs=50)
        # For neural networks, compute metrics manually
        metrics = {
            'train_accuracy': float(np.mean(np.argmax(model.predict_proba(X_train), axis=1) == y_train)),
            'val_accuracy': float(np.mean(np.argmax(model.predict_proba(X_val), axis=1) == y_val))
        }
    else:
        metrics = model.train(X_train, y_train, X_val, y_val)

    # Display results
    print(f"[OK] Training complete:")
    for metric_name, value in metrics.items():
        if isinstance(value, dict):
            continue
        print(f"   {metric_name}: {value:.4f}" if isinstance(value, float) else f"   {metric_name}: {value}")

    # Save model (will NOT save scaler.pkl - purified version)
    if 'tc_nn' in model_name:
        # Neural networks save to .pth file
        os.makedirs(full_path, exist_ok=True)
        model.save(os.path.join(full_path, f'{model_name}.pth'))
    else:
        model.save()

    return model


def train_all_base_models(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, object]:
    """
    Train all 10 base models.

    Returns:
        Dictionary mapping model names to trained model instances
    """
    print("\n" + "="*80)
    print("TRAINING BASE MODELS (8 models)")
    print("="*80)

    trained_models = {}

    for model_name, model_class, model_path in BASE_MODELS:
        # Train model
        model = train_base_model(
            model_name, model_class, model_path,
            X_train, y_train, X_val, y_val
        )
        trained_models[model_name] = model

    print("\n" + "="*80)
    print(f"[OK] All 8 base models trained successfully")
    print("="*80)

    return trained_models


# ============================================================================
# BASE MODEL PREDICTIONS
# ============================================================================

def _detect_model_type(model: object, model_name: str) -> str:
    """
    Determine model type based strictly on available methods.

    Returns:
        "neural" if model has predict_proba_batch,
        "tree"   if model has predict_proba only.

    Raises:
        ValueError if neither required interface is present.
    """
    has_batch = hasattr(model, "predict_proba_batch")
    has_proba = hasattr(model, "predict_proba")

    if has_batch:
        return "neural"
    if has_proba and not has_batch:
        return "tree"

    raise ValueError(
        f"Model '{model_name}' does not implement required prediction methods: "
        f"expected 'predict_proba_batch' for neural or 'predict_proba' for tree."
    )


def _get_probs_for_model(
    model: object,
    model_name: str,
    X: np.ndarray,
    model_type: str,
    batch_size: int = 4096,
    debug_check_normalization: bool = False,
) -> np.ndarray:
    """
    Get batch probabilities for a single model on a given dataset X.

    Invariants:
      - No per-sample prediction.
      - No device transfers here (NumPy→tensor must happen inside predict_proba_batch).
      - Output is always shape (N, 3), dtype float32.

    Args:
        model:      Model instance.
        model_name: Name used for error messages.
        X:          Input features, shape (N, F).
        model_type: "tree" or "neural".
        batch_size: Batch size for neural models.
        debug_check_normalization: If True, verify probs sum to 1 along axis=1.

    Returns:
        probs: np.ndarray of shape (N, 3), dtype float32.
    """
    # Normalize input: convert to numpy array
    X = np.asarray(X)

    # Auto-convert 1D input to 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Validate 2D after normalization
    if X.ndim != 2:
        raise ValueError(
            f"{model_name}: X must be 2D after normalization, got {X.ndim}D"
        )

    N = len(X)

    # Empty edge case: return correctly shaped empty array
    if N == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Tree models: direct predict_proba on full array
    if model_type == "tree":
        probs = model.predict_proba(X)

        # Convert to numpy array
        probs = np.asarray(probs)

        # Validate probs is 2D
        if probs.ndim != 2:
            raise ValueError(
                f"{model_name}: predict_proba returned {probs.ndim}D output; expected 2D (N, 3)"
            )

        # Validate 3 classes
        if probs.shape[1] != 3:
            raise ValueError(
                f"{model_name}: predict_proba returned {probs.shape[1]} classes; expected 3"
            )

    # Neural models: eval() + no_grad() + predict_proba_batch
    elif model_type == "neural":
        import torch

        if not hasattr(model, "model"):
            raise ValueError(
                f"Neural model '{model_name}' is missing 'model' attribute required for eval()/train()."
            )

        base_model = model.model
        if not hasattr(base_model, "eval") or not hasattr(base_model, "train"):
            raise ValueError(
                f"Neural model '{model_name}.model' does not support eval()/train() methods."
            )

        if not hasattr(model, "predict_proba_batch"):
            raise ValueError(
                f"Neural model '{model_name}' must implement 'predict_proba_batch' for batched inference."
            )

        original_training = base_model.training
        base_model.eval()
        try:
            with torch.no_grad():
                # All NumPy→tensor conversion and device handling must occur
                # inside predict_proba_batch, not here.
                probs = model.predict_proba_batch(X, batch_size=batch_size)
        finally:
            # Restore original training state unconditionally
            if original_training:
                base_model.train()

    else:
        raise ValueError(f"Unknown model_type '{model_type}' for model '{model_name}'.")

    # Shape validation: must be (N, 3)
    if probs.shape[0] != N or probs.shape[1] != 3:
        raise ValueError(
            f"{model_name} predictions shape mismatch: "
            f"expected ({N}, 3), got {probs.shape}"
        )

    # Cast to float32 for memory efficiency and consistency
    probs = probs.astype(np.float32)

    # Optional debug-only normalization check
    if debug_check_normalization:
        row_sums = probs.sum(axis=1)
        if not np.isclose(row_sums, 1.0, atol=1e-5).all():
            raise ValueError(
                f"{model_name} produced non-normalized probabilities in debug mode."
            )

    return probs


def generate_base_predictions(
    trained_models: Dict[str, object],
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """
    Generate 3-class predictions from all base models using BATCH prediction.

    Canonical, contamination-free implementation with explicit invariants:
      - No per-sample prediction calls.
      - Strict 3-class output: [p_down, p_neutral, p_up].
      - Deterministic ordering: no shuffling, sorting, or reindexing.
      - Neural models use predict_proba_batch with eval() + no_grad().
      - All outputs are np.float32.

    Args:
        trained_models: Dict of model_name -> trained model instance.
        X_train:        Training features, shape (N_train, F).
        X_val:          Validation features, shape (N_val, F).

    Returns:
        train_predictions: List of length N_train.
            Each element: {model_name: np.ndarray(3,), dtype float32}.
        val_predictions:   List of length N_val.
            Same structure as train_predictions.
    """
    print("\n" + "=" * 80)
    print("GENERATING BASE MODEL PREDICTIONS FOR META-LEARNER (3-class BATCH MODE)")
    print("=" * 80)

    # Step 1: collect full batch predictions per model
    train_probs_dict: Dict[str, np.ndarray] = {}
    val_probs_dict: Dict[str, np.ndarray] = {}

    for model_name, model in trained_models.items():
        print(f"\n[{model_name.upper()}] Batch prediction...")

        # Strict interface-based model type detection
        model_type = _detect_model_type(model, model_name)

        # Training predictions
        print(f"  Training: {len(X_train):,} samples (BATCH, type={model_type})...")
        train_probs = _get_probs_for_model(
            model=model,
            model_name=model_name,
            X=X_train,
            model_type=model_type,
            batch_size=4096,
            debug_check_normalization=False,  # set True only for debug runs
        )
        train_probs_dict[model_name] = train_probs

        # Validation predictions
        print(f"  Validation: {len(X_val):,} samples (BATCH, type={model_type})...")
        val_probs = _get_probs_for_model(
            model=model,
            model_name=model_name,
            X=X_val,
            model_type=model_type,
            batch_size=4096,
            debug_check_normalization=False,  # set True only for debug runs
        )
        val_probs_dict[model_name] = val_probs

        print(f"  [OK] Batch complete: {len(X_train):,} train + {len(X_val):,} val")

    # Step 2: build per-sample structures using list comprehensions
    print("\n[DISTRIBUTION] Building per-sample prediction dictionaries...")

    # Preserve insertion order of models (Python 3.7+)
    model_names = list(trained_models.keys())

    # Training predictions: list of dicts {model_name: probs[i]}
    N_train = len(X_train)
    train_predictions: List[Dict[str, np.ndarray]] = [
        {name: train_probs_dict[name][i] for name in model_names}
        for i in range(N_train)
    ]

    # Validation predictions: same structure
    N_val = len(X_val)
    val_predictions: List[Dict[str, np.ndarray]] = [
        {name: val_probs_dict[name][i] for name in model_names}
        for i in range(N_val)
    ]

    print(f"  [OK] Distributed {len(train_predictions):,} train samples")
    print(f"  [OK] Distributed {len(val_predictions):,} val samples")

    print("\n" + "=" * 80)
    print("[OK] ALL BASE MODEL PREDICTIONS GENERATED (3-class BATCH MODE)")
    print("=" * 80)

    return train_predictions, val_predictions


# ============================================================================
# META-LEARNER TRAINING
# ============================================================================

def train_meta_learner(train_predictions: List[Dict], y_train: np.ndarray,
                       val_predictions: List[Dict], y_val: np.ndarray) -> MetaLearner:
    """
    Train meta-learner on base model predictions.

    Args:
        train_predictions: List of base prediction dicts (training set)
        y_train: True labels (training set)
        val_predictions: List of base prediction dicts (validation set)
        y_val: True labels (validation set)

    Returns:
        Trained MetaLearner instance
    """
    print("\n" + "="*80)
    print("TRAINING META-LEARNER")
    print("="*80)

    # Initialize meta-learner
    meta_path = os.path.join(MODEL_BASE_PATH, 'meta_learner')
    meta_learner = MetaLearner(model_path=meta_path, use_gpu=True)

    # Train on RAW probability vectors (NO SCALING - probabilities already in [0,1])
    print(f"Training on {len(train_predictions):,} base prediction sets")
    print(f"Meta-features per sample: {len(train_predictions[0]) * 3} (8 models × 3 class probs)")
    print(f"  3-class format: [prob_down, prob_neutral, prob_up]")

    metrics = meta_learner.train(
        train_predictions, y_train,
        val_predictions, y_val
    )

    # Display results
    print(f"\n[OK] Meta-learner training complete:")
    print(f"   Train Accuracy: {metrics['train_accuracy']:.4f}")
    if 'val_accuracy' in metrics:
        print(f"   Val Accuracy: {metrics['val_accuracy']:.4f}")

    # Display model importance
    if 'model_importance' in metrics:
        print(f"\n   Model Importance:")
        for model_name, importance in sorted(metrics['model_importance'].items(),
                                             key=lambda x: x[1], reverse=True):
            print(f"      {model_name:20s}: {importance:6.2f}%")

    # Save model (will NOT save scaler.pkl - purified version)
    meta_learner.save()

    return meta_learner


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_all_models(trained_models: Dict[str, object],
                        meta_learner: MetaLearner,
                        X_val: np.ndarray, y_val: np.ndarray):
    """
    Evaluate all models on validation set and display results.
    """
    print("\n" + "="*80)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("="*80)

    results = {}

    # Evaluate base models
    print("\nBase Models:")
    print("-" * 80)
    for model_name, model in trained_models.items():
        metrics = model.evaluate(X_val, y_val)
        accuracy = metrics['accuracy']
        results[model_name] = accuracy
        print(f"  {model_name:25s}: {accuracy*100:6.2f}%")

    # Evaluate meta-learner
    print("\n" + "-" * 80)
    # Generate base predictions using BATCH mode (reuse existing function)
    _, val_predictions = generate_base_predictions(
        trained_models, X_val[:0], X_val  # Empty train, full validation
    )

    meta_metrics = meta_learner.evaluate(val_predictions, y_val)
    meta_accuracy = meta_metrics['accuracy']
    results['meta_learner'] = meta_accuracy

    print(f"  {'Meta-Learner (Ensemble)':25s}: {meta_accuracy*100:6.2f}%")
    print("=" * 80)

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"\n[BEST] Best Model: {best_model[0]} ({best_model[1]*100:.2f}%)")

    return results


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """
    Main training pipeline for purified TurboMode architecture.

    Training Flow:
    1. Load RAW training data (179 features)
    2. Train 10 base models on RAW features
    3. Generate base model predictions (probabilities)
    4. Train meta-learner on RAW probability vectors
    5. Evaluate all models
    """
    print("\n" + "="*80)
    print("TURBOMODE TRAINING PIPELINE - PURIFIED ARCHITECTURE")
    print("Genesis Rebuild - NO StandardScaler, NO Contamination")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Load RAW training data
    X_train, y_train, X_val, y_val = load_training_data()

    # Step 2: Train all base models
    trained_models = train_all_base_models(X_train, y_train, X_val, y_val)

    # Step 3: Generate base model predictions
    train_predictions, val_predictions = generate_base_predictions(
        trained_models, X_train, X_val
    )

    # Step 4: Train meta-learner
    meta_learner = train_meta_learner(
        train_predictions, y_train,
        val_predictions, y_val
    )

    # Step 5: Final evaluation
    results = evaluate_all_models(trained_models, meta_learner, X_val, y_val)

    # Complete
    print("\n" + "="*80)
    print("[OK] TRAINING COMPLETE - ALL 9 MODELS TRAINED (8 base + 1 meta)")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModels saved to: {MODEL_BASE_PATH}")
    print("\n[READY] TurboMode is now ready for predictions!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
