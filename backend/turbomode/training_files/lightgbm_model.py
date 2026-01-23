"""
LightGBM Model - Purified Version (NO StandardScaler)

LightGBM with GPU acceleration for TurboMode.
This model is SCALE-INVARIANT and accepts RAW numpy arrays directly.

FORBIDDEN IMPORTS:
- sklearn.preprocessing.StandardScaler

SAFETY RULE: This file must NEVER import or use StandardScaler.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import os
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class TurboLightGBMWrapper:
    """
    Canonical wrapper for LightGBM Booster that enforces strict TurboMode contracts.

    This wrapper:
    - Normalizes input shapes (auto-converts 1D to 2D)
    - Enforces 3-class output validation
    - Raises clear errors for malformed inputs or stale models
    - Does not change prediction values, only shapes and validation
    """

    def __init__(self, booster: lgb.Booster):
        """
        Initialize wrapper around a LightGBM Booster.

        Args:
            booster: Trained LightGBM Booster instance
        """
        self.booster = booster

    def _normalize_input(self, X):
        """
        Normalize input to 2D numpy array.

        Args:
            X: Input array (1D or 2D)

        Returns:
            2D numpy array

        Raises:
            ValueError: If X cannot be converted to 2D
        """
        # Convert to numpy array
        X = np.asarray(X)

        # Auto-convert 1D input to 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Validate 2D after normalization
        if X.ndim != 2:
            raise ValueError(f"X must be 2D after normalization, got {X.ndim}D")

        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities with strict validation.

        Args:
            X: Input features (1D or 2D numpy array)

        Returns:
            (N, 3) probability matrix

        Raises:
            ValueError: If output is not 2D or does not have 3 classes
        """
        # Normalize input
        X = self._normalize_input(X)

        if X.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Call underlying Booster
        raw_preds = self.booster.predict(X, raw_score=False)

        # Convert to numpy array
        raw_preds = np.asarray(raw_preds)

        # Validate output is 2D
        if raw_preds.ndim == 1:
            raise ValueError(
                "Model returned 1D output; expected multiclass probabilities (N, 3). "
                "This likely indicates a stale binary classification model. "
                "Retrain the model with 3-class labels (0=down, 1=neutral, 2=up)."
            )

        # Validate sample count
        if raw_preds.shape[0] != X.shape[0]:
            raise ValueError(f"Sample count mismatch: input {X.shape[0]}, output {raw_preds.shape[0]}")

        # Validate 3-class output
        if raw_preds.shape[1] != 3:
            raise ValueError(
                f"Expected exactly 3 classes, got {raw_preds.shape[1]}. "
                f"This indicates a stale model trained with {raw_preds.shape[1]} classes. "
                f"Retrain the model with 3-class labels (0=down, 1=neutral, 2=up)."
            )

        return raw_preds.astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for single sample.

        Args:
            features: 1D feature vector or 2D array

        Returns:
            1D probability vector [prob_down, prob_neutral, prob_up]
        """
        probs = self.predict_proba(features)

        if probs.shape[0] != 1:
            raise ValueError(f"Expected single sample, got {probs.shape[0]} samples")

        return probs[0]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for batch of samples.

        Args:
            X: 2D feature matrix

        Returns:
            (N, 3) probability matrix
        """
        return self.predict_proba(X)


class LightGBMModel:
    """
    LightGBM GPU-accelerated classifier for TurboMode.

    CRITICAL: This model accepts RAW features (NO scaling/normalization).
    Tree-based models are scale-invariant.
    """

    def __init__(self, model_path: str = None, use_gpu: bool = True):
        """
        Initialize LightGBM model.

        Args:
            model_path: Path to save/load model artifacts
            use_gpu: Whether to use GPU acceleration
        """
        self.model = None
        self.wrapper = None
        self.is_trained = False
        self.use_gpu = use_gpu
        self.model_path = model_path or 'backend/data/turbomode_models/lightgbm'
        self.feature_names = []

        # NO self.scaler initialization - FORBIDDEN

        self.hyperparameters = {
            'device': 'gpu' if use_gpu else 'cpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'boosting_type': 'gbdt',
            'objective': 'multiclass',  # FORCE 3-class mode
            'num_class': 3,             # Required for multiclass
            'metric': 'multi_logloss',  # 3-class metric
            'n_estimators': 400,
            'num_leaves': 63,
            'max_depth': 9,
            'learning_rate': 0.04,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.85,
            'min_child_samples': 25,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,
            'reg_lambda': 0.9,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Use wrapper for all predictions."""
        if not self.is_trained or self.wrapper is None:
            raise ValueError("Model must be trained before prediction")
        return self.wrapper.predict_proba(X)

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """
        Train LightGBM model on RAW features.

        Args:
            X: Training features (RAW numpy array, NO scaling needed)
            y: Training labels
            X_val: Validation features (RAW)
            y_val: Validation labels

        Returns:
            Dictionary with training metrics
        """
        # [OK] Use RAW features directly (NO SCALING)
        X_train = X
        y_train = y

        # Store feature names for later use
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # FORCE 3-class mode (down=0, neutral=1, up=2)
        n_classes = len(np.unique(y))
        if n_classes != 3:
            raise ValueError(f"Expected 3 classes, got {n_classes}. Labels must be 0=down, 1=neutral, 2=up")

        # Configure for 3-class classification
        hyperparams = self.hyperparameters.copy()
        hyperparams['objective'] = 'multiclass'
        hyperparams['num_class'] = 3
        hyperparams['metric'] = 'multi_logloss'

        # Initialize model
        self.model = lgb.LGBMClassifier(**hyperparams)

        # Train with validation set if provided
        if X_val is not None and y_val is not None:
            # Use RAW validation features (NO scaling)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            # Calculate accuracies
            train_accuracy = self.model.score(X_train, y_train)
            val_accuracy = self.model.score(X_val, y_val)

            metrics = {
                'train_accuracy': float(train_accuracy),
                'val_accuracy': float(val_accuracy),
                'n_estimators': self.model.n_estimators,
                'best_iteration': self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else None
            }
        else:
            # Train without validation
            self.model.fit(X_train, y_train)
            train_accuracy = self.model.score(X_train, y_train)

            metrics = {
                'train_accuracy': float(train_accuracy),
                'n_estimators': self.model.n_estimators
            }

        # Wrap the trained booster
        self.wrapper = TurboLightGBMWrapper(self.model.booster_)
        self.is_trained = True
        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict 3-class probabilities on RAW features.

        Args:
            features: RAW feature vector (NO scaling needed)

        Returns:
            3-class probability array: [prob_down, prob_neutral, prob_up]
        """
        if not self.is_trained or self.wrapper is None:
            raise ValueError("Model must be trained before prediction")
        return self.wrapper.predict(features)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for batch of samples.

        Args:
            X: RAW feature matrix (NO scaling needed)

        Returns:
            (N, 3) probability matrix
        """
        if not self.is_trained or self.wrapper is None:
            raise ValueError("Model must be trained before prediction")
        return self.wrapper.predict_batch(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X: Test features (RAW, NO scaling needed)
            y: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before evaluation")

        # [OK] Use RAW features directly (NO SCALING)
        accuracy = self.model.score(X, y)

        # Get predictions
        y_pred = self.model.predict(X)

        # Calculate per-class accuracy if multi-class
        n_classes = len(np.unique(y))
        if n_classes > 2:
            class_accuracies = {}
            for cls in range(n_classes):
                mask = y == cls
                if mask.sum() > 0:
                    class_acc = (y_pred[mask] == cls).mean()
                    class_accuracies[f'class_{cls}_accuracy'] = float(class_acc)

            return {
                'accuracy': float(accuracy),
                'n_samples': len(y),
                **class_accuracies
            }
        else:
            return {
                'accuracy': float(accuracy),
                'n_samples': len(y)
            }

    def save(self) -> None:
        """
        Save model to disk.

        Saves:
        - lightgbm_model.txt: LightGBM model in text format
        - metadata.json: Training metadata and hyperparameters

        DOES NOT SAVE:
        - scaler.pkl (FORBIDDEN - not used in purified version)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")

        # Create directory if needed
        os.makedirs(self.model_path, exist_ok=True)

        # Save LightGBM model
        model_file = os.path.join(self.model_path, 'lightgbm_model.txt')
        self.model.booster_.save_model(model_file)

        # Save metadata
        metadata = {
            'is_trained': self.is_trained,
            'use_gpu': self.use_gpu,
            'hyperparameters': self.hyperparameters,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'model_version': '1.0.0'
        }

        metadata_file = os.path.join(self.model_path, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] LightGBM model saved to {self.model_path}")
        print(f"   - lightgbm_model.txt: LightGBM model")
        print(f"   - metadata.json: Training metadata")
        print(f"   - NO scaler.pkl (purified version)")

    def load(self) -> None:
        """
        Load model from disk.

        Loads:
        - lightgbm_model.txt: LightGBM model
        - metadata.json: Training metadata

        DOES NOT LOAD:
        - scaler.pkl (FORBIDDEN - not used in purified version)
        """
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")

        # Load metadata
        metadata_file = os.path.join(self.model_path, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.hyperparameters = metadata.get('hyperparameters', self.hyperparameters)
            self.feature_names = metadata.get('feature_names', [])
            self.use_gpu = metadata.get('use_gpu', self.use_gpu)

        # Load LightGBM model
        model_file = os.path.join(self.model_path, 'lightgbm_model.txt')
        if not os.path.exists(model_file):
            raise ValueError(f"Model file not found: {model_file}")

        # Load booster directly
        booster = lgb.Booster(model_file=model_file)

        # Create model instance and attach booster
        self.model = lgb.LGBMClassifier(**self.hyperparameters)
        self.model._Booster = booster
        self.model._n_features = booster.num_feature()
        self.model._n_classes = 3
        self.model.fitted_ = True

        # Wrap the loaded booster
        self.wrapper = TurboLightGBMWrapper(booster)

        self.is_trained = True
        print(f"[OK] LightGBM model loaded from {self.model_path}")
        print(f"   - lightgbm_model.txt: LightGBM model loaded")
        print(f"   - NO scaler.pkl loaded (purified version)")
