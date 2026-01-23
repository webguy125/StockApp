"""
XGBoost GBLinear Model - Purified Version (NO StandardScaler)

Uses linear boosting instead of tree-based boosting.
NOTE: Linear models typically benefit from scaling, but we keep this purified
for consistency with the rest of the TurboMode architecture.

FORBIDDEN IMPORTS:
- sklearn.preprocessing.StandardScaler

SAFETY RULE: This file must NEVER import or use StandardScaler.
"""

import numpy as np
import xgboost as xgb
import json
import os
from typing import Dict
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class XGBoostGBLinearModel:
    """
    XGBoost with linear booster for TurboMode.

    CRITICAL: Accepts RAW features only (NO scaling).
    CRITICAL: GBLinear does NOT support inplace_predict. Uses sklearn API.
    """

    def __init__(self, model_path: str = None, use_gpu: bool = True):
        self.model = None
        self.is_trained = False
        self.use_gpu = use_gpu
        self.model_path = model_path or 'backend/data/turbomode_models/xgboost_gblinear'
        self.feature_names = []

        self.hyperparameters = {
            'device': 'cuda' if use_gpu else 'cpu',
            'booster': 'gblinear',
            'n_estimators': 250,
            'learning_rate': 0.06,
            'reg_alpha': 0.15,
            'reg_lambda': 1.2,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': 50
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict 3-class probabilities.

        CRITICAL: GBLinear does NOT support booster.inplace_predict().
        Must use sklearn API: self.model.predict_proba(X)

        Returns:
            np.ndarray: Shape (N, 3) probability matrix
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Convert to numpy array
        X = np.asarray(X)

        # Auto-convert 1D input to 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Validate 2D after reshape
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")

        if X.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        # GBLinear does NOT support inplace_predict. Use sklearn API.
        # With objective='multi:softprob', this returns probabilities directly (not logits)
        raw_preds = self.model.predict_proba(X)

        # Convert to numpy array
        raw_preds = np.asarray(raw_preds)

        # Validate output is 2D
        if raw_preds.ndim != 2:
            raise ValueError(
                f"Model returned {raw_preds.ndim}D output; expected 2D probabilities (N, 3). "
                f"This likely indicates a stale binary classification model. "
                f"Retrain the model with 3-class labels (0=down, 1=neutral, 2=up)."
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

        # Remap class labels to [0, 1, 2] if needed
        classes_source = None
        if hasattr(self.model, "classes_"):
            classes_source = self.model.classes_
        elif hasattr(self, "classes_"):
            classes_source = self.classes_

        if classes_source is not None:
            classes = np.asarray(classes_source)
            if len(classes) != 3:
                raise ValueError(f"Model has {len(classes)} classes, expected exactly 3")

            target_labels = np.array([0, 1, 2], dtype=classes.dtype)

            try:
                idx_map = [np.where(classes == label)[0][0] for label in target_labels]
                raw_preds = raw_preds[:, idx_map]
            except IndexError:
                raise ValueError(
                    f"Class labels {classes.tolist()} do not contain expected {target_labels.tolist()}"
                )

        return raw_preds.astype(np.float32)

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """Train on RAW features (NO scaling)."""
        X_train = X
        y_train = y

        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # FORCE 3-class mode
        n_classes = len(np.unique(y))
        if n_classes != 3:
            raise ValueError(f"Expected 3 classes, got {n_classes}")

        hyperparams = self.hyperparameters.copy()
        hyperparams['objective'] = 'multi:softprob'
        hyperparams['num_class'] = 3
        hyperparams['eval_metric'] = 'mlogloss'

        self.model = xgb.XGBClassifier(**hyperparams)

        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            train_accuracy = self.model.score(X_train, y_train)
            val_accuracy = self.model.score(X_val, y_val)
            metrics = {
                'train_accuracy': float(train_accuracy),
                'val_accuracy': float(val_accuracy)
            }
        else:
            self.model.fit(X_train, y_train, verbose=False)
            train_accuracy = self.model.score(X_train, y_train)
            metrics = {'train_accuracy': float(train_accuracy)}

        self.is_trained = True
        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict 3-class probabilities for a single sample.

        Returns:
            np.ndarray: Shape (3,) - [prob_down, prob_neutral, prob_up]
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Use canonical predict_proba method
        probs = self.predict_proba(features)[0]

        if len(probs) != 3:
            raise ValueError(f"Expected 3-class output, got {len(probs)}")

        return probs  # [prob_down, prob_neutral, prob_up]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Batch predict 3-class probabilities.

        Returns:
            np.ndarray: Shape (N, 3) probability matrix
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Use canonical predict_proba method
        return self.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate on RAW features (NO scaling)."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before evaluation")
        accuracy = self.model.score(X, y)
        return {'accuracy': float(accuracy), 'n_samples': len(y)}

    def save(self) -> None:
        """Save model (NO scaler.pkl)."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")

        os.makedirs(self.model_path, exist_ok=True)
        model_file = os.path.join(self.model_path, 'model.json')
        self.model.save_model(model_file)

        metadata = {
            'is_trained': self.is_trained,
            'use_gpu': self.use_gpu,
            'hyperparameters': self.hyperparameters,
            'feature_names': self.feature_names,
            'model_version': '1.0.0'
        }

        metadata_file = os.path.join(self.model_path, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] XGBoost GBLinear model saved to {self.model_path}")
        print(f"   - model.json: XGBoost GBLinear model")
        print(f"   - metadata.json: Training metadata")
        print(f"   - NO scaler.pkl (purified version)")

    def load(self) -> None:
        """Load model (NO scaler.pkl)."""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")

        model_file = os.path.join(self.model_path, 'model.json')
        metadata_file = os.path.join(self.model_path, 'metadata.json')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.hyperparameters = metadata.get('hyperparameters', self.hyperparameters)
            self.feature_names = metadata.get('feature_names', [])

        self.model = xgb.XGBClassifier(**self.hyperparameters)
        self.model.load_model(model_file)
        self.is_trained = True

        print(f"[OK] XGBoost GBLinear model loaded (NO scaler.pkl)")
