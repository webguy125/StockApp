"""
XGBoost ExtraTrees Model - Purified Version (NO StandardScaler)

Uses extremely randomized trees (ExtraTrees) as the base learner.
This model is SCALE-INVARIANT and accepts RAW numpy arrays directly.

FORBIDDEN IMPORTS:
- sklearn.preprocessing.StandardScaler

SAFETY RULE: This file must NEVER import or use StandardScaler.
"""

import numpy as np
import xgboost as xgb
import json
import os
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class XGBoostETModel:
    """
    XGBoost with ExtraTrees (extremely randomized trees) for TurboMode.

    CRITICAL: Accepts RAW features only (NO scaling).
    """

    def __init__(self, model_path: str = None, use_gpu: bool = True):
        self.model = None
        self.is_trained = False
        self.use_gpu = use_gpu
        self.model_path = model_path or 'backend/data/turbomode_models/xgboost_et'
        self.feature_names = []

        self.hyperparameters = {
            'device': 'cuda' if use_gpu else 'cpu',
            'tree_method': 'hist',
            'booster': 'gbtree',
            'n_estimators': 400,
            'max_depth': 10,
            'learning_rate': 0.03,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'colsample_bylevel': 0.9,
            'gamma': 0.05,
            'min_child_weight': 2,
            'reg_alpha': 0.05,
            'reg_lambda': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': 50
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray")

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

        # Use fast Booster API instead of sklearn wrapper
        booster = self.model.get_booster()
        raw_preds = booster.inplace_predict(
            X,
            iteration_range=(0, self.model.best_iteration),
            validate_features=False
        )

        # Reshape if needed (XGBoost may return 1D for single sample)
        if raw_preds.ndim == 1:
            raw_preds = raw_preds.reshape(-1, 3)

        if raw_preds.ndim == 1:
            raise ValueError("Model returned 1D output; expected multiclass probabilities (N, 3)")
        if raw_preds.shape[0] != X.shape[0]:
            raise ValueError(f"Sample count mismatch: input {X.shape[0]}, output {raw_preds.shape[0]}")
        if raw_preds.shape[1] != 3:
            raise ValueError(f"Expected exactly 3 classes, got {raw_preds.shape[1]}")

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
        # [OK] Use RAW features
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
                'val_accuracy': float(val_accuracy),
                'n_estimators': self.model.n_estimators
            }
        else:
            self.model.fit(X_train, y_train, verbose=False)
            train_accuracy = self.model.score(X_train, y_train)
            metrics = {'train_accuracy': float(train_accuracy)}

        self.is_trained = True
        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict 3-class probabilities on RAW features."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        X = features.reshape(1, -1) if features.ndim == 1 else features
        probs = self.predict_proba(X)[0]  # Use canonical predict_proba method

        if len(probs) != 3:
            raise ValueError(f"Expected 3-class output, got {len(probs)}")

        return probs  # [prob_down, prob_neutral, prob_up]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch predict on RAW features (NO scaling)."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        return self.predict_proba(X)  # Return (N, 3) probabilities, not class labels

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

        print(f"[OK] XGBoost ET model saved to {self.model_path}")
        print(f"   - model.json: XGBoost ET model")
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
            self.use_gpu = metadata.get('use_gpu', self.use_gpu)

        self.model = xgb.XGBClassifier(**self.hyperparameters)
        self.model.load_model(model_file)
        self.is_trained = True

        print(f"[OK] XGBoost ET model loaded (NO scaler.pkl)")
