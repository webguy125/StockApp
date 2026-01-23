"""
Random Forest Model - Purified Version (NO StandardScaler)

Scikit-learn RandomForestClassifier for TurboMode.
This model is SCALE-INVARIANT and accepts RAW numpy arrays directly.

FORBIDDEN IMPORTS:
- sklearn.preprocessing.StandardScaler

SAFETY RULE: This file must NEVER import or use StandardScaler.
"""

import numpy as np
import json
import os
import pickle
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class RandomForestModel:
    """
    Random Forest classifier for TurboMode.

    CRITICAL: This model accepts RAW features (NO scaling/normalization).
    Tree-based models are scale-invariant.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize Random Forest model.

        Args:
            model_path: Path to save/load model artifacts
        """
        self.model = None
        self.is_trained = False
        self.model_path = model_path or 'backend/data/turbomode_models/random_forest'
        self.feature_names = []

        # NO self.scaler initialization - FORBIDDEN

        self.hyperparameters = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'n_jobs': -1,
            'random_state': 42,
            'verbose': 0
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features (1D or 2D numpy array)

        Returns:
            (N, 3) probability matrix
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Normalize input
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Predict probabilities
        probs = self.model.predict_proba(X)

        # Validate 3-class output
        if probs.shape[1] != 3:
            raise ValueError(
                f"Expected exactly 3 classes, got {probs.shape[1]}. "
                f"Retrain the model with 3-class labels (0=down, 1=neutral, 2=up)."
            )

        return probs.astype(np.float32)

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """
        Train Random Forest model on RAW features.

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

        # Store feature names
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # FORCE 3-class mode (down=0, neutral=1, up=2)
        n_classes = len(np.unique(y))
        if n_classes != 3:
            raise ValueError(f"Expected 3 classes, got {n_classes}. Labels must be 0=down, 1=neutral, 2=up")

        # Initialize model
        self.model = RandomForestClassifier(**self.hyperparameters)

        # Train model
        self.model.fit(X_train, y_train)

        # Calculate accuracies
        train_accuracy = self.model.score(X_train, y_train)

        metrics = {
            'train_accuracy': float(train_accuracy),
            'n_estimators': self.model.n_estimators
        }

        if X_val is not None and y_val is not None:
            val_accuracy = self.model.score(X_val, y_val)
            metrics['val_accuracy'] = float(val_accuracy)

            if hasattr(self.model, 'oob_score_'):
                metrics['oob_score'] = float(self.model.oob_score_)

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
        probs = self.predict_proba(features)
        if probs.shape[0] != 1:
            raise ValueError(f"Expected single sample, got {probs.shape[0]} samples")
        return probs[0]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for batch of samples.

        Args:
            X: RAW feature matrix (NO scaling needed)

        Returns:
            (N, 3) probability matrix
        """
        return self.predict_proba(X)

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

        # Calculate per-class accuracy
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
        - random_forest_model.pkl: Scikit-learn RandomForest model
        - metadata.json: Training metadata and hyperparameters

        DOES NOT SAVE:
        - scaler.pkl (FORBIDDEN - not used in purified version)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")

        # Create directory if needed
        os.makedirs(self.model_path, exist_ok=True)

        # Save RandomForest model
        model_file = os.path.join(self.model_path, 'random_forest_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)

        # Save metadata
        metadata = {
            'is_trained': self.is_trained,
            'hyperparameters': self.hyperparameters,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'model_version': '1.0.0'
        }

        metadata_file = os.path.join(self.model_path, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] Random Forest model saved to {self.model_path}")
        print(f"   - random_forest_model.pkl: Random Forest model")
        print(f"   - metadata.json: Training metadata")
        print(f"   - NO scaler.pkl (purified version)")

    def load(self) -> None:
        """
        Load model from disk.

        Loads:
        - random_forest_model.pkl: RandomForest model
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

        # Load RandomForest model
        model_file = os.path.join(self.model_path, 'random_forest_model.pkl')
        if not os.path.exists(model_file):
            raise ValueError(f"Model file not found: {model_file}")

        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        self.is_trained = True
        print(f"[OK] Random Forest model loaded from {self.model_path}")
        print(f"   - random_forest_model.pkl: Random Forest model loaded")
        print(f"   - NO scaler.pkl loaded (purified version)")
