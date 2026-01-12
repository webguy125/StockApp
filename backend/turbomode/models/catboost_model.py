"""
CatBoost Model - Purified Version (NO StandardScaler)

CatBoost with GPU acceleration and class weight balancing.
This model is SCALE-INVARIANT and accepts RAW numpy arrays directly.

FORBIDDEN IMPORTS:
- sklearn.preprocessing.StandardScaler

SAFETY RULE: This file must NEVER import or use StandardScaler.
"""

import numpy as np
from catboost import CatBoostClassifier, Pool
import json
import os
from typing import Dict
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class CatBoostModel:
    """
    CatBoost GPU-accelerated classifier for TurboMode.

    CRITICAL: Accepts RAW features only (NO scaling).
    CatBoost handles feature processing internally.
    """

    def __init__(self, model_path: str = None, use_gpu: bool = True):
        self.model = None
        self.is_trained = False
        self.use_gpu = use_gpu
        self.model_path = model_path or 'backend/data/turbomode_models/catboost'
        self.feature_names = []

        self.hyperparameters = {
            'task_type': 'GPU' if use_gpu else 'CPU',
            'devices': '0',
            'loss_function': 'MultiClass',  # FORCE 3-class mode
            'classes_count': 3,             # Required for MultiClass
            'eval_metric': 'MultiClass',    # 3-class metric
            'iterations': 500,
            'depth': 8,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3.0,
            'bagging_temperature': 1.0,
            'random_strength': 1.0,
            'border_count': 254,
            'grow_policy': 'SymmetricTree',
            'bootstrap_type': 'Bayesian',
            # NOTE: subsample removed - incompatible with Bayesian bootstrap
            'sampling_frequency': 'PerTree',
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
            'early_stopping_rounds': 50
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if X.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        raw_preds = self.model.predict(X, prediction_type="Probability")

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
        """
        Train CatBoost on RAW features (NO scaling).
        CatBoost handles feature processing internally.
        """
        # [OK] Use RAW features directly (NO SCALING)
        X_train = X
        y_train = y

        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # FORCE 3-class mode (down=0, neutral=1, up=2)
        n_classes = len(np.unique(y))
        if n_classes != 3:
            raise ValueError(f"Expected 3 classes, got {n_classes}. Labels must be 0=down, 1=neutral, 2=up")

        # Configure for 3-class classification
        hyperparams = self.hyperparameters.copy()
        hyperparams['loss_function'] = 'MultiClass'
        hyperparams['classes_count'] = 3
        hyperparams['eval_metric'] = 'MultiClass'

        # Compute balanced class weights for imbalanced datasets
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        hyperparams['class_weights'] = class_weights.tolist()

        # Create CatBoost Pool for training (uses RAW features)
        train_pool = Pool(
            data=X_train,
            label=y_train,
            feature_names=self.feature_names
        )

        # Create validation pool if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(
                data=X_val,
                label=y_val,
                feature_names=self.feature_names
            )

        # Initialize and train model
        self.model = CatBoostClassifier(**hyperparams)

        if eval_set is not None:
            self.model.fit(
                train_pool,
                eval_set=eval_set,
                verbose=False
            )

            # Calculate accuracies
            train_accuracy = self.model.score(X_train, y_train)
            val_accuracy = self.model.score(X_val, y_val)

            metrics = {
                'train_accuracy': float(train_accuracy),
                'val_accuracy': float(val_accuracy),
                'n_trees': self.model.tree_count_
            }
        else:
            self.model.fit(train_pool, verbose=False)
            train_accuracy = self.model.score(X_train, y_train)

            metrics = {
                'train_accuracy': float(train_accuracy),
                'n_trees': self.model.tree_count_
            }

        self.is_trained = True
        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict 3-class probabilities on RAW features.

        Args:
            features: RAW feature vector

        Returns:
            3-class probability array: [prob_down, prob_neutral, prob_up]
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        # [OK] Use RAW features directly (NO SCALING)
        X = features.reshape(1, -1) if features.ndim == 1 else features

        # Get 3-class probabilities
        probs = self.model.predict_proba(X)[0]

        if len(probs) != 3:
            raise ValueError(f"Expected 3-class output, got {len(probs)} classes")

        return probs  # [prob_down, prob_neutral, prob_up]

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for batch of samples.

        Args:
            X: RAW feature matrix (NO scaling needed)

        Returns:
            Array of predicted class labels
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before prediction")

        # [OK] Use RAW features directly (NO SCALING)
        return self.model.predict(X)

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
        - catboost_model.cbm: CatBoost model binary
        - metadata.json: Training metadata and hyperparameters

        DOES NOT SAVE:
        - scaler.pkl (FORBIDDEN - not used in purified version)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before saving")

        # Create directory if needed
        os.makedirs(self.model_path, exist_ok=True)

        # Save CatBoost model
        model_file = os.path.join(self.model_path, 'catboost_model.cbm')
        self.model.save_model(model_file)

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

        print(f"[OK] CatBoost model saved to {self.model_path}")
        print(f"   - catboost_model.cbm: CatBoost model")
        print(f"   - metadata.json: Training metadata")
        print(f"   - NO scaler.pkl (purified version)")

    def load(self) -> None:
        """
        Load model from disk.

        Loads:
        - catboost_model.cbm: CatBoost model
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

        # Load CatBoost model
        model_file = os.path.join(self.model_path, 'catboost_model.cbm')
        if not os.path.exists(model_file):
            raise ValueError(f"Model file not found: {model_file}")

        self.model = CatBoostClassifier()
        self.model.load_model(model_file)

        self.is_trained = True
        print(f"[OK] CatBoost model loaded from {self.model_path}")
        print(f"   - catboost_model.cbm: CatBoost model loaded")
        print(f"   - NO scaler.pkl loaded (purified version)")
