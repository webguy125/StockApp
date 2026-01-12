from backend.turbomode.shared.prediction_utils import format_prediction

"""
XGBoost Model for Trading Signals
Gradient boosting optimized for 300+ features
"""

import numpy as np
import pandas as pd
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Run: pip install xgboost")

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json


class XGBoostModel:
    """
    XGBoost classifier for trading signal prediction

    Features:
    - Handles 300+ input features
    - Binary classification: Buy (0), Sell (1)
    - GPU acceleration support (if available)
    - Feature importance tracking
    - Model persistence (save/load)
    """

    def __init__(self, model_path: str = "backend/data/ml_models/xgboost", use_gpu: bool = True):
        """
        Initialize XGBoost model

        Args:
            model_path: Directory to save/load model files
            use_gpu: Whether to use GPU for training (requires CUDA) - Default TRUE for RTX 3070
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # Model hyperparameters (optimized for trading with anti-overfitting)
        self.hyperparameters = {
            'device': 'cuda' if use_gpu else 'cpu',  # XGBoost 3.x GPU device
            'tree_method': 'hist',          # XGBoost 3.x: always 'hist', GPU controlled by device
            'predictor': 'gpu_predictor',   # GPU accelerated inference
            'n_estimators': 300,            # Number of boosting rounds
            'max_depth': 6,                 # Reduced: 8 → 6
            'learning_rate': 0.03,          # Reduced: 0.05 → 0.03 (slower learning)
            'subsample': 0.7,               # Reduced: 0.8 → 0.7 (more row sampling)
            'colsample_bytree': 0.7,        # Reduced: 0.8 → 0.7
            'colsample_bylevel': 0.7,       # Reduced: 0.8 → 0.7
            'gamma': 0.3,                   # Increased: 0.1 → 0.3 (min loss reduction)
            'min_child_weight': 5,          # Increased: 3 → 5 (min samples per leaf)
            'reg_alpha': 0.3,               # Increased L1: 0.1 → 0.3
            'reg_lambda': 2.0,              # Increased L2: 1.0 → 2.0
            'objective': 'binary:logistic',  # Binary classification
            'eval_metric': 'logloss',       # Binary log loss
            'random_state': 42,
            'n_jobs': -1,                   # Use all CPU cores
            'verbosity': 0                  # Suppress warnings
        }

        # Performance metrics
        self.training_metrics = {}
        self.feature_importance = {}

        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)

        # Try to load existing model
        self.load()

    def prepare_features(self, features_dict: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dictionary to numpy array

        Args:
            features_dict: Dictionary of features from FeatureEngineer

        Returns:
            1D numpy array of feature values
        """
        # Remove metadata fields
        exclude_keys = ['feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error']

        feature_values = []
        feature_names = []

        for key, value in sorted(features_dict.items()):
            if key not in exclude_keys:
                # Handle any NaN or infinite values
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    feature_values.append(float(value))
                    feature_names.append(key)

        # Store feature names for consistency
        if not self.feature_names:
            self.feature_names = feature_names

        return np.array(feature_values).reshape(1, -1)

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None,
              validate: bool = False, early_stopping_rounds: int = 50, sample_weight: np.ndarray = None) -> Dict[str, Any]:
        """
        Train XGBoost model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,) - 0=Buy, 1=Sell
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
            validate: Whether to run cross-validation
            early_stopping_rounds: Stop if no improvement for N rounds
            sample_weight: Sample weights for regime-aware training (Module 5)

        Returns:
            Training metrics dictionary
        """
        print(f"\n[TRAIN] XGBoost Model")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")

        # Detect number of classes
        n_classes = len(np.unique(y))
        print(f"  Classes: {n_classes}")
        print(f"  Using GPU: {self.use_gpu}")

        if sample_weight is not None:
            print(f"  Sample Weights: {len(sample_weight)} (range: {np.min(sample_weight):.2f}x - {np.max(sample_weight):.2f}x)")

        # Initialize scaler and model
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X)

        # Use provided validation set if available, otherwise create internal split
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_train = y
            print(f"  Validation samples: {X_val.shape[0]}")
        else:
            # Fallback: Split for early stopping (80/20)
            split_idx = int(0.8 * len(X))
            X_val_scaled = X_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y[:split_idx]
            y_val = y[split_idx:]

            # Split sample weights if provided
            if sample_weight is not None:
                sample_weight = sample_weight[:split_idx]

        # Auto-configure for binary vs multi-class classification
        hyperparams = self.hyperparameters.copy()
        if n_classes > 2:
            # Multi-class classification (e.g., BUY, HOLD, SELL)
            print(f"  Mode: Multi-class classification ({n_classes} classes)")
            hyperparams['objective'] = 'multi:softmax'
            hyperparams['eval_metric'] = 'mlogloss'
            hyperparams['num_class'] = n_classes
        else:
            # Binary classification (BUY vs SELL)
            print(f"  Mode: Binary classification")
            hyperparams['objective'] = 'binary:logistic'
            hyperparams['eval_metric'] = 'logloss'

        # Initialize model with hyperparameters
        self.model = xgb.XGBClassifier(**hyperparams)

        # Train model
        print("  Training...")

        # For XGBoost 3.x sklearn API, simply fit with eval_set
        # Early stopping is enabled automatically when eval_set is provided
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )

        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val_scaled, y_val)

        # Cross-validation (if requested and enough samples)
        cv_scores = []
        if validate and X.shape[0] >= 50:
            print("  Cross-validating...")
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, n_jobs=-1)

        # Feature importance
        importance_dict = self.model.get_booster().get_score(importance_type='gain')

        # Map feature indices to names
        self.feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_key = f'f{i}'
            if feature_key in importance_dict:
                self.feature_importance[name] = float(importance_dict[feature_key])
            else:
                self.feature_importance[name] = 0.0

        # Sort by importance
        top_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        # Store metrics
        self.training_metrics = {
            'train_accuracy': float(train_score),
            'val_accuracy': float(val_score),
            'cv_mean': float(np.mean(cv_scores)) if len(cv_scores) > 0 else 0.0,
            'cv_std': float(np.std(cv_scores)) if len(cv_scores) > 0 else 0.0,
            'best_iteration': int(self.model.best_iteration) if hasattr(self.model, 'best_iteration') else 0,
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_estimators': self.hyperparameters['n_estimators'],
            'top_features': [(name, float(importance)) for name, importance in top_features],
            'timestamp': datetime.now().isoformat()
        }

        self.is_trained = True

        # Print results
        print(f"\n  Training Accuracy: {train_score:.4f}")
        print(f"  Validation Accuracy: {val_score:.4f}")
        print(f"  Best Iteration: {self.model.best_iteration if hasattr(self.model, 'best_iteration') else 'N/A'}")
        if len(cv_scores) > 0:
            print(f"  CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        print(f"\n  Top 5 Features:")
        for i, (name, importance) in enumerate(top_features[:5]):
            print(f"    {i+1}. {name}: {importance:.0f}")

        # Auto-save after training
        self.save()

        return self.training_metrics

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single sample

        Args:
            features_dict: Dictionary of features from FeatureEngineer

        Returns:
            Prediction dictionary with probabilities and confidence
        """
        if not self.is_trained:
            return {
                'prediction': 'buy',
                'buy_prob': 0.50,
                'sell_prob': 0.50,
                'confidence': 0.0,
                'error': 'Model not trained'
            }

        # Prepare features
        X = self.prepare_features(features_dict)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get prediction and probabilities (binary classification)
        prediction_class = int(self.model.predict(X_scaled)[0])
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Use unified prediction layer
        return format_prediction(probabilities, prediction_class, 'xgboost')

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples (batch prediction)

        Args:
            features_list: List of feature dictionaries

        Returns:
            List of prediction dictionaries
        """
        if not self.is_trained:
            return [self.predict(features) for features in features_list]

        # Prepare all features
        X_list = [self.prepare_features(features) for features in features_list]
        X = np.vstack(X_list)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Detect number of classes from probability shape
        n_classes = probabilities.shape[1]

        # Format results (auto-detect binary vs multi-class)
        if n_classes == 2:
            class_labels = ['buy', 'sell']
        else:
            class_labels = ['buy', 'hold', 'sell']

        results = []

        for i in range(len(predictions)):
            pred_class = int(predictions[i])
            probs = probabilities[i]

            # Use unified prediction layer
            result = format_prediction(probs, pred_class, 'xgboost')
            results.append(result)

        return results

    def save(self) -> bool:
        """
        Save model, scaler, and metadata to disk

        Returns:
            True if successful
        """
        if not self.is_trained:
            print("[WARNING] Cannot save untrained model")
            return False

        try:
            # Save model (XGBoost native format)
            model_file = os.path.join(self.model_path, "model.json")
            self.model.save_model(model_file)

            # Save scaler
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            joblib.dump(self.scaler, scaler_file)

            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'hyperparameters': self.hyperparameters,
                'training_metrics': self.training_metrics,
                'is_trained': self.is_trained,
                'use_gpu': self.use_gpu,
                'model_version': '1.0.0',
                'saved_at': datetime.now().isoformat()
            }

            metadata_file = os.path.join(self.model_path, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save feature importance separately
            importance_file = os.path.join(self.model_path, "feature_importance.json")
            with open(importance_file, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)

            print(f"[OK] Model saved to {self.model_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            return False

    def load(self) -> bool:
        """
        Load model, scaler, and metadata from disk

        Returns:
            True if successful
        """
        try:
            model_file = os.path.join(self.model_path, "model.json")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.json")

            # Check if files exist
            if not all(os.path.exists(f) for f in [model_file, scaler_file, metadata_file]):
                return False

            # Load metadata first to get hyperparameters
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.feature_names = metadata['feature_names']
            self.hyperparameters = metadata['hyperparameters']
            self.training_metrics = metadata['training_metrics']
            self.is_trained = metadata['is_trained']
            self.use_gpu = metadata.get('use_gpu', False)

            # Initialize model with hyperparameters
            self.model = xgb.XGBClassifier(**self.hyperparameters)

            # Load model weights
            self.model.load_model(model_file)

            # Load scaler
            self.scaler = joblib.load(scaler_file)

            # Load feature importance if exists
            importance_file = os.path.join(self.model_path, "feature_importance.json")
            if os.path.exists(importance_file):
                with open(importance_file, 'r') as f:
                    self.feature_importance = json.load(f)

            print(f"[OK] Model loaded from {self.model_path}")
            print(f"  Trained on {self.training_metrics.get('n_samples', 0)} samples")
            print(f"  Accuracy: {self.training_metrics.get('train_accuracy', 0):.4f}")

            return True

        except Exception as e:
            print(f"[INFO] No existing model found ({e})")
            return False

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top N most important features

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if not self.is_trained:
            return []

        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_features[:top_n]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test data

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y)

        # Per-class accuracy (binary classification)
        class_labels = ['buy', 'sell']
        class_accuracies = {}

        for i, label in enumerate(class_labels):
            mask = (y == i)
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred[mask] == y[mask])
                class_accuracies[label] = float(class_acc)

        # Confidence statistics
        confidences = np.max(y_proba, axis=1)

        return {
            'accuracy': float(accuracy),
            'class_accuracies': class_accuracies,
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'n_samples': int(len(y)),
            'timestamp': datetime.now().isoformat()
        }

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        samples = self.training_metrics.get('n_samples', 0) if self.is_trained else 0
        accuracy = self.training_metrics.get('train_accuracy', 0) if self.is_trained else 0

        return f"<XGBoostModel status={status} samples={samples} accuracy={accuracy:.3f} gpu={self.use_gpu}>"


if __name__ == '__main__':
    # Test XGBoost model
    print("Testing XGBoost Model...")

    # Create synthetic training data
    n_samples = 1000
    n_features = 100

    np.random.seed(42)
    X_train = np.random.randn(n_samples, n_features)

    # Create labels with some pattern
    # Buy if feature_0 > 0.5, Sell if < -0.5, else Hold
    y_train = np.ones(n_samples, dtype=int)  # Default to Hold (1)
    y_train[X_train[:, 0] > 0.5] = 0  # Buy
    y_train[X_train[:, 0] < -0.5] = 2  # Sell

    # Initialize and train
    model = XGBoostModel(model_path="backend/data/ml_models/xgboost_test", use_gpu=False)
    metrics = model.train(X_train, y_train, validate=True, early_stopping_rounds=20)

    # Test prediction
    print("\nTesting predictions...")

    # Create test sample
    test_features = {f'feature_{i}': float(X_train[0, i]) for i in range(n_features)}
    prediction = model.predict(test_features)

    print(f"\nPrediction: {prediction['prediction']}")
    print(f"  Buy:  {prediction['buy_prob']:.3f}")
    print(f"  Hold: {prediction['hold_prob']:.3f}")
    print(f"  Sell: {prediction['sell_prob']:.3f}")
    print(f"  Confidence: {prediction['confidence']:.3f}")

    # Test batch prediction
    print("\nTesting batch predictions...")
    test_batch = [{f'feature_{i}': float(X_train[j, i]) for i in range(n_features)} for j in range(10)]
    predictions = model.predict_batch(test_batch)

    print(f"Predicted {len(predictions)} samples")
    for i, pred in enumerate(predictions[:3]):
        print(f"  {i+1}. {pred['prediction']} (confidence: {pred['confidence']:.3f})")

    # Evaluate
    print("\nEvaluating on training data...")
    eval_metrics = model.evaluate(X_train, y_train)
    print(f"  Overall Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  Buy Accuracy: {eval_metrics['class_accuracies'].get('buy', 0):.4f}")
    print(f"  Hold Accuracy: {eval_metrics['class_accuracies'].get('hold', 0):.4f}")
    print(f"  Sell Accuracy: {eval_metrics['class_accuracies'].get('sell', 0):.4f}")

    print("\n[OK] XGBoost model test complete!")
