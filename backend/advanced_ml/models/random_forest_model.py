"""
Random Forest Model for Trading Signals
Ensemble of decision trees optimized for 300+ features
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json


class RandomForestModel:
    """
    Random Forest classifier for trading signal prediction

    Features:
    - Handles 300+ input features
    - 3-class prediction: Buy (0), Hold (1), Sell (2)
    - Cross-validation for hyperparameter tuning
    - Feature importance tracking
    - Model persistence (save/load)
    """

    def __init__(self, model_path: str = "backend/data/ml_models/random_forest"):
        """
        Initialize Random Forest model

        Args:
            model_path: Directory to save/load model files
        """
        self.model_path = model_path
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # Model hyperparameters (baseline configuration - matches Extra Trees)
        self.hyperparameters = {
            'n_estimators': 200,           # Number of trees
            'max_depth': None,              # Unlimited depth (like Extra Trees)
            'min_samples_split': 5,         # Min samples to split node
            'min_samples_leaf': 2,          # Min samples in leaf
            'max_features': 'sqrt',         # Features per split
            'bootstrap': True,              # Bootstrap sampling
            'class_weight': 'balanced',     # Handle imbalanced classes
            'random_state': 42,
            'n_jobs': -1                    # Use all CPU cores
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

    def train(self, X: np.ndarray, y: np.ndarray, validate: bool = True, sample_weight: np.ndarray = None) -> Dict[str, Any]:
        """
        Train Random Forest model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,) - 0=Buy, 1=Hold, 2=Sell
            validate: Whether to run cross-validation
            sample_weight: Sample weights for regime-aware training (Module 5)

        Returns:
            Training metrics dictionary
        """
        print(f"\n[TRAIN] Random Forest Model")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")
        if sample_weight is not None:
            print(f"  Sample Weights: {len(sample_weight)} (range: {np.min(sample_weight):.2f}x - {np.max(sample_weight):.2f}x)")

        # Initialize scaler and model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model with hyperparameters
        self.model = RandomForestClassifier(**self.hyperparameters)

        # Train model (with sample weights if provided)
        print("  Training...")
        self.model.fit(X_scaled, y, sample_weight=sample_weight)

        # Calculate metrics
        train_score = self.model.score(X_scaled, y)
        oob_score = self.model.oob_score_ if self.model.oob_score else 0.0

        # Cross-validation (if requested and enough samples)
        cv_scores = []
        if validate and X.shape[0] >= 50:
            print("  Cross-validating...")
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, n_jobs=-1)

        # Feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))

        # Sort by importance
        top_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        # Store metrics
        self.training_metrics = {
            'train_accuracy': float(train_score),
            'oob_accuracy': float(oob_score),
            'cv_mean': float(np.mean(cv_scores)) if len(cv_scores) > 0 else 0.0,
            'cv_std': float(np.std(cv_scores)) if len(cv_scores) > 0 else 0.0,
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_trees': self.hyperparameters['n_estimators'],
            'top_features': [(name, float(importance)) for name, importance in top_features],
            'timestamp': datetime.now().isoformat()
        }

        self.is_trained = True

        # Print results
        print(f"\n  Training Accuracy: {train_score:.4f}")
        print(f"  OOB Accuracy: {oob_score:.4f}")
        if len(cv_scores) > 0:
            print(f"  CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        print(f"\n  Top 5 Features:")
        for i, (name, importance) in enumerate(top_features[:5]):
            print(f"    {i+1}. {name}: {importance:.4f}")

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
                'prediction': 'hold',
                'buy_prob': 0.33,
                'hold_prob': 0.34,
                'sell_prob': 0.33,
                'confidence': 0.0,
                'error': 'Model not trained'
            }

        # Prepare features
        X = self.prepare_features(features_dict)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get prediction and probabilities
        prediction_class = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Map class to label
        class_labels = ['buy', 'hold', 'sell']
        prediction_label = class_labels[prediction_class]

        # Confidence = max probability
        confidence = float(np.max(probabilities))

        return {
            'prediction': prediction_label,
            'buy_prob': float(probabilities[0]),
            'hold_prob': float(probabilities[1]),
            'sell_prob': float(probabilities[2]),
            'confidence': confidence,
            'model': 'random_forest'
        }

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

        # Format results
        class_labels = ['buy', 'hold', 'sell']
        results = []

        for i in range(len(predictions)):
            pred_class = predictions[i]
            probs = probabilities[i]

            results.append({
                'prediction': class_labels[pred_class],
                'buy_prob': float(probs[0]),
                'hold_prob': float(probs[1]),
                'sell_prob': float(probs[2]),
                'confidence': float(np.max(probs)),
                'model': 'random_forest'
            })

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
            # Save model
            model_file = os.path.join(self.model_path, "model.pkl")
            joblib.dump(self.model, model_file)

            # Save scaler
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            joblib.dump(self.scaler, scaler_file)

            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'hyperparameters': self.hyperparameters,
                'training_metrics': self.training_metrics,
                'is_trained': self.is_trained,
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
            model_file = os.path.join(self.model_path, "model.pkl")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.json")

            # Check if files exist
            if not all(os.path.exists(f) for f in [model_file, scaler_file, metadata_file]):
                return False

            # Load model and scaler
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)

            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.feature_names = metadata['feature_names']
            self.hyperparameters = metadata['hyperparameters']
            self.training_metrics = metadata['training_metrics']
            self.is_trained = metadata['is_trained']

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

        # Per-class accuracy
        class_labels = ['buy', 'hold', 'sell']
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

        return f"<RandomForestModel status={status} samples={samples} accuracy={accuracy:.3f}>"


if __name__ == '__main__':
    # Test Random Forest model
    print("Testing Random Forest Model...")

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
    model = RandomForestModel(model_path="backend/data/ml_models/random_forest_test")
    metrics = model.train(X_train, y_train, validate=True)

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

    print("\n[OK] Random Forest model test complete!")
