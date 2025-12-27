"""
Neural Network Model for Advanced ML Trading System
Deep learning approach with multiple hidden layers
"""

import numpy as np
from typing import Dict, List, Any, Optional
import os
import json
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class NeuralNetworkModel:
    """
    Multi-Layer Perceptron Neural Network for trading predictions

    Architecture:
    - Input layer: 179 features
    - Hidden layers: 256 -> 128 -> 64 neurons (ReLU activation)
    - Output layer: 3 classes (Buy/Hold/Sell)
    - Dropout for regularization
    - Adam optimizer
    """

    def __init__(self, model_path: str = "backend/data/ml_models/neural_network"):
        """
        Initialize Neural Network model

        Args:
            model_path: Directory to save/load model files
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None  # Neural nets need feature scaling
        self.is_trained = False

        # Hyperparameters
        self.hyperparameters = {
            'hidden_layer_sizes': (256, 128, 64),  # 3 hidden layers
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,  # L2 regularization
            'batch_size': 32,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 20,
            'random_state': 42,
            'verbose': False
        }

        self.feature_names = []
        self.training_metrics = {}

        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: Optional[List[str]] = None, sample_weight: np.ndarray = None) -> Dict[str, Any]:
        """
        Train neural network model

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - 0=Buy, 1=Hold, 2=Sell
            feature_names: Optional list of feature names
            sample_weight: Sample weights for regime-aware training (Module 5)

        Returns:
            Training metrics dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"[TRAIN] Neural Network Model")
        print(f"  Samples: {len(X_train)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")
        print(f"  Training...")

        if sample_weight is not None:
            print(f"  Sample Weights: {len(sample_weight)} (range: {np.min(sample_weight):.2f}x - {np.max(sample_weight):.2f}x)")
            print(f"  Note: sklearn MLPClassifier does not support sample_weight - parameter will be ignored")

        # Store feature names
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Scale features (critical for neural networks)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Initialize model
        self.model = MLPClassifier(**self.hyperparameters)

        # Train
        self.model.fit(X_scaled, y_train)

        # Mark as trained
        self.is_trained = True

        # Calculate training accuracy
        train_score = self.model.score(X_scaled, y_train)

        # Cross-validation
        print("  Cross-validating...")
        cv_scores = cross_val_score(
            MLPClassifier(**self.hyperparameters),
            X_scaled,
            y_train,
            cv=5,
            n_jobs=-1
        )

        print(f"\n  Training Accuracy: {train_score:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Iterations: {self.model.n_iter_}")

        # Store metrics
        self.training_metrics = {
            'train_accuracy': float(train_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_scores': [float(s) for s in cv_scores],
            'n_iterations': int(self.model.n_iter_),
            'n_samples': int(len(X_train)),
            'n_features': int(X_train.shape[1])
        }

        return self.training_metrics

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make prediction for a single sample

        Args:
            features: Dictionary of feature name -> value

        Returns:
            Prediction dictionary with probabilities and class
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Convert features to array in correct order
        X = np.array([features.get(name, 0.0) for name in self.feature_names]).reshape(1, -1)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Apply temperature scaling to calibrate confidence
        # Neural networks tend to be overconfident, so we "soften" the probabilities
        temperature = 1.5  # Higher = less confident (more calibrated)
        calibrated_probs = np.exp(np.log(probabilities + 1e-10) / temperature)
        calibrated_probs = calibrated_probs / calibrated_probs.sum()

        prediction = self.model.predict(X_scaled)[0]

        return {
            'prediction': int(prediction),
            'buy_prob': float(calibrated_probs[0]),
            'hold_prob': float(calibrated_probs[1]),
            'sell_prob': float(calibrated_probs[2]),
            'confidence': float(np.max(probabilities))
        }

    def predict_batch(self, feature_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples

        Args:
            feature_list: List of feature dictionaries

        Returns:
            List of prediction dictionaries
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Convert to array
        X = np.array([[feat.get(name, 0.0) for name in self.feature_names]
                      for feat in feature_list])

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        probabilities = self.model.predict_proba(X_scaled)
        predictions = self.model.predict(X_scaled)

        # Apply temperature scaling to calibrate confidence
        temperature = 1.5  # Same as single prediction
        calibrated_probs = np.exp(np.log(probabilities + 1e-10) / temperature)
        calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)

        results = []
        for i in range(len(feature_list)):
            results.append({
                'prediction': int(predictions[i]),
                'buy_prob': float(calibrated_probs[i][0]),
                'hold_prob': float(calibrated_probs[i][1]),
                'sell_prob': float(calibrated_probs[i][2]),
                'confidence': float(np.max(probabilities[i]))  # Keep original for reference
            })

        return results

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test set

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Scale test features
        X_scaled = self.scaler.transform(X_test)

        # Predictions
        y_pred = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Calculate metrics
        accuracy = self.model.score(X_scaled, y_test)
        confidence = np.max(probabilities, axis=1).mean()

        # Per-class accuracy
        class_accuracies = {}
        for cls in range(3):
            mask = y_test == cls
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == y_test[mask]).mean()
                class_accuracies[cls] = float(class_acc)

        return {
            'accuracy': float(accuracy),
            'mean_confidence': float(confidence),
            'n_samples': int(len(X_test)),
            'class_accuracies': class_accuracies
        }

    def save(self) -> bool:
        """
        Save model to disk

        Returns:
            True if successful
        """
        if not self.is_trained:
            print("[WARNING] Attempting to save untrained model")
            return False

        try:
            # Save model
            model_file = os.path.join(self.model_path, 'neural_network.joblib')
            joblib.dump(self.model, model_file)

            # Save scaler
            scaler_file = os.path.join(self.model_path, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_file)

            # Save metadata
            metadata = {
                'hyperparameters': self.hyperparameters,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'n_features': len(self.feature_names)
            }

            metadata_file = os.path.join(self.model_path, 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"[OK] Model saved to {self.model_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            return False

    def load(self) -> bool:
        """
        Load model from disk

        Returns:
            True if successful
        """
        try:
            # Load model
            model_file = os.path.join(self.model_path, 'neural_network.joblib')
            self.model = joblib.load(model_file)

            # Load scaler
            scaler_file = os.path.join(self.model_path, 'scaler.joblib')
            self.scaler = joblib.load(scaler_file)

            # Load metadata
            metadata_file = os.path.join(self.model_path, 'metadata.json')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.hyperparameters = metadata['hyperparameters']
            self.feature_names = metadata['feature_names']
            self.training_metrics = metadata['training_metrics']

            self.is_trained = True

            print(f"[OK] Model loaded from {self.model_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False


if __name__ == '__main__':
    # Test neural network model
    print("Testing Neural Network Model")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 179

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 3, n_samples)

    X_test = np.random.randn(200, n_features)
    y_test = np.random.randint(0, 3, 200)

    # Train model
    model = NeuralNetworkModel()
    metrics = model.train(X_train, y_train)

    print(f"\nTraining Metrics:")
    print(f"  Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  CV Accuracy: {metrics['cv_mean']:.4f}")

    # Evaluate
    eval_metrics = model.evaluate(X_test, y_test)
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  Confidence: {eval_metrics['mean_confidence']:.4f}")

    # Test prediction
    features = {f'feature_{i}': np.random.randn() for i in range(n_features)}
    pred = model.predict(features)
    print(f"\nSample Prediction:")
    print(f"  Class: {pred['prediction']}")
    print(f"  Buy: {pred['buy_prob']:.4f}")
    print(f"  Hold: {pred['hold_prob']:.4f}")
    print(f"  Sell: {pred['sell_prob']:.4f}")

    # Save/load test
    model.save()
    model2 = NeuralNetworkModel()
    model2.load()

    print("\n[OK] Neural Network model test complete!")
