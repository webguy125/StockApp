"""
LightGBM Model for Advanced ML Trading System
Microsoft's gradient boosting framework - fast and efficient
"""

import numpy as np
from typing import Dict, List, Any, Optional
import os
import json
import joblib
import lightgbm as lgb
from sklearn.model_selection import cross_val_score


class LightGBMModel:
    """
    LightGBM Classifier for trading predictions

    Key Features:
    - Leaf-wise tree growth (vs level-wise in XGBoost)
    - Faster training and lower memory usage
    - Better accuracy on large datasets
    - Handles categorical features natively
    """

    def __init__(self, model_path: str = "backend/data/ml_models/lightgbm", use_gpu: bool = False):
        """
        Initialize LightGBM model

        Args:
            model_path: Directory to save/load model files
            use_gpu: Whether to use GPU acceleration (CUDA)
        """
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.use_gpu = use_gpu

        # Hyperparameters optimized for trading with anti-overfitting
        self.hyperparameters = {
            'objective': 'multiclass',
            'num_class': 2,  # Binary classification: buy vs sell
            'boosting_type': 'gbdt',
            'num_leaves': 15,               # Reduced: 31 → 15 (prevent overfitting)
            'max_depth': 6,                 # Limited: -1 → 6
            'learning_rate': 0.03,          # Reduced: 0.05 → 0.03
            'n_estimators': 300,
            'subsample_for_bin': 200000,
            'min_split_gain': 0.1,          # Increased: 0.0 → 0.1
            'min_child_weight': 0.01,       # Increased: 0.001 → 0.01
            'min_child_samples': 30,        # Increased: 20 → 30
            'subsample': 0.7,                # Reduced: 0.8 → 0.7
            'subsample_freq': 1,
            'colsample_bytree': 0.7,         # Reduced: 0.8 → 0.7
            'reg_alpha': 0.5,                # Increased L1: 0.1 → 0.5
            'reg_lambda': 2.0,               # Increased L2: 0.1 → 2.0
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # Enable GPU if requested
        if use_gpu:
            self.hyperparameters['device'] = 'gpu'
            self.hyperparameters['gpu_platform_id'] = 0
            self.hyperparameters['gpu_device_id'] = 0
            print("[LIGHTGBM] GPU acceleration enabled")

        self.feature_names = []
        self.training_metrics = {}

        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: Optional[List[str]] = None,
              validate: bool = False, sample_weight: np.ndarray = None) -> Dict[str, Any]:
        """
        Train LightGBM model

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - 0=Buy, 1=Hold, 2=Sell
            feature_names: Optional list of feature names
            validate: Whether to perform cross-validation
            sample_weight: Sample weights for regime-aware training (Module 5)

        Returns:
            Training metrics dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"[TRAIN] LightGBM Model")
        print(f"  Samples: {len(X_train)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")
        print(f"  Training...")

        if sample_weight is not None:
            print(f"  Sample Weights: {len(sample_weight)} (range: {np.min(sample_weight):.2f}x - {np.max(sample_weight):.2f}x)")

        # Store feature names
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Initialize model
        self.model = lgb.LGBMClassifier(**self.hyperparameters)

        # Train
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

        # Mark as trained
        self.is_trained = True

        # Calculate training accuracy
        train_score = self.model.score(X_train, y_train)

        print(f"\n  Training Accuracy: {train_score:.4f}")

        # Cross-validation (if requested)
        cv_scores = None
        if validate:
            print("  Cross-validating...")
            cv_scores = cross_val_score(
                lgb.LGBMClassifier(**self.hyperparameters),
                X_train,
                y_train,
                cv=5,
                n_jobs=-1
            )
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Get feature importance
        feature_importance = self.model.feature_importances_
        feature_importance_dict = {
            name: float(imp) for name, imp in zip(self.feature_names, feature_importance)
        }

        # Sort by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: -x[1])

        print(f"\n  Top 5 Features:")
        # Print top 5 (without showing the actual feature names to save space)

        # Store metrics
        self.training_metrics = {
            'train_accuracy': float(train_score),
            'n_samples': int(len(X_train)),
            'n_features': int(X_train.shape[1]),
            'feature_importance': dict(sorted_features[:20])  # Top 20
        }

        if cv_scores is not None:
            self.training_metrics['cv_mean'] = float(cv_scores.mean())
            self.training_metrics['cv_std'] = float(cv_scores.std())
            self.training_metrics['cv_scores'] = [float(s) for s in cv_scores]

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

        # Predict
        probabilities = self.model.predict_proba(X)[0]
        prediction = self.model.predict(X)[0]

        return {
            'prediction': int(prediction),
            'buy_prob': float(probabilities[0]),
            'sell_prob': float(probabilities[1]),
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

        # Predict
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)

        results = []
        for i in range(len(feature_list)):
            results.append({
                'prediction': int(predictions[i]),
                'buy_prob': float(probabilities[i][0]),
                'sell_prob': float(probabilities[i][1]),
                'confidence': float(np.max(probabilities[i]))
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

        # Predictions
        y_pred = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        # Calculate metrics
        accuracy = self.model.score(X_test, y_test)
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
            model_file = os.path.join(self.model_path, 'lightgbm.joblib')
            joblib.dump(self.model, model_file)

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
            model_file = os.path.join(self.model_path, 'lightgbm.joblib')
            self.model = joblib.load(model_file)

            # Load metadata
            metadata_file = os.path.join(self.model_path, 'metadata.json')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.hyperparameters = metadata['hyperparameters']
            self.feature_names = metadata['feature_names']
            self.training_metrics = metadata['training_metrics']

            self.is_trained = True

            print(f"[OK] Model loaded from {self.model_path}")
            print(f"  Trained on {self.training_metrics.get('n_samples', 'unknown')} samples")
            print(f"  Accuracy: {self.training_metrics.get('train_accuracy', 0.0):.4f}")

            return True

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
