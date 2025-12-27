"""
Simple Trading Model
Random Forest classifier for signal prediction
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime


class SimpleTradingModel:
    """
    Simple ML model using Random Forest for classification

    Predicts: BUY, SELL, or HOLD based on analyzer features
    """

    def __init__(self, model_path: str = "backend/data/ml_models/trading_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.feature_importance = {}

        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

        # Try to load existing model
        self._load_model()

    def _ensure_model_dir(self):
        """Ensure model directory exists"""
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

    def _load_model(self):
        """Load pre-trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.feature_importance = saved_data.get('feature_importance', {})
                    self.is_trained = True
                print(f"✅ Loaded pre-trained model from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                self.is_trained = False

    def save_model(self):
        """Save trained model to disk"""
        self._ensure_model_dir()

        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_importance': self.feature_importance,
                    'trained_at': datetime.now().isoformat()
                }, f)
            print(f"✅ Model saved to {self.model_path}")
        except Exception as e:
            print(f"❌ Could not save model: {e}")

    def train(self, features: np.ndarray, labels: np.ndarray, validation_split: float = 0.2):
        """
        Train the model

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Target labels (n_samples,) - 0=SELL, 1=HOLD, 2=BUY
            validation_split: Fraction of data for validation

        Returns:
            Dict with training metrics
        """
        if len(features) < 10:
            print("⚠️ Insufficient training data (need at least 10 samples)")
            return {'error': 'Insufficient data'}

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=validation_split,
            random_state=42
        )

        # Train model
        print(f"🎓 Training model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)

        # Validate
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)

        self.is_trained = True

        # Save model
        self.save_model()

        print(f"✅ Training complete - Train: {train_score:.2%}, Val: {val_score:.2%}")

        return {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'train_samples': len(X_train),
            'val_samples': len(X_val)
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,) - 0=SELL, 1=HOLD, 2=BUY
        """
        if not self.is_trained:
            # Return HOLD for all if not trained
            return np.ones(len(features), dtype=int)

        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Probabilities (n_samples, 3) for [SELL, HOLD, BUY]
        """
        if not self.is_trained:
            # Return neutral probabilities if not trained
            n_samples = len(features)
            return np.array([[0.33, 0.34, 0.33]] * n_samples)

        return self.model.predict_proba(features)

    def get_signal(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Get trading signal from features

        Args:
            features: Single feature vector (n_features,)

        Returns:
            {
                'prediction': str ('BUY', 'HOLD', 'SELL'),
                'confidence': float (0.0 to 1.0),
                'probabilities': dict {'buy': float, 'hold': float, 'sell': float}
            }
        """
        # Reshape for prediction
        features_2d = features.reshape(1, -1)

        # Get probabilities
        proba = self.predict_proba(features_2d)[0]

        # Map to labels
        labels = ['SELL', 'HOLD', 'BUY']
        prediction_idx = np.argmax(proba)
        prediction = labels[prediction_idx]
        confidence = proba[prediction_idx]

        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': {
                'sell': float(proba[0]),
                'hold': float(proba[1]),
                'buy': float(proba[2])
            }
        }

    def get_feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}

        importances = self.model.feature_importances_

        if feature_names and len(feature_names) == len(importances):
            return dict(zip(feature_names, importances))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        return f"<SimpleTradingModel status={status} estimators={self.model.n_estimators}>"
