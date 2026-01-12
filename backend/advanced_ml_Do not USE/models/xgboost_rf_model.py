"""
XGBoost GPU Random Forest Model for Trading Signals
GPU-accelerated Random Forest using XGBoost
Replaces: Random Forest (sklearn)
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class XGBoostRFModel:
    """
    XGBoost GPU Random Forest classifier for trading signal prediction

    GPU Advantages:
    - 10-40x faster than sklearn RandomForest
    - GPU parallelization across all trees
    - Better handling of large datasets
    """

    def __init__(self, model_path: str = "backend/data/ml_models/xgboost_rf"):
        self.model_path = model_path
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # GPU hyperparameters (Random Forest mode with anti-overfitting)
        self.hyperparameters = {
            'device': 'cuda',
            'tree_method': 'hist',      # XGBoost 3.x: use 'hist' with device='cuda'
            'n_estimators': 100,
            'max_depth': 6,             # Reduced: 16 → 6 (prevent memorization)
            'learning_rate': 1.0,       # RF uses 1.0 learning rate
            'subsample': 0.8,            # Row sampling like RF
            'colsample_bytree': 0.3,     # Feature sampling like RF
            'colsample_bylevel': 1.0,
            'colsample_bynode': 0.8,
            'num_parallel_tree': 1,      # Standard mode
            'min_child_weight': 5,       # NEW: Minimum samples per leaf
            'reg_lambda': 1.0,           # NEW: L2 regularization
            'reg_alpha': 0.1,            # NEW: L1 regularization
            'random_state': 42,
            'verbosity': 0
        }

        self.training_metrics = {}
        self.feature_importance = {}
        os.makedirs(self.model_path, exist_ok=True)
        self.load()

    def prepare_features(self, features_dict: Dict[str, Any]) -> np.ndarray:
        exclude_keys = ['feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error']
        feature_values = []
        feature_names = []

        for key, value in sorted(features_dict.items()):
            if key not in exclude_keys:
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    feature_values.append(float(value))
                    feature_names.append(key)

        if not self.feature_names:
            self.feature_names = feature_names

        return np.array(feature_values).reshape(1, -1)

    def train(self, X: np.ndarray, y: np.ndarray, validate: bool = True, sample_weight: np.ndarray = None) -> Dict[str, Any]:
        print(f"\n[TRAIN] XGBoost GPU Random Forest Model")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")
        print(f"  Using GPU: True")

        # Initialize scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Initialize XGBoost GPU model (RF mode)
        self.model = xgb.XGBClassifier(**self.hyperparameters)

        # Train on GPU
        print("  Training on GPU...")
        self.model.fit(X_scaled, y, sample_weight=sample_weight)

        # Calculate metrics
        train_score = self.model.score(X_scaled, y)

        # Feature importance (convert to native Python float)
        self.feature_importance = dict(zip(
            self.feature_names if self.feature_names else [f'feature_{i}' for i in range(X.shape[1])],
            [float(x) for x in self.model.feature_importances_]
        ))

        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        # Store metrics
        self.training_metrics = {
            'train_accuracy': float(train_score),
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'timestamp': datetime.now().isoformat(),
            'gpu_enabled': True
        }

        self.is_trained = True

        # Print results
        print(f"\n  Training Accuracy: {train_score:.4f}")
        print(f"\n  Top 5 Features:")
        for feat, importance in top_features:
            print(f"    {feat}: {importance:.4f}")

        # Auto-save
        self.save()

        return self.training_metrics

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_trained:
            return {'prediction': 'buy', 'buy_prob': 0.50, 'sell_prob': 0.50, 'confidence': 0.0, 'model': 'xgboost_rf_untrained'}

        X = self.prepare_features(features)
        X_scaled = self.scaler.transform(X)

        prediction_class = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        class_labels = ['buy', 'sell']
        prediction_label = class_labels[prediction_class]
        confidence = float(np.max(probabilities))

        return {
            'prediction': prediction_label,
            'buy_prob': float(probabilities[0]),
            'sell_prob': float(probabilities[1]),
            'confidence': confidence,
            'model': 'xgboost_rf_gpu'
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.full((X.shape[0], 2), 0.5)  # Binary classification
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch prediction for multiple samples"""
        if not self.is_trained:
            return [self.predict(features) for features in features_list]

        # Prepare all features
        X_list = [self.prepare_features(features) for features in features_list]
        X = np.vstack(X_list)
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Format results (binary classification)
        class_labels = ['buy', 'sell']
        results = []
        for i in range(len(predictions)):
            results.append({
                'prediction': class_labels[predictions[i]],
                'buy_prob': float(probabilities[i][0]),
                'sell_prob': float(probabilities[i][1]),
                'confidence': float(np.max(probabilities[i])),
                'model': 'xgboost_rf_gpu'
            })
        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test data"""
        if not self.is_trained:
            return {'error': 'Model not trained'}

        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)

        accuracy = float(np.mean(y_pred == y))

        class_labels = ['buy', 'hold', 'sell']
        class_accuracies = {}
        for i, label in enumerate(class_labels):
            mask = (y == i)
            if np.sum(mask) > 0:
                class_acc = float(np.mean(y_pred[mask] == i))
                class_accuracies[f'{label}_accuracy'] = class_acc

        return {
            'accuracy': accuracy,
            **class_accuracies,
            'model': 'xgboost_rf_gpu'
        }

    def save(self) -> bool:
        if not self.is_trained:
            return False
        try:
            model_file = os.path.join(self.model_path, "model.json")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.json")

            self.model.save_model(model_file)
            joblib.dump(self.scaler, scaler_file)

            metadata = {
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained,
                'model_version': '1.0.0',
                'saved_at': datetime.now().isoformat()
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"[OK] Model saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            return False

    def load(self) -> bool:
        try:
            model_file = os.path.join(self.model_path, "model.json")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.json")

            if not all(os.path.exists(f) for f in [model_file, scaler_file, metadata_file]):
                return False

            self.model = xgb.XGBClassifier()
            self.model.load_model(model_file)
            self.scaler = joblib.load(scaler_file)

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.feature_names = metadata['feature_names']
            self.training_metrics = metadata['training_metrics']
            self.feature_importance = metadata['feature_importance']
            self.is_trained = metadata['is_trained']

            print(f"[OK] Model loaded from {self.model_path}")
            print(f"  Trained on {self.training_metrics.get('n_samples', 0)} samples")
            print(f"  Accuracy: {self.training_metrics.get('train_accuracy', 0):.4f}")

            return True
        except Exception as e:
            return False
