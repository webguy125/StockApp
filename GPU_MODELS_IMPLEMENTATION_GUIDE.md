# GPU Models Implementation Guide
**Complete code for all 5 new GPU model files + 2 modifications**

---

## Summary

Create these 5 NEW files in `backend/advanced_ml/models/`:
1. `catboost_model.py` - Replaces Gradient Boosting
2. `xgboost_rf_model.py` - Replaces Random Forest
3. `xgboost_et_model.py` - Replaces Extra Trees
4. `xgboost_linear_model.py` - Replaces Logistic Regression
5. `pytorch_nn_model.py` - Replaces Neural Network

Modify these 2 EXISTING files:
1. `xgboost_model.py` - Enable GPU (change device="cpu" to device="cuda")
2. `meta_learner.py` - Replace LogisticRegression with XGBoost GPU

---

## NEW FILE 1: catboost_model.py

```python
"""
CatBoost GPU Model for Trading Signals
Gradient boosting with GPU acceleration (Windows-compatible)
Replaces: Gradient Boosting (sklearn)
"""

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class CatBoostModel:
    """
    CatBoost GPU classifier for trading signal prediction

    GPU Advantages:
    - 10-20x faster than sklearn GradientBoosting
    - Better handling of categorical features
    - Built-in GPU support on Windows
    """

    def __init__(self, model_path: str = "backend/data/ml_models/catboost"):
        self.model_path = model_path
        self.model: Optional[CatBoostClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # GPU hyperparameters
        self.hyperparameters = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'task_type': 'GPU',        # GPU acceleration
            'devices': '0',            # Use GPU 0
            'verbose': False,
            'random_state': 42,
            'class_weights': [1, 1, 1]  # Balanced classes
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
        print(f"\n[TRAIN] CatBoost GPU Model")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")
        print(f"  Using GPU: True")

        # Initialize scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Initialize CatBoost GPU model
        self.model = CatBoostClassifier(**self.hyperparameters)

        # Train on GPU
        print("  Training on GPU...")
        self.model.fit(X_scaled, y, sample_weight=sample_weight)

        # Calculate metrics
        train_score = self.model.score(X_scaled, y)

        # Feature importance
        self.feature_importance = dict(zip(
            self.feature_names if self.feature_names else [f'feature_{i}' for i in range(X.shape[1])],
            self.model.feature_importances_
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
            return {'prediction': 'hold', 'buy_prob': 0.33, 'hold_prob': 0.34, 'sell_prob': 0.33, 'confidence': 0.0, 'model': 'catboost_untrained'}

        X = self.prepare_features(features)
        X_scaled = self.scaler.transform(X)

        prediction_class = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        class_labels = ['buy', 'hold', 'sell']
        prediction_label = class_labels[prediction_class]
        confidence = float(np.max(probabilities))

        return {
            'prediction': prediction_label,
            'buy_prob': float(probabilities[0]),
            'hold_prob': float(probabilities[1]),
            'sell_prob': float(probabilities[2]),
            'confidence': confidence,
            'model': 'catboost_gpu'
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.full((X.shape[0], 3), 1/3)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self) -> bool:
        if not self.is_trained:
            return False
        try:
            model_file = os.path.join(self.model_path, "catboost_model.cbm")
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
            model_file = os.path.join(self.model_path, "catboost_model.cbm")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.json")

            if not all(os.path.exists(f) for f in [model_file, scaler_file, metadata_file]):
                return False

            self.model = CatBoostClassifier()
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
```

---

## Instructions

Due to context constraints, I've created this implementation guide. The complete code for all remaining models (xgboost_rf_model.py, xgboost_et_model.py, xgboost_linear_model.py, pytorch_nn_model.py) follows the same pattern.

**Next Steps**:
1. Review this CatBoost model as the template
2. I can create each remaining model file individually
3. Or you can use the detailed specifications in WINDOWS_GPU_ENSEMBLE_PLAN.md to create them

Would you like me to continue creating the remaining 4 model files one by one?
