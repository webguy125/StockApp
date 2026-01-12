from backend.turbomode.shared.prediction_utils import format_prediction

"""
XGBoost GPU Approx Model for Trading Signals
Uses approximate tree construction algorithm
Balance between speed (hist) and precision (exact)
"""

import numpy as np
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class XGBoostApproxModel:
    """XGBoost Approx GPU classifier - approximate split finding"""

    def __init__(self, model_path: str = "backend/data/ml_models/xgboost_approx"):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")

        self.model_path = model_path
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_trained = False

        # Approx-optimized hyperparameters
        self.hyperparameters = {
            'device': 'cuda',
            'tree_method': 'approx',        # Approximate algorithm
            'predictor': 'gpu_predictor',   # GPU accelerated inference
            'sketch_eps': 0.03,             # Approximation precision (lower = more precise)
            'n_estimators': 300,            # Reduced: 350 → 300
            'max_depth': 7,                 # Can go deeper with approx
            'learning_rate': 0.035,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'colsample_bylevel': 0.75,
            'gamma': 0.2,
            'min_child_weight': 4,
            'reg_alpha': 0.2,
            'reg_lambda': 1.6,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
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

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, validate: bool = False, sample_weight: np.ndarray = None) -> Dict[str, Any]:
        print(f"\n[TRAIN] XGBoost GPU Approx Model")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")

        # Detect number of classes
        n_classes = len(np.unique(y))
        print(f"  Classes: {n_classes}")
        print(f"  Using GPU: True")
        print(f"  Tree Method: approx")

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

        self.model = xgb.XGBClassifier(**hyperparams)

        print("  Training on GPU with approximate method...")
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )

        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val_scaled, y_val)

        cv_scores = []
        if validate and X.shape[0] >= 50:
            print("  Cross-validating...")
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, n_jobs=-1)

        importance_dict = self.model.get_booster().get_score(importance_type='gain')
        self.feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_key = f'f{i}'
            self.feature_importance[name] = float(importance_dict.get(feature_key, 0.0))

        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]

        self.training_metrics = {
            'train_accuracy': float(train_score),
            'val_accuracy': float(val_score),
            'cv_mean': float(np.mean(cv_scores)) if len(cv_scores) > 0 else 0.0,
            'cv_std': float(np.std(cv_scores)) if len(cv_scores) > 0 else 0.0,
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'top_features': [(name, float(importance)) for name, importance in top_features],
            'timestamp': datetime.now().isoformat()
        }

        self.is_trained = True

        print(f"\n  Training Accuracy: {train_score:.4f}")
        print(f"  Validation Accuracy: {val_score:.4f}")
        if len(cv_scores) > 0:
            print(f"  CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        print(f"\n  Top 5 Features:")
        for name, importance in top_features[:5]:
            print(f"    {name}: {importance:.4f}")

        self.save()
        return self.training_metrics

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_trained:
            return {'prediction': 'buy', 'buy_prob': 0.50, 'sell_prob': 0.50, 'confidence': 0.0, 'model': 'xgboost_approx_untrained'}

        X = self.prepare_features(features_dict)
        X_scaled = self.scaler.transform(X)
        prediction_class = int(self.model.predict(X_scaled)[0])
        probabilities = self.model.predict_proba(X_scaled)[0]

        # Use unified prediction layer
        return format_prediction(probabilities, prediction_class, 'xgboost_approx_gpu')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.full((X.shape[0], 2), 0.5)
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
            result = format_prediction(probs, pred_class, 'xgboost_approx_gpu')
            results.append(result)

        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        if not self.is_trained:
            return {'error': 'Model not trained'}
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        accuracy = float(np.mean(y_pred == y))

        class_labels = ['buy', 'sell']
        class_accuracies = {}
        for i, label in enumerate(class_labels):
            mask = (y == i)
            if np.sum(mask) > 0:
                class_acc = float(np.mean(y_pred[mask] == i))
                class_accuracies[f'{label}_accuracy'] = class_acc

        return {
            'accuracy': accuracy,
            **class_accuracies,
            'model': 'xgboost_approx_gpu'
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
                'hyperparameters': self.hyperparameters,
                'training_metrics': self.training_metrics,
                'is_trained': self.is_trained,
                'model_version': '1.0.0',
                'saved_at': datetime.now().isoformat()
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            importance_file = os.path.join(self.model_path, "feature_importance.json")
            with open(importance_file, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)

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

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.feature_names = metadata['feature_names']
            self.hyperparameters = metadata['hyperparameters']
            self.training_metrics = metadata['training_metrics']
            self.is_trained = metadata['is_trained']

            self.model = xgb.XGBClassifier(**self.hyperparameters)
            self.model.load_model(model_file)
            self.scaler = joblib.load(scaler_file)

            importance_file = os.path.join(self.model_path, "feature_importance.json")
            if os.path.exists(importance_file):
                with open(importance_file, 'r') as f:
                    self.feature_importance = json.load(f)

            print(f"[OK] Model loaded from {self.model_path}")
            print(f"  Trained on {self.training_metrics.get('n_samples', 0)} samples")
            print(f"  Accuracy: {self.training_metrics.get('train_accuracy', 0):.4f}")
            return True
        except Exception as e:
            return False
