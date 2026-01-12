"""
Meta-Learner Ensemble Model
Combines predictions from multiple base models using stacking

Architecture:
- Base Models: Random Forest, XGBoost (extensible to LSTM, CNN, Transformer)
- Meta Model: XGBoost GPU (learns optimal combination with non-linear patterns)
- Input: Probabilities from all base models
- Output: Final ensemble prediction with confidence
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
import joblib
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json

try:
    from .random_forest_model import RandomForestModel
    from .xgboost_model import XGBoostModel
except ImportError:
    # When running as standalone script
    from random_forest_model import RandomForestModel
    from xgboost_model import XGBoostModel


class MetaLearner:
    """
    Ensemble model that combines predictions from multiple base models

    Features:
    - Stacking: Meta-model learns optimal weighting
    - Supports 2-6 base models
    - Tracks individual model contributions
    - Model persistence
    """

    def __init__(self, model_path: str = "backend/data/ml_models/meta_learner", use_gpu: bool = True):
        """
        Initialize meta-learner

        Args:
            model_path: Directory to save/load model files
            use_gpu: Whether to use GPU for meta-learner (default True)
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.meta_model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False

        # Base models (will be loaded or trained separately)
        self.base_models = {}

        # Performance tracking
        self.training_metrics = {}
        self.model_weights = {}

        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)

        # Try to load existing model
        self.load()

    def register_base_model(self, name: str, model):
        """
        Register a base model for ensemble

        Args:
            name: Model name (e.g., 'random_forest', 'xgboost')
            model: Trained model instance
        """
        if not model.is_trained:
            print(f"[WARNING] Model {name} is not trained")

        self.base_models[name] = model
        print(f"[OK] Registered base model: {name}")

    def prepare_meta_features(self, base_predictions: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Convert base model predictions to meta-features

        Args:
            base_predictions: Dict mapping model name to prediction dict
                             {
                                 'random_forest': {'buy_prob': 0.6, 'sell_prob': 0.4},
                                 'xgboost': {'buy_prob': 0.7, 'sell_prob': 0.3}
                             }

        Returns:
            1D numpy array of concatenated probabilities (flat vector, shape (N,))
        """
        meta_features = []

        # Consistent order of base models
        for model_name in sorted(base_predictions.keys()):
            pred = base_predictions[model_name]
            # Extract probabilities in consistent order
            meta_features.extend([
                pred.get('buy_prob', 0.50),
                pred.get('sell_prob', 0.50)
            ])

        # Return FLAT vector - no reshape here
        return np.array(meta_features, dtype=np.float32)

    def train(self, X_base_predictions: List[Dict[str, Dict[str, float]]],
              y_true: np.ndarray,
              X_val_base_predictions: List[Dict[str, Dict[str, float]]] = None,
              y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train meta-learner to combine base model predictions

        Args:
            X_base_predictions: List of base prediction dicts for each sample
                               Each item is a dict mapping model name to predictions
            y_true: True labels (n_samples,) - 0=Buy, 1=Sell
            X_val_base_predictions: Validation base predictions (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)

        Returns:
            Training metrics dictionary
        """
        print(f"\n[TRAIN] Meta-Learner Ensemble")
        print(f"  Samples: {len(X_base_predictions)}")
        print(f"  Base Models: {len(self.base_models)}")

        # Convert base predictions to meta-features
        X_meta = []
        for base_preds in X_base_predictions:
            meta_feats = self.prepare_meta_features(base_preds)
            X_meta.append(meta_feats)  # Already flat, no need to flatten

        X_meta = np.array(X_meta)

        # Convert validation predictions to meta-features if provided
        X_val_meta = None
        if X_val_base_predictions is not None and y_val is not None:
            X_val_meta = []
            for base_preds in X_val_base_predictions:
                meta_feats = self.prepare_meta_features(base_preds)
                X_val_meta.append(meta_feats)  # Already flat, no need to flatten
            X_val_meta = np.array(X_val_meta)
            print(f"  Validation samples: {len(X_val_base_predictions)}")

        print(f"  Meta-features: {X_meta.shape[1]}")
        print(f"  Using GPU: {self.use_gpu}")

        # Convert to DataFrame with feature names for sklearn consistency
        meta_feature_names = [f'meta_feat_{i}' for i in range(X_meta.shape[1])]
        X_meta_df = pd.DataFrame(X_meta, columns=meta_feature_names)

        if X_val_meta is not None:
            X_val_meta_df = pd.DataFrame(X_val_meta, columns=meta_feature_names)

        # Initialize meta-model (XGBoost GPU for non-linear stacking)
        # Shallow trees to prevent overfitting on meta-features
        self.meta_model = xgb.XGBClassifier(
            device='cuda' if self.use_gpu else 'cpu',  # XGBoost 3.x GPU
            tree_method='hist',             # XGBoost 3.x: always 'hist', GPU via device
            predictor='gpu_predictor',      # GPU accelerated inference
            n_estimators=100,
            max_depth=3,  # Shallow trees for meta-learning
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=50,       # Early stopping
            random_state=42,
            verbosity=0
        )

        # Train meta-model with validation set if provided
        print("  Training meta-model...")
        if X_val_meta is not None and y_val is not None:
            self.meta_model.fit(X_meta_df, y_true, eval_set=[(X_val_meta_df, y_val)], verbose=False)
        else:
            self.meta_model.fit(X_meta_df, y_true)

        # Calculate metrics
        train_score = self.meta_model.score(X_meta_df, y_true)

        # Analyze model weights (feature importance from XGBoost)
        # Get feature importance scores
        try:
            importance_dict = self.meta_model.get_booster().get_score(importance_type='gain')
            feature_names = []
            for model_name in sorted(self.base_models.keys()):
                feature_names.extend([
                    f'{model_name}_buy_prob',
                    f'{model_name}_sell_prob'
                ])

            # Map feature indices to names
            self.model_weights = {}
            for i, name in enumerate(feature_names):
                feature_key = f'f{i}'
                if feature_key in importance_dict:
                    self.model_weights[name] = float(importance_dict[feature_key])
                else:
                    self.model_weights[name] = 0.0

            # Aggregate by model
            model_importance = {}
            for model_name in self.base_models.keys():
                prefix = f'{model_name}_'
                importance = sum(v for k, v in self.model_weights.items() if k.startswith(prefix))
                model_importance[model_name] = float(importance)

            # Normalize to percentages
            total_importance = sum(model_importance.values())
            if total_importance > 0:
                model_importance = {k: v / total_importance * 100 for k, v in model_importance.items()}
            else:
                # Fallback: equal importance
                model_importance = {k: 100.0 / len(self.base_models) for k in self.base_models.keys()}
        except Exception as e:
            # Fallback: equal importance
            model_importance = {k: 100.0 / len(self.base_models) for k in self.base_models.keys()}

        # Store metrics
        self.training_metrics = {
            'train_accuracy': float(train_score),
            'n_samples': len(X_base_predictions),
            'n_base_models': len(self.base_models),
            'n_meta_features': int(X_meta.shape[1]),
            'model_importance': model_importance,
            'timestamp': datetime.now().isoformat()
        }

        self.is_trained = True

        # Print results
        print(f"\n  Training Accuracy: {train_score:.4f}")
        print(f"\n  Model Importance:")
        for model, importance in model_importance.items():
            print(f"    {model}: {importance:.2f}%")

        # Auto-save after training
        self.save()

        return self.training_metrics

    def predict(self, base_predictions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Make ensemble prediction by combining base model predictions

        Args:
            base_predictions: Dict mapping model name to prediction dict

        Returns:
            Ensemble prediction dictionary
        """
        if not self.is_trained:
            # Fallback: Average base predictions
            return self._simple_average(base_predictions)

        # Prepare meta-features
        X_meta = self.prepare_meta_features(base_predictions)

        # Convert to DataFrame with feature names to avoid sklearn warnings
        X_meta_reshaped = X_meta.reshape(1, -1)
        meta_feature_names = [f'meta_feat_{i}' for i in range(X_meta_reshaped.shape[1])]
        X_meta_df = pd.DataFrame(X_meta_reshaped, columns=meta_feature_names)

        # Get meta-model prediction
        prediction_class = self.meta_model.predict(X_meta_df)[0]
        probabilities = self.meta_model.predict_proba(X_meta_df)[0]

        # Map class to label (handle both 2-class and 3-class systems)
        if len(probabilities) == 3:
            class_labels = ['buy', 'hold', 'sell']
            return {
                'prediction': class_labels[prediction_class],
                'buy_prob': float(probabilities[0]),
                'hold_prob': float(probabilities[1]),
                'sell_prob': float(probabilities[2]),
                'confidence': float(np.max(probabilities)),
                'model': 'meta_learner',
                'base_predictions': base_predictions
            }
        else:
            class_labels = ['buy', 'sell']
            prediction_label = class_labels[prediction_class]

            # Confidence = max probability
            confidence = float(np.max(probabilities))

            # Include base model predictions for transparency
            return {
                'prediction': prediction_label,
                'buy_prob': float(probabilities[0]),
                'sell_prob': float(probabilities[1]),
                'confidence': confidence,
                'model': 'meta_learner',
                'base_predictions': base_predictions
            }

    def predict_with_confidence_masking(self, base_predictions: Dict[str, Dict[str, float]],
                                       buy_threshold: float = 0.65,
                                       sell_threshold: float = 0.75) -> Dict[str, Any]:
        """
        Make ensemble prediction with asymmetric confidence masking

        ASYMMETRIC HOLD THRESHOLDS:
        - BUY requires ≥65% confidence (more aggressive on entries)
        - SELL requires ≥75% confidence (more conservative on exits)
        - Below threshold → HOLD (wait for better opportunity)

        This creates an effective 3-class output (BUY/SELL/HOLD) from
        binary-trained models (BUY/SELL only).

        Args:
            base_predictions: Dict mapping model name to prediction dict
            buy_threshold: Confidence threshold for BUY signals (default: 0.65)
            sell_threshold: Confidence threshold for SELL signals (default: 0.75)

        Returns:
            Ensemble prediction with HOLD masking applied
        """
        # Get base binary prediction
        base_result = self.predict(base_predictions)

        prediction = base_result['prediction']
        buy_prob = base_result['buy_prob']
        sell_prob = base_result['sell_prob']
        confidence = base_result['confidence']

        # Apply asymmetric confidence thresholds
        if prediction == 'buy':
            if confidence >= buy_threshold:
                final_prediction = 'buy'
                reason = f'high_confidence_buy (≥{buy_threshold:.0%})'
            else:
                final_prediction = 'hold'
                reason = f'low_confidence_buy (<{buy_threshold:.0%})'
        else:  # prediction == 'sell'
            if confidence >= sell_threshold:
                final_prediction = 'sell'
                reason = f'high_confidence_sell (≥{sell_threshold:.0%})'
            else:
                final_prediction = 'hold'
                reason = f'low_confidence_sell (<{sell_threshold:.0%})'

        return {
            'prediction': final_prediction,
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'confidence': confidence,
            'model': 'meta_learner_with_hold_masking',
            'base_prediction': prediction,  # Original BUY/SELL before masking
            'masking_reason': reason,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'base_predictions': base_result.get('base_predictions', {})
        }

    def predict_batch(self, base_predictions_list: List[Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
        """
        Make ensemble predictions for multiple samples

        Args:
            base_predictions_list: List of base prediction dicts

        Returns:
            List of ensemble prediction dicts
        """
        if not self.is_trained:
            return [self._simple_average(preds) for preds in base_predictions_list]

        # Prepare meta-features for all samples
        X_meta = []
        for base_preds in base_predictions_list:
            meta_feats = self.prepare_meta_features(base_preds)
            X_meta.append(meta_feats)  # Already flat, no need to flatten

        X_meta = np.array(X_meta)

        # Convert to DataFrame with feature names to avoid sklearn warnings
        meta_feature_names = [f'meta_feat_{i}' for i in range(X_meta.shape[1])]
        X_meta_df = pd.DataFrame(X_meta, columns=meta_feature_names)

        # Predict
        predictions = self.meta_model.predict(X_meta_df)
        probabilities = self.meta_model.predict_proba(X_meta_df)

        # Format results
        class_labels = ['buy', 'sell']
        results = []

        for i in range(len(predictions)):
            pred_class = predictions[i]
            probs = probabilities[i]

            results.append({
                'prediction': class_labels[pred_class],
                'buy_prob': float(probs[0]),
                'sell_prob': float(probs[1]),
                'confidence': float(np.max(probs)),
                'model': 'meta_learner',
                'base_predictions': base_predictions_list[i]
            })

        return results

    def _simple_average(self, base_predictions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Fallback: Simple average of base model probabilities

        Args:
            base_predictions: Dict mapping model name to prediction dict

        Returns:
            Averaged prediction dictionary
        """
        if not base_predictions:
            return {
                'prediction': 'buy',
                'buy_prob': 0.50,
                'sell_prob': 0.50,
                'confidence': 0.0,
                'model': 'meta_learner_untrained'
            }

        # Average probabilities
        buy_probs = [p.get('buy_prob', 0.50) for p in base_predictions.values()]
        sell_probs = [p.get('sell_prob', 0.50) for p in base_predictions.values()]

        avg_buy = np.mean(buy_probs)
        avg_sell = np.mean(sell_probs)

        # Determine prediction
        probs = np.array([avg_buy, avg_sell])
        pred_class = np.argmax(probs)
        class_labels = ['buy', 'sell']

        return {
            'prediction': class_labels[pred_class],
            'buy_prob': float(avg_buy),
            'sell_prob': float(avg_sell),
            'confidence': float(np.max(probs)),
            'model': 'meta_learner_simple_average',
            'base_predictions': base_predictions
        }

    def save(self) -> bool:
        """
        Save meta-model and metadata to disk

        Returns:
            True if successful
        """
        if not self.is_trained:
            print("[WARNING] Cannot save untrained meta-model")
            return False

        try:
            # Save meta-model
            model_file = os.path.join(self.model_path, "meta_model.pkl")
            joblib.dump(self.meta_model, model_file)

            # Save metadata
            metadata = {
                'base_model_names': list(self.base_models.keys()),
                'training_metrics': self.training_metrics,
                'model_weights': self.model_weights,
                'is_trained': self.is_trained,
                'model_version': '1.0.0',
                'saved_at': datetime.now().isoformat()
            }

            metadata_file = os.path.join(self.model_path, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"[OK] Meta-learner saved to {self.model_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to save meta-learner: {e}")
            return False

    def load(self) -> bool:
        """
        Load meta-model and metadata from disk

        Returns:
            True if successful
        """
        try:
            model_file = os.path.join(self.model_path, "meta_model.pkl")
            metadata_file = os.path.join(self.model_path, "metadata.json")

            # Check if files exist
            if not all(os.path.exists(f) for f in [model_file, metadata_file]):
                return False

            # Load meta-model
            self.meta_model = joblib.load(model_file)

            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.training_metrics = metadata['training_metrics']
            self.model_weights = metadata['model_weights']
            self.is_trained = metadata['is_trained']

            print(f"[OK] Meta-learner loaded from {self.model_path}")
            print(f"  Trained on {self.training_metrics.get('n_samples', 0)} samples")
            print(f"  Accuracy: {self.training_metrics.get('train_accuracy', 0):.4f}")

            return True

        except Exception as e:
            print(f"[INFO] No existing meta-learner found ({e})")
            return False

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        n_base = len(self.base_models)
        samples = self.training_metrics.get('n_samples', 0) if self.is_trained else 0
        accuracy = self.training_metrics.get('train_accuracy', 0) if self.is_trained else 0

        return f"<MetaLearner status={status} base_models={n_base} samples={samples} accuracy={accuracy:.3f}>"


if __name__ == '__main__':
    # Test meta-learner
    print("Testing Meta-Learner Ensemble...")

    # Create synthetic base predictions and labels
    n_samples = 500

    # Simulate base model predictions
    np.random.seed(42)

    # Random Forest predictions
    rf_predictions = []
    # XGBoost predictions
    xgb_predictions = []
    # True labels
    y_true = []

    for i in range(n_samples):
        # Simulate RF prediction (slightly noisy)
        rf_buy = max(0, min(1, np.random.rand() * 0.5 + 0.2))
        rf_sell = max(0, min(1, np.random.rand() * 0.3 + 0.1))
        # rf_hold = 1 - rf_buy - rf_sell  # Binary classification

        # Simulate XGB prediction (different noise pattern)
        xgb_buy = max(0, min(1, np.random.rand() * 0.6 + 0.1))
        xgb_sell = max(0, min(1, np.random.rand() * 0.2 + 0.15))
        # xgb_hold = 1 - xgb_buy - xgb_sell  # Binary classification

        rf_predictions.append({
            'buy_prob': rf_buy,
            'sell_prob': rf_sell
        })

        xgb_predictions.append({
            'buy_prob': xgb_buy,
            'sell_prob': xgb_sell
        })

        # True label (based on average with some noise)
        avg_buy = (rf_buy + xgb_buy) / 2
        avg_sell = (rf_sell + xgb_sell) / 2
        avg_hold = (rf_hold + xgb_hold) / 2

        probs = np.array([avg_buy, avg_sell])
        label = np.argmax(probs)
        y_true.append(label)

    y_true = np.array(y_true)

    # Combine predictions
    X_base_predictions = [
        {'random_forest': rf, 'xgboost': xgb}
        for rf, xgb in zip(rf_predictions, xgb_predictions)
    ]

    # Initialize meta-learner
    meta = MetaLearner(model_path="backend/data/ml_models/meta_learner_test")

    # Register dummy base models (for structure)
    class DummyModel:
        def __init__(self):
            self.is_trained = True

    meta.register_base_model('random_forest', DummyModel())
    meta.register_base_model('xgboost', DummyModel())

    # Train meta-learner
    metrics = meta.train(X_base_predictions, y_true)

    # Test prediction
    print("\nTesting single prediction...")
    test_preds = X_base_predictions[0]
    prediction = meta.predict(test_preds)

    print(f"\nPrediction: {prediction['prediction']}")
    print(f"  Buy:  {prediction['buy_prob']:.3f}")
    # print(f"  Hold: {prediction['hold_prob']:.3f}")  # Binary classification
    print(f"  Sell: {prediction['sell_prob']:.3f}")
    print(f"  Confidence: {prediction['confidence']:.3f}")

    # Test batch prediction
    print("\nTesting batch predictions...")
    batch_predictions = meta.predict_batch(X_base_predictions[:10])

    print(f"Predicted {len(batch_predictions)} samples")
    for i, pred in enumerate(batch_predictions[:3]):
        print(f"  {i+1}. {pred['prediction']} (confidence: {pred['confidence']:.3f})")

    print("\n[OK] Meta-learner test complete!")
