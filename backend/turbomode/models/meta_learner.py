"""
Meta-Learner (Stacking Ensemble) - Purified & Unified Architecture

Combines 3-class probability outputs from 8 base models using LightGBM.

CRITICAL DESIGN - PERFECT TRAINING/INFERENCE SYMMETRY:
- Input: 24-dimensional feature vector (8 models × 3 classes)
- Each base model contributes: [prob_down, prob_neutral, prob_up]
- Order is FIXED and DETERMINISTIC
- NO StandardScaler, NO transforms, NO preprocessing

FORBIDDEN IMPORTS:
- sklearn.preprocessing.StandardScaler

Base Model Order (MUST MATCH TRAINING):
1. xgboost
2. xgboost_et
3. xgboost_hist
4. xgboost_dart
5. xgboost_gblinear
6. xgboost_approx
7. lightgbm
8. catboost

NOTE: Neural networks (tc_nn_lstm, tc_nn_gru) disabled due to architectural mismatch.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import json
import os
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class MetaLearner:
    """
    Meta-learner that combines 3-class predictions from 8 base models.

    Architecture:
    - Input: 24 features (8 models × 3 class probabilities)
    - Meta-model: LightGBM classifier
    - Output: 3-class prediction (down/neutral/up)

    Feature Vector Format (FIXED ORDER):
    [
        xgboost_prob_down, xgboost_prob_neutral, xgboost_prob_up,
        xgboost_et_prob_down, xgboost_et_prob_neutral, xgboost_et_prob_up,
        xgboost_hist_prob_down, xgboost_hist_prob_neutral, xgboost_hist_prob_up,
        xgboost_dart_prob_down, xgboost_dart_prob_neutral, xgboost_dart_prob_up,
        xgboost_gblinear_prob_down, xgboost_gblinear_prob_neutral, xgboost_gblinear_prob_up,
        xgboost_approx_prob_down, xgboost_approx_prob_neutral, xgboost_approx_prob_up,
        lightgbm_prob_down, lightgbm_prob_neutral, lightgbm_prob_up,
        catboost_prob_down, catboost_prob_neutral, catboost_prob_up
    ]
    """

    # CANONICAL BASE MODEL ORDER (MUST NEVER CHANGE)
    BASE_MODEL_ORDER = [
        'xgboost',
        'xgboost_et',
        'xgboost_hist',
        'xgboost_dart',
        'xgboost_gblinear',
        'xgboost_approx',
        'lightgbm',
        'catboost'
    ]

    # Expected feature dimension (8 models × 3 classes)
    META_FEATURE_DIM = 24

    def __init__(self, model_path: str = None, use_gpu: bool = True):
        """
        Initialize meta-learner.

        Args:
            model_path: Path to save/load model artifacts
            use_gpu: Whether to use GPU acceleration
        """
        self.meta_model = None
        self.is_trained = False
        self.use_gpu = use_gpu
        self.model_path = model_path or 'backend/data/turbomode_models/meta_learner'

        self.hyperparameters = {
            'device': 'gpu' if use_gpu else 'cpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'n_estimators': 250,
            'num_leaves': 31,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

    def prepare_meta_features(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert base model 3-class predictions to 24-dimensional meta-feature vector.

        Args:
            base_predictions: Dictionary mapping model names to 3-class probability arrays
                             Example: {
                                 'xgboost': np.array([0.2, 0.3, 0.5]),  # [down, neutral, up]
                                 'xgboost_et': np.array([0.1, 0.4, 0.5]),
                                 ...
                             }

        Returns:
            Flat 24-dimensional feature vector in CANONICAL ORDER

        CRITICAL: This method MUST produce identical output during training and inference.
        """
        meta_features = []

        for model_name in self.BASE_MODEL_ORDER:
            if model_name in base_predictions:
                probs = base_predictions[model_name]

                # Ensure we have exactly 3 probabilities
                if len(probs) != 3:
                    raise ValueError(f"{model_name} must return 3-class probabilities, got {len(probs)}")

                # Append in order: [prob_down, prob_neutral, prob_up]
                meta_features.extend([
                    float(probs[0]),  # prob_down
                    float(probs[1]),  # prob_neutral
                    float(probs[2])   # prob_up
                ])
            else:
                # If model missing, use uniform distribution [0.33, 0.33, 0.33]
                meta_features.extend([0.333, 0.333, 0.333])

        # Convert to numpy array
        meta_features = np.array(meta_features, dtype=np.float32)

        # Verify dimension
        if len(meta_features) != self.META_FEATURE_DIM:
            raise ValueError(f"Expected {self.META_FEATURE_DIM} features, got {len(meta_features)}")

        return meta_features

    def train(self, base_predictions_train: List[Dict[str, np.ndarray]],
              y_train: np.ndarray,
              base_predictions_val: List[Dict[str, np.ndarray]] = None,
              y_val: np.ndarray = None) -> Dict[str, float]:
        """
        Train meta-learner on base model 3-class predictions.

        Args:
            base_predictions_train: List of prediction dicts (one per training sample)
                                   Each dict: {'xgboost': [p_down, p_neutral, p_up], ...}
            y_train: True labels for training set (0=down, 1=neutral, 2=up)
            base_predictions_val: Validation predictions (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary with training metrics
        """
        # Convert base predictions to 24-dimensional meta-features
        print(f"Preparing meta-features from {len(base_predictions_train)} training samples...")
        X_meta_list = []
        for pred_dict in base_predictions_train:
            meta_feat = self.prepare_meta_features(pred_dict)
            X_meta_list.append(meta_feat)

        X_meta = np.array(X_meta_list)
        y_true = y_train

        print(f"Meta-feature matrix shape: {X_meta.shape}")
        print(f"Expected: ({len(base_predictions_train)}, {self.META_FEATURE_DIM})")

        # Verify dimensions
        if X_meta.shape[1] != self.META_FEATURE_DIM:
            raise ValueError(f"Meta-feature dimension mismatch: {X_meta.shape[1]} != {self.META_FEATURE_DIM}")

        # Determine number of classes
        n_classes = len(np.unique(y_true))
        if n_classes != 3:
            raise ValueError(f"Expected 3 classes, got {n_classes}")

        # Convert to DataFrame with feature names
        feature_names = []
        for model_name in self.BASE_MODEL_ORDER:
            feature_names.extend([
                f'{model_name}_prob_down',
                f'{model_name}_prob_neutral',
                f'{model_name}_prob_up'
            ])

        X_meta_df = pd.DataFrame(X_meta, columns=feature_names)

        # Prepare validation set if provided
        X_val_meta_df = None
        if base_predictions_val is not None and y_val is not None:
            print(f"Preparing meta-features from {len(base_predictions_val)} validation samples...")
            X_val_meta_list = []
            for pred_dict in base_predictions_val:
                meta_feat = self.prepare_meta_features(pred_dict)
                X_val_meta_list.append(meta_feat)

            X_val_meta = np.array(X_val_meta_list)
            X_val_meta_df = pd.DataFrame(X_val_meta, columns=feature_names)

        # Initialize meta-model
        self.meta_model = lgb.LGBMClassifier(**self.hyperparameters)

        # Train with validation set if provided
        if X_val_meta_df is not None:
            self.meta_model.fit(
                X_meta_df, y_true,
                eval_set=[(X_meta_df, y_true), (X_val_meta_df, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
            )

            # Calculate accuracies
            train_accuracy = self.meta_model.score(X_meta_df, y_true)
            val_accuracy = self.meta_model.score(X_val_meta_df, y_val)

            # Calculate feature importance (which base models are most important)
            feature_importance = self.meta_model.feature_importances_
            model_importance = {}
            for i, model_name in enumerate(self.BASE_MODEL_ORDER):
                # Each model contributes 3 features
                importance = feature_importance[i*3:(i+1)*3].sum()
                model_importance[model_name] = float(importance)

            # Normalize importance to percentages
            total_importance = sum(model_importance.values())
            if total_importance > 0:
                model_importance = {k: (v/total_importance)*100 for k, v in model_importance.items()}

            metrics = {
                'train_accuracy': float(train_accuracy),
                'val_accuracy': float(val_accuracy),
                'n_meta_features': X_meta.shape[1],
                'n_base_models': len(self.BASE_MODEL_ORDER),
                'model_importance': model_importance
            }
        else:
            # Train without validation
            self.meta_model.fit(X_meta_df, y_true)
            train_accuracy = self.meta_model.score(X_meta_df, y_true)

            metrics = {
                'train_accuracy': float(train_accuracy),
                'n_meta_features': X_meta.shape[1],
                'n_base_models': len(self.BASE_MODEL_ORDER)
            }

        self.is_trained = True
        return metrics

    def predict(self, base_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Generate final prediction from base model 3-class predictions.

        Args:
            base_predictions: Dictionary of 3-class predictions
                             Example: {'xgboost': [0.2, 0.3, 0.5], ...}

        Returns:
            Dictionary with:
            - prob_down: Probability of down class
            - prob_neutral: Probability of neutral class
            - prob_up: Probability of up class
            - predicted_class: 0=down, 1=neutral, 2=up
        """
        if not self.is_trained or self.meta_model is None:
            raise ValueError("Meta-learner must be trained before prediction")

        # Check if input has override-aware features (55) or just base probabilities (24)
        # If it has scalar override features, it's the new 55-feature format
        has_override_features = any(isinstance(v, (int, float, np.integer, np.floating))
                                   for k, v in base_predictions.items()
                                   if '_asymmetry' in k or '_max_directional' in k)

        if has_override_features:
            # New 55-feature format: use features directly
            model_names = ['xgboost', 'xgboost_et', 'lightgbm', 'catboost',
                          'xgboost_hist', 'xgboost_dart', 'xgboost_gblinear', 'xgboost_approx']

            # Build feature array in the correct order (must match training)
            features = []

            # 24 base probability features
            for model_name in model_names:
                probs_array = base_predictions[model_name]
                features.extend([probs_array[0], probs_array[1], probs_array[2]])  # down, neutral, up

            # 24 per-model override features
            for model_name in model_names:
                features.append(base_predictions[f'{model_name}_asymmetry'])
                features.append(base_predictions[f'{model_name}_max_directional'])
                features.append(base_predictions[f'{model_name}_neutral_dominance'])

            # 7 aggregate features
            features.append(base_predictions['avg_asymmetry'])
            features.append(base_predictions['max_asymmetry'])
            features.append(base_predictions['avg_max_directional'])
            features.append(base_predictions['avg_neutral_dominance'])
            features.append(base_predictions['models_favor_up'])
            features.append(base_predictions['models_favor_down'])
            features.append(base_predictions['directional_consensus'])

            X_meta = np.array([features])

            # Build feature names for DataFrame
            feature_names = []
            for model_name in model_names:
                feature_names.extend([
                    f'{model_name}_prob_down',
                    f'{model_name}_prob_neutral',
                    f'{model_name}_prob_up'
                ])
            for model_name in model_names:
                feature_names.extend([
                    f'{model_name}_asymmetry',
                    f'{model_name}_max_directional',
                    f'{model_name}_neutral_dominance'
                ])
            feature_names.extend([
                'avg_asymmetry', 'max_asymmetry', 'avg_max_directional', 'avg_neutral_dominance',
                'models_favor_up', 'models_favor_down', 'directional_consensus'
            ])

            X_meta_df = pd.DataFrame(X_meta, columns=feature_names)
        else:
            # Old 24-feature format: use prepare_meta_features
            meta_features = self.prepare_meta_features(base_predictions)
            X_meta = meta_features.reshape(1, -1)

            feature_names = []
            for model_name in self.BASE_MODEL_ORDER:
                feature_names.extend([
                    f'{model_name}_prob_down',
                    f'{model_name}_prob_neutral',
                    f'{model_name}_prob_up'
                ])

            X_meta_df = pd.DataFrame(X_meta, columns=feature_names)

        # Get 3-class probabilities
        probs = self.meta_model.predict_proba(X_meta_df)[0]
        predicted_class = int(np.argmax(probs))  # Get class with highest probability

        return {
            'prob_down': float(probs[0]),
            'prob_neutral': float(probs[1]),
            'prob_up': float(probs[2]),
            'predicted_class': predicted_class
        }

    def predict_batch(self, base_predictions_list: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Generate predictions for batch of samples.

        Args:
            base_predictions_list: List of base prediction dictionaries

        Returns:
            Array of predicted class labels
        """
        if not self.is_trained or self.meta_model is None:
            raise ValueError("Meta-learner must be trained before prediction")

        # Convert all base predictions to meta-features
        X_meta_list = []
        for pred_dict in base_predictions_list:
            meta_feat = self.prepare_meta_features(pred_dict)
            X_meta_list.append(meta_feat)

        X_meta = np.array(X_meta_list)

        # Convert to DataFrame
        feature_names = []
        for model_name in self.BASE_MODEL_ORDER:
            feature_names.extend([
                f'{model_name}_prob_down',
                f'{model_name}_prob_neutral',
                f'{model_name}_prob_up'
            ])

        X_meta_df = pd.DataFrame(X_meta, columns=feature_names)

        # Predict using predict_proba then argmax
        probs = self.meta_model.predict_proba(X_meta_df)
        predictions = np.argmax(probs, axis=1)
        return predictions

    def evaluate(self, base_predictions_list: List[Dict[str, np.ndarray]],
                 y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate meta-learner on test set.

        Args:
            base_predictions_list: List of base prediction dictionaries
            y_true: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained or self.meta_model is None:
            raise ValueError("Meta-learner must be trained before evaluation")

        # Convert to meta-features
        X_meta_list = []
        for pred_dict in base_predictions_list:
            meta_feat = self.prepare_meta_features(pred_dict)
            X_meta_list.append(meta_feat)

        X_meta = np.array(X_meta_list)

        # Convert to DataFrame
        feature_names = []
        for model_name in self.BASE_MODEL_ORDER:
            feature_names.extend([
                f'{model_name}_prob_down',
                f'{model_name}_prob_neutral',
                f'{model_name}_prob_up'
            ])

        X_meta_df = pd.DataFrame(X_meta, columns=feature_names)

        # Calculate accuracy
        accuracy = self.meta_model.score(X_meta_df, y_true)

        return {
            'accuracy': float(accuracy),
            'n_samples': len(y_true)
        }

    def save(self) -> None:
        """
        Save meta-learner to disk.

        Saves:
        - meta_learner.txt: LightGBM meta-model
        - metadata.json: Training metadata and model importance
        """
        if not self.is_trained or self.meta_model is None:
            raise ValueError("Meta-learner must be trained before saving")

        # Create directory if needed
        os.makedirs(self.model_path, exist_ok=True)

        # Save LightGBM meta-model
        model_file = os.path.join(self.model_path, 'meta_learner.txt')
        self.meta_model.booster_.save_model(model_file)

        # Save metadata
        metadata = {
            'is_trained': self.is_trained,
            'use_gpu': self.use_gpu,
            'hyperparameters': self.hyperparameters,
            'base_model_order': self.BASE_MODEL_ORDER,
            'n_base_models': len(self.BASE_MODEL_ORDER),
            'meta_feature_dim': self.META_FEATURE_DIM,
            'model_version': '2.0.0',  # Updated for 3-class architecture
            'architecture': '3-class (down/neutral/up)'
        }

        metadata_file = os.path.join(self.model_path, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[OK] Meta-learner saved to {self.model_path}")
        print(f"   - meta_learner.txt: LightGBM meta-model")
        print(f"   - metadata.json: Training metadata")
        print(f"   - Architecture: 3-class (24 features)")

    def load(self) -> None:
        """
        Load meta-learner from disk.

        Loads:
        - meta_learner.txt: LightGBM meta-model
        - metadata.json: Training metadata
        """
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")

        # Load metadata
        metadata_file = os.path.join(self.model_path, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            self.hyperparameters = metadata.get('hyperparameters', self.hyperparameters)
            self.use_gpu = metadata.get('use_gpu', self.use_gpu)

            # Verify architecture
            if metadata.get('meta_feature_dim') != self.META_FEATURE_DIM:
                print(f"WARNING: Loaded model has {metadata.get('meta_feature_dim')} features, expected {self.META_FEATURE_DIM}")

        # Load LightGBM meta-model
        model_file = os.path.join(self.model_path, 'meta_learner.txt')
        if not os.path.exists(model_file):
            raise ValueError(f"Model file not found: {model_file}")

        # Load booster directly
        booster = lgb.Booster(model_file=model_file)

        # Create model instance and attach booster
        self.meta_model = lgb.LGBMClassifier(**self.hyperparameters)
        self.meta_model._Booster = booster
        self.meta_model._n_features = booster.num_feature()
        self.meta_model._n_classes = 3
        self.meta_model.fitted_ = True

        self.is_trained = True
        print(f"[OK] Meta-learner loaded from {self.model_path}")
        print(f"   - meta_learner.txt: LightGBM meta-model loaded")
        print(f"   - Architecture: 3-class (24 features)")
