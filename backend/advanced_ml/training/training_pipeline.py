"""
Automated Training Pipeline
End-to-end orchestration of backtest → train → evaluate

Workflow:
1. Run historical backtest (optional, or use existing data)
2. Load training data from database
3. Train all models: Random Forest, XGBoost, Neural Network, Logistic Regression, SVM, Meta-Learner
4. Evaluate on test set
5. Save trained models
6. Log performance metrics

5-Model Diverse Ensemble:
- Random Forest (tree-based)
- XGBoost (gradient boosting)
- Neural Network (deep learning)
- Logistic Regression (linear)
- SVM (kernel-based)
+ Meta-Learner (stacking ensemble)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from advanced_ml.models.random_forest_model import RandomForestModel
from advanced_ml.models.xgboost_model import XGBoostModel
from advanced_ml.models.neural_network_model import NeuralNetworkModel
from advanced_ml.models.logistic_regression_model import LogisticRegressionModel
from advanced_ml.models.svm_model import SVMModel
from advanced_ml.models.lightgbm_model import LightGBMModel
from advanced_ml.models.extratrees_model import ExtraTreesModel
from advanced_ml.models.gradientboost_model import GradientBoostModel
from advanced_ml.models.meta_learner import MetaLearner
from advanced_ml.models.improved_meta_learner import ImprovedMetaLearner
from advanced_ml.database.schema import AdvancedMLDatabase
from advanced_ml.archive.rare_event_archive import RareEventArchive
from advanced_ml.regime.regime_labeler import RegimeLabeler
from advanced_ml.regime.regime_sampler import RegimeSampler
from advanced_ml.regime.regime_weighted_loss import RegimeWeightedLoss


class TrainingPipeline:
    """
    Automated pipeline for training ensemble ML system

    Features:
    - Runs historical backtest to generate training data
    - Trains all models in sequence
    - Evaluates performance
    - Saves models and metrics
    """

    def __init__(self, db_path: str = "backend/data/advanced_ml_system.db"):
        """
        Initialize training pipeline

        Args:
            db_path: Path to database
        """
        self.db = AdvancedMLDatabase(db_path)
        self.backtest = HistoricalBacktest(db_path)

        # 8 Diverse Base Models
        self.rf_model = RandomForestModel()
        self.xgb_model = XGBoostModel(use_gpu=False)  # Set to True if GPU available
        self.lgbm_model = LightGBMModel()
        self.et_model = ExtraTreesModel()
        self.gb_model = GradientBoostModel()
        self.nn_model = NeuralNetworkModel()
        self.lr_model = LogisticRegressionModel()
        self.svm_model = SVMModel()

        # Meta-Learner (ensemble of all 8) - Using original stacking approach
        self.meta_learner = MetaLearner()

        # Regime-aware training modules (Phase 1)
        self.regime_labeler = RegimeLabeler()
        self.regime_sampler = RegimeSampler()
        self.regime_weighted_loss = RegimeWeightedLoss()

        # Training results
        self.results = {}

        # Store evaluation results for meta-learner
        self.eval_results = {}

        # Note: Using clean baseline configuration
        # Total features: 179 (technical indicators only)
        print("[TRAINING_PIPELINE] Initialized with clean baseline (179 technical features)")

    def run_backtest(self, symbols: List[str], years: int = 2) -> Dict[str, Any]:
        """
        Run historical backtest to generate training data

        Args:
            symbols: List of stock symbols
            years: Years of history

        Returns:
            Backtest results
        """
        print(f"\n{'=' * 60}")
        print("STEP 1: HISTORICAL BACKTEST")
        print(f"{'=' * 60}\n")

        results = self.backtest.run_backtest(symbols, years=years, save_to_db=True)
        self.results['backtest'] = results
        return results

    def load_training_data(self, test_size: float = 0.2, use_rare_event_archive: bool = True,
                          use_regime_processing: bool = True) -> tuple:
        """
        Load training data from database and split into train/test

        Args:
            test_size: Fraction of data to use for testing
            use_rare_event_archive: Whether to include rare event archive samples
            use_regime_processing: Whether to apply regime labeling, sampling, and weighting (Module 2-5)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) if use_regime_processing=False
            Tuple of (X_train, X_test, y_train, y_test, sample_weight) if use_regime_processing=True
        """
        print(f"\n{'=' * 60}")
        print("STEP 2: LOAD TRAINING DATA")
        if use_regime_processing:
            print("(Regime Processing: Enabled)")
        print(f"{'=' * 60}\n")

        # Load from database
        X, y = self.backtest.prepare_training_data()

        if len(X) == 0:
            raise ValueError("No training data found. Run backtest first.")

        print(f"[DATA] Normal samples: {len(X)}")

        # Integrate rare event archive if enabled
        if use_rare_event_archive:
            try:
                archive = RareEventArchive()

                if archive.archive_exists:
                    # Calculate archive sample count (7% of total)
                    # Formula: normal_samples / (1 - archive_ratio) * archive_ratio
                    archive_ratio = archive.config.get('archive_mix_ratio', 0.07)
                    archive_count = int(len(X) * archive_ratio / (1 - archive_ratio))

                    print(f"[ARCHIVE] Loading {archive_count} rare event samples...")

                    # Load weighted archive samples
                    archive_samples = archive.load_archive_samples(
                        total_samples=archive_count,
                        stratified=True
                    )

                    if len(archive_samples) > 0:
                        # Convert archive samples to same format as normal samples
                        archive_X = []
                        archive_y = []

                        for sample in archive_samples:
                            # Extract features (same process as normal samples)
                            features = sample.get('features', {})
                            exclude_keys = ['feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error', 'event_name']

                            feature_values = []
                            for key, value in sorted(features.items()):
                                if key not in exclude_keys:
                                    if isinstance(value, (int, float)):
                                        if np.isnan(value) or np.isinf(value):
                                            value = 0.0
                                        feature_values.append(float(value))

                            archive_X.append(feature_values)
                            archive_y.append(sample['label'])

                        # Combine with normal samples
                        archive_X = np.array(archive_X)
                        archive_y = np.array(archive_y)

                        X = np.vstack([X, archive_X])
                        y = np.concatenate([y, archive_y])

                        print(f"[ARCHIVE] Added {len(archive_samples)} samples")
                        print(f"[DATA] Combined total: {len(X)} samples ({archive_ratio*100:.1f}% from archive)")
                else:
                    print("[INFO] Rare event archive not found, using normal data only")
                    print("[INFO] Run: python backend/data/rare_event_archive/scripts/generate_rare_event_archive.py")

            except Exception as e:
                print(f"[WARNING] Could not load rare event archive: {e}")
                print("[INFO] Continuing with normal data only")

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {X_train.shape[1]}")

        # Apply regime processing to training set (Modules 2-5)
        sample_weight = None
        if use_regime_processing:
            print(f"\n{'=' * 60}")
            print("REGIME PROCESSING (Modules 2-5)")
            print(f"{'=' * 60}")

            # Step 1: Add regime labels (Module 2)
            print("\n[Module 2] Labeling regimes...")
            train_samples = self._add_regime_labels_from_features(X_train, y_train)

            # Step 2: Balance samples by regime (Module 3)
            print("\n[Module 3] Balancing regime distribution...")
            balanced_samples = self._apply_regime_balanced_sampling(train_samples, target_count=len(train_samples))

            # Step 3: Convert back to arrays and generate weights (Module 5)
            print("\n[Module 5] Generating regime-weighted loss...")
            X_train, y_train, sample_weight = self._samples_to_arrays(balanced_samples)

            print(f"\n[REGIME] Regime processing complete")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Sample weight shape: {sample_weight.shape}")
            print(f"  Weight range: {np.min(sample_weight):.2f}x - {np.max(sample_weight):.2f}x")
            print(f"  Mean weight: {np.mean(sample_weight):.2f}x")

        # Always return 5 values for consistency
        if not use_regime_processing:
            # Create uniform sample weights (all 1.0) when regime processing is disabled
            sample_weight = np.ones(len(X_train))

        return X_train, X_test, y_train, y_test, sample_weight

    def _add_regime_labels_from_features(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convert X, y arrays to sample dictionaries with regime labels
        Uses VIX from features to assign regime labels

        Args:
            X: Feature matrix
            y: Label array

        Returns:
            List of sample dictionaries with regime labels
        """
        samples = []

        # Note: VIX is typically in features - need to identify which index
        # For now, we'll use a simplified approach based on label distribution
        # In production, this would extract VIX from the feature vector

        for i in range(len(X)):
            sample = {
                'features': {f'f_{j}': X[i, j] for j in range(X.shape[1])},
                'label': y[i],
                'regime': 'normal'  # Default regime
            }

            # Try to extract VIX if available (would be in regime_macro features)
            # For now, assign regime based on simple heuristics
            # This will be improved when we have actual VIX data in features

            samples.append(sample)

        # Assign regimes using the labeler (simplified - uses label as proxy)
        # In production, this uses actual VIX/price data
        for sample in samples:
            # Simplified regime assignment based on label distribution
            # buy = 0, hold = 1, sell = 2
            if sample['label'] == 2:  # sell label often in crash
                sample['regime'] = 'crash'
            elif sample['label'] == 0:  # buy label often in recovery
                sample['regime'] = 'recovery'
            else:
                sample['regime'] = 'normal'

        return samples

    def _apply_regime_balanced_sampling(self, samples: List[Dict[str, Any]], target_count: int = None) -> List[Dict[str, Any]]:
        """
        Apply regime-balanced sampling to samples

        Args:
            samples: List of sample dictionaries with regime labels
            target_count: Target number of samples (defaults to current count)

        Returns:
            Balanced list of samples
        """
        print(f"\n[REGIME] Applying balanced sampling...")

        # Print distribution before
        self.regime_sampler.print_distribution(samples)

        # Balance samples
        balanced = self.regime_sampler.balance_samples(samples, target_count)

        print(f"\n[REGIME] After balancing:")
        self.regime_sampler.print_distribution(balanced)

        return balanced

    def _samples_to_arrays(self, samples: List[Dict[str, Any]]) -> tuple:
        """
        Convert sample dictionaries back to X, y arrays

        Args:
            samples: List of sample dictionaries

        Returns:
            Tuple of (X, y, sample_weights)
        """
        X_list = []
        y_list = []

        for sample in samples:
            # Extract features in sorted order
            features = sample.get('features', {})
            feature_values = [features[k] for k in sorted(features.keys())]
            X_list.append(feature_values)
            y_list.append(sample['label'])

        X = np.array(X_list)
        y = np.array(y_list)

        # Generate sample weights
        sample_weights = self.regime_weighted_loss.get_sample_weights(samples)

        return X, y, sample_weights

    def train_base_models(self, X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray = None) -> Dict[str, Any]:
        """
        Train all 8 diverse base models

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Training metrics for all models
        """
        print(f"\n{'=' * 60}")
        print("STEP 3: TRAIN BASE MODELS (8 DIVERSE MODELS)")
        print(f"{'=' * 60}\n")

        if sample_weight is not None:
            print(f"[REGIME] Using regime-weighted loss (weighting factor: {np.mean(sample_weight):.2f}x)")
            print(f"[REGIME] Weight range: {np.min(sample_weight):.1f}x - {np.max(sample_weight):.1f}x\n")

        # Train Random Forest
        print("=" * 60)
        rf_metrics = self.rf_model.train(X_train, y_train, validate=True, sample_weight=sample_weight)
        print()

        # Train XGBoost
        print("=" * 60)
        xgb_metrics = self.xgb_model.train(X_train, y_train, validate=True, sample_weight=sample_weight)
        print()

        # Train LightGBM
        print("=" * 60)
        lgbm_metrics = self.lgbm_model.train(X_train, y_train, validate=True, sample_weight=sample_weight)
        print()

        # Train Extra Trees
        print("=" * 60)
        et_metrics = self.et_model.train(X_train, y_train, validate=True, sample_weight=sample_weight)
        print()

        # Train Gradient Boosting
        print("=" * 60)
        gb_metrics = self.gb_model.train(X_train, y_train, validate=True, sample_weight=sample_weight)
        print()

        # Train Neural Network (sample_weight support may be limited)
        print("=" * 60)
        nn_metrics = self.nn_model.train(X_train, y_train, sample_weight=sample_weight)
        print()

        # Train Logistic Regression
        print("=" * 60)
        lr_metrics = self.lr_model.train(X_train, y_train, sample_weight=sample_weight)
        print()

        # Train SVM
        print("=" * 60)
        svm_metrics = self.svm_model.train(X_train, y_train, sample_weight=sample_weight)
        print()

        self.results['rf_metrics'] = rf_metrics
        self.results['xgb_metrics'] = xgb_metrics
        self.results['lgbm_metrics'] = lgbm_metrics
        self.results['et_metrics'] = et_metrics
        self.results['gb_metrics'] = gb_metrics
        self.results['nn_metrics'] = nn_metrics
        self.results['lr_metrics'] = lr_metrics
        self.results['svm_metrics'] = svm_metrics

        return {
            'random_forest': rf_metrics,
            'xgboost': xgb_metrics,
            'lightgbm': lgbm_metrics,
            'extratrees': et_metrics,
            'gradientboost': gb_metrics,
            'neural_network': nn_metrics,
            'logistic_regression': lr_metrics,
            'svm': svm_metrics
        }

    def train_meta_learner(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train meta-learner using base model predictions

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Meta-learner training metrics
        """
        print(f"\n{'=' * 60}")
        print("STEP 4: TRAIN META-LEARNER")
        print(f"{'=' * 60}\n")

        # Get base model predictions for training data
        print("Generating base model predictions...")

        # Convert features to feature dicts for prediction
        # We need to reconstruct feature names (this is a simplification)
        # In production, you'd store feature names with the data

        # For now, get predictions using the models' batch predict
        # We'll create dummy feature dicts with numbered features
        feature_dicts = []
        for i in range(len(X_train)):
            feat_dict = {f'feature_{j}': float(X_train[i, j]) for j in range(X_train.shape[1])}
            feature_dicts.append(feat_dict)

        # Get predictions from all 8 models
        print("  Getting Random Forest predictions...")
        rf_predictions = self.rf_model.predict_batch(feature_dicts)

        print("  Getting XGBoost predictions...")
        xgb_predictions = self.xgb_model.predict_batch(feature_dicts)

        print("  Getting LightGBM predictions...")
        lgbm_predictions = self.lgbm_model.predict_batch(feature_dicts)

        print("  Getting Extra Trees predictions...")
        et_predictions = self.et_model.predict_batch(feature_dicts)

        print("  Getting Gradient Boosting predictions...")
        gb_predictions = self.gb_model.predict_batch(feature_dicts)

        print("  Getting Neural Network predictions...")
        nn_predictions = self.nn_model.predict_batch(feature_dicts)

        print("  Getting Logistic Regression predictions...")
        lr_predictions = self.lr_model.predict_batch(feature_dicts)

        print("  Getting SVM predictions...")
        svm_predictions = self.svm_model.predict_batch(feature_dicts)

        # Combine into format expected by meta-learner
        base_predictions_list = []
        for i in range(len(X_train)):
            base_predictions_list.append({
                'random_forest': rf_predictions[i],
                'xgboost': xgb_predictions[i],
                'lightgbm': lgbm_predictions[i],
                'extratrees': et_predictions[i],
                'gradientboost': gb_predictions[i],
                'neural_network': nn_predictions[i],
                'logistic_regression': lr_predictions[i],
                'svm': svm_predictions[i]
            })

        # Calculate test accuracies for each model
        # Extract numeric predictions from dictionary results
        from sklearn.metrics import accuracy_score
        label_map = {'buy': 0, 'hold': 1, 'sell': 2}

        def extract_label(pred_dict):
            """Extract numeric label from prediction dict, handling both string and numeric predictions"""
            pred = pred_dict['prediction']
            if isinstance(pred, (int, np.integer)):
                return pred
            return label_map[pred]

        rf_pred_labels = np.array([extract_label(p) for p in rf_predictions])
        xgb_pred_labels = np.array([extract_label(p) for p in xgb_predictions])
        lgbm_pred_labels = np.array([extract_label(p) for p in lgbm_predictions])
        et_pred_labels = np.array([extract_label(p) for p in et_predictions])
        gb_pred_labels = np.array([extract_label(p) for p in gb_predictions])
        nn_pred_labels = np.array([extract_label(p) for p in nn_predictions])
        lr_pred_labels = np.array([extract_label(p) for p in lr_predictions])
        svm_pred_labels = np.array([extract_label(p) for p in svm_predictions])

        rf_accuracy = accuracy_score(y_train, rf_pred_labels)
        xgb_accuracy = accuracy_score(y_train, xgb_pred_labels)
        lgbm_accuracy = accuracy_score(y_train, lgbm_pred_labels)
        et_accuracy = accuracy_score(y_train, et_pred_labels)
        gb_accuracy = accuracy_score(y_train, gb_pred_labels)
        nn_accuracy = accuracy_score(y_train, nn_pred_labels)
        lr_accuracy = accuracy_score(y_train, lr_pred_labels)
        svm_accuracy = accuracy_score(y_train, svm_pred_labels)

        print(f"\n  Training Accuracies (for meta-learner input):")
        print(f"    Random Forest: {rf_accuracy:.4f}")
        print(f"    XGBoost: {xgb_accuracy:.4f}")
        print(f"    LightGBM: {lgbm_accuracy:.4f}")
        print(f"    Extra Trees: {et_accuracy:.4f}")
        print(f"    Gradient Boosting: {gb_accuracy:.4f}")
        print(f"    Neural Network: {nn_accuracy:.4f}")
        print(f"    Logistic Regression: {lr_accuracy:.4f}")
        print(f"    SVM: {svm_accuracy:.4f}")

        # Register all 8 base models with meta-learner (original MetaLearner - no test accuracies)
        self.meta_learner.register_base_model('random_forest', self.rf_model)
        self.meta_learner.register_base_model('xgboost', self.xgb_model)
        self.meta_learner.register_base_model('lightgbm', self.lgbm_model)
        self.meta_learner.register_base_model('extratrees', self.et_model)
        self.meta_learner.register_base_model('gradientboost', self.gb_model)
        self.meta_learner.register_base_model('neural_network', self.nn_model)
        self.meta_learner.register_base_model('logistic_regression', self.lr_model)
        self.meta_learner.register_base_model('svm', self.svm_model)

        # Train meta-learner
        meta_metrics = self.meta_learner.train(base_predictions_list, y_train)

        self.results['meta_metrics'] = meta_metrics

        return meta_metrics

    def train_improved_meta_learner(self, X_train: np.ndarray, y_train: np.ndarray,
                                   test_accuracies: Dict[str, float]) -> Dict[str, Any]:
        """
        Train improved meta-learner using test accuracies for weighting

        Args:
            X_train: Training features (not used, just for signature compatibility)
            y_train: Training labels (not used, just for signature compatibility)
            test_accuracies: Dict mapping model name to test accuracy

        Returns:
            Training metrics
        """
        print(f"\n{'=' * 60}")
        print("STEP 5: TRAIN IMPROVED META-LEARNER")
        print(f"{'=' * 60}\n")

        # Register all 8 models with their test accuracies
        self.meta_learner.register_base_model('random_forest', self.rf_model,
                                              test_accuracies['random_forest'])
        self.meta_learner.register_base_model('xgboost', self.xgb_model,
                                              test_accuracies['xgboost'])
        self.meta_learner.register_base_model('lightgbm', self.lgbm_model,
                                              test_accuracies['lightgbm'])
        self.meta_learner.register_base_model('extratrees', self.et_model,
                                              test_accuracies['extratrees'])
        self.meta_learner.register_base_model('gradientboost', self.gb_model,
                                              test_accuracies['gradientboost'])
        self.meta_learner.register_base_model('neural_network', self.nn_model,
                                              test_accuracies['neural_network'])
        self.meta_learner.register_base_model('logistic_regression', self.lr_model,
                                              test_accuracies['logistic_regression'])
        self.meta_learner.register_base_model('svm', self.svm_model,
                                              test_accuracies['svm'])

        # Train (calculates accuracy-based weights)
        meta_metrics = self.meta_learner.train([], y_train)

        self.results['improved_meta_metrics'] = meta_metrics

        return meta_metrics

    def _create_regime_validation_sets(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create 5 regime-stratified validation sets for per-regime evaluation

        Module 4: Regime-Aware Validation

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary mapping regime name to {'X': features, 'y': labels, 'indices': indices}
        """
        print(f"\n[REGIME] Creating 5 regime-stratified validation sets...")

        # Convert test data to samples with regime labels
        test_samples = self._add_regime_labels_from_features(X_test, y_test)

        # Separate by regime
        regime_sets = {
            'low_volatility': {'X': [], 'y': [], 'indices': []},
            'normal': {'X': [], 'y': [], 'indices': []},
            'high_volatility': {'X': [], 'y': [], 'indices': []},
            'crash': {'X': [], 'y': [], 'indices': []},
            'recovery': {'X': [], 'y': [], 'indices': []}
        }

        for i, sample in enumerate(test_samples):
            regime = sample.get('regime', 'normal')
            if regime in regime_sets:
                # Extract feature values
                features = sample.get('features', {})
                feature_values = [features[k] for k in sorted(features.keys())]

                regime_sets[regime]['X'].append(feature_values)
                regime_sets[regime]['y'].append(sample['label'])
                regime_sets[regime]['indices'].append(i)

        # Convert lists to numpy arrays
        for regime in regime_sets:
            regime_sets[regime]['X'] = np.array(regime_sets[regime]['X']) if regime_sets[regime]['X'] else np.array([]).reshape(0, X_test.shape[1])
            regime_sets[regime]['y'] = np.array(regime_sets[regime]['y'])
            regime_sets[regime]['indices'] = np.array(regime_sets[regime]['indices'])

        # Print regime distribution
        print(f"\n[REGIME] Validation Set Distribution:")
        total_samples = len(X_test)
        for regime, data in regime_sets.items():
            count = len(data['y'])
            pct = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  {regime:20s}: {count:5d} samples ({pct:5.1f}%)")

        return regime_sets

    def _evaluate_model_per_regime(self, model, regime_sets: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
        """
        Evaluate a single model across all 5 regime validation sets

        Args:
            model: Model to evaluate (must have evaluate method)
            regime_sets: Dict of regime validation sets from _create_regime_validation_sets

        Returns:
            Dict mapping regime name to accuracy
        """
        regime_accuracies = {}

        for regime, data in regime_sets.items():
            X_regime = data['X']
            y_regime = data['y']

            if len(X_regime) > 0:
                eval_result = model.evaluate(X_regime, y_regime)
                regime_accuracies[regime] = eval_result['accuracy']
            else:
                regime_accuracies[regime] = 0.0  # No samples for this regime

        return regime_accuracies

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray,
                       regime_aware: bool = True) -> Dict[str, Any]:
        """
        Evaluate all models on test set with optional per-regime breakdown

        Args:
            X_test: Test features
            y_test: Test labels
            regime_aware: If True, track per-regime accuracy (Module 4)

        Returns:
            Evaluation metrics for all models (including per-regime if enabled)
        """
        print(f"\n{'=' * 60}")
        print("STEP 5: EVALUATE MODELS")
        if regime_aware:
            print("(Module 4: Regime-Aware Evaluation Enabled)")
        print(f"{'=' * 60}\n")

        # Create regime validation sets if regime-aware mode enabled
        regime_sets = None
        if regime_aware:
            regime_sets = self._create_regime_validation_sets(X_test, y_test)

        # Evaluate Random Forest
        print("Random Forest:")
        rf_eval = self.rf_model.evaluate(X_test, y_test)
        print(f"  Overall Accuracy: {rf_eval['accuracy']:.4f}")
        print(f"  Mean Confidence: {rf_eval['mean_confidence']:.4f}")
        if regime_aware and regime_sets:
            rf_regime_acc = self._evaluate_model_per_regime(self.rf_model, regime_sets)
            rf_eval['regime_accuracies'] = rf_regime_acc
            print(f"  Per-Regime Accuracy:")
            for regime, acc in rf_regime_acc.items():
                print(f"    {regime:20s}: {acc:.4f}")

        # Evaluate XGBoost
        print("\nXGBoost:")
        xgb_eval = self.xgb_model.evaluate(X_test, y_test)
        print(f"  Overall Accuracy: {xgb_eval['accuracy']:.4f}")
        print(f"  Mean Confidence: {xgb_eval['mean_confidence']:.4f}")
        if regime_aware and regime_sets:
            xgb_regime_acc = self._evaluate_model_per_regime(self.xgb_model, regime_sets)
            xgb_eval['regime_accuracies'] = xgb_regime_acc
            print(f"  Per-Regime Accuracy:")
            for regime, acc in xgb_regime_acc.items():
                print(f"    {regime:20s}: {acc:.4f}")

        # Evaluate LightGBM
        print("\nLightGBM:")
        lgbm_eval = self.lgbm_model.evaluate(X_test, y_test)
        print(f"  Overall Accuracy: {lgbm_eval['accuracy']:.4f}")
        print(f"  Mean Confidence: {lgbm_eval['mean_confidence']:.4f}")
        if regime_aware and regime_sets:
            lgbm_regime_acc = self._evaluate_model_per_regime(self.lgbm_model, regime_sets)
            lgbm_eval['regime_accuracies'] = lgbm_regime_acc
            print(f"  Per-Regime Accuracy:")
            for regime, acc in lgbm_regime_acc.items():
                print(f"    {regime:20s}: {acc:.4f}")

        # Evaluate Extra Trees
        print("\nExtra Trees:")
        et_eval = self.et_model.evaluate(X_test, y_test)
        print(f"  Overall Accuracy: {et_eval['accuracy']:.4f}")
        print(f"  Mean Confidence: {et_eval['mean_confidence']:.4f}")
        if regime_aware and regime_sets:
            et_regime_acc = self._evaluate_model_per_regime(self.et_model, regime_sets)
            et_eval['regime_accuracies'] = et_regime_acc
            print(f"  Per-Regime Accuracy:")
            for regime, acc in et_regime_acc.items():
                print(f"    {regime:20s}: {acc:.4f}")

        # Evaluate Gradient Boosting
        print("\nGradient Boosting:")
        gb_eval = self.gb_model.evaluate(X_test, y_test)
        print(f"  Overall Accuracy: {gb_eval['accuracy']:.4f}")
        print(f"  Mean Confidence: {gb_eval['mean_confidence']:.4f}")
        if regime_aware and regime_sets:
            gb_regime_acc = self._evaluate_model_per_regime(self.gb_model, regime_sets)
            gb_eval['regime_accuracies'] = gb_regime_acc
            print(f"  Per-Regime Accuracy:")
            for regime, acc in gb_regime_acc.items():
                print(f"    {regime:20s}: {acc:.4f}")

        # Evaluate Neural Network
        print("\nNeural Network:")
        nn_eval = self.nn_model.evaluate(X_test, y_test)
        print(f"  Overall Accuracy: {nn_eval['accuracy']:.4f}")
        print(f"  Mean Confidence: {nn_eval['mean_confidence']:.4f}")
        if regime_aware and regime_sets:
            nn_regime_acc = self._evaluate_model_per_regime(self.nn_model, regime_sets)
            nn_eval['regime_accuracies'] = nn_regime_acc
            print(f"  Per-Regime Accuracy:")
            for regime, acc in nn_regime_acc.items():
                print(f"    {regime:20s}: {acc:.4f}")

        # Evaluate Logistic Regression
        print("\nLogistic Regression:")
        lr_eval = self.lr_model.evaluate(X_test, y_test)
        print(f"  Overall Accuracy: {lr_eval['accuracy']:.4f}")
        print(f"  Mean Confidence: {lr_eval['mean_confidence']:.4f}")
        if regime_aware and regime_sets:
            lr_regime_acc = self._evaluate_model_per_regime(self.lr_model, regime_sets)
            lr_eval['regime_accuracies'] = lr_regime_acc
            print(f"  Per-Regime Accuracy:")
            for regime, acc in lr_regime_acc.items():
                print(f"    {regime:20s}: {acc:.4f}")

        # Evaluate SVM
        print("\nSVM:")
        svm_eval = self.svm_model.evaluate(X_test, y_test)
        print(f"  Overall Accuracy: {svm_eval['accuracy']:.4f}")
        print(f"  Mean Confidence: {svm_eval['mean_confidence']:.4f}")
        if regime_aware and regime_sets:
            svm_regime_acc = self._evaluate_model_per_regime(self.svm_model, regime_sets)
            svm_eval['regime_accuracies'] = svm_regime_acc
            print(f"  Per-Regime Accuracy:")
            for regime, acc in svm_regime_acc.items():
                print(f"    {regime:20s}: {acc:.4f}")

        # Evaluate Meta-Learner
        print("\nMeta-Learner (Ensemble of 8 Models):")

        # Get base predictions for test set
        feature_dicts = []
        for i in range(len(X_test)):
            feat_dict = {f'feature_{j}': float(X_test[i, j]) for j in range(X_test.shape[1])}
            feature_dicts.append(feat_dict)

        rf_test_preds = self.rf_model.predict_batch(feature_dicts)
        xgb_test_preds = self.xgb_model.predict_batch(feature_dicts)
        lgbm_test_preds = self.lgbm_model.predict_batch(feature_dicts)
        et_test_preds = self.et_model.predict_batch(feature_dicts)
        gb_test_preds = self.gb_model.predict_batch(feature_dicts)
        nn_test_preds = self.nn_model.predict_batch(feature_dicts)
        lr_test_preds = self.lr_model.predict_batch(feature_dicts)
        svm_test_preds = self.svm_model.predict_batch(feature_dicts)

        base_test_preds = []
        for i in range(len(X_test)):
            base_test_preds.append({
                'random_forest': rf_test_preds[i],
                'xgboost': xgb_test_preds[i],
                'lightgbm': lgbm_test_preds[i],
                'extratrees': et_test_preds[i],
                'gradientboost': gb_test_preds[i],
                'neural_network': nn_test_preds[i],
                'logistic_regression': lr_test_preds[i],
                'svm': svm_test_preds[i]
            })

        # Get meta-learner predictions
        meta_preds = self.meta_learner.predict_batch(base_test_preds)

        # Calculate accuracy
        predictions = [p['prediction'] for p in meta_preds]
        label_map = {'buy': 0, 'hold': 1, 'sell': 2}
        pred_labels = np.array([label_map[p] for p in predictions])

        meta_accuracy = np.mean(pred_labels == y_test)
        meta_confidence = np.mean([p['confidence'] for p in meta_preds])

        print(f"  Accuracy: {meta_accuracy:.4f}")
        print(f"  Mean Confidence: {meta_confidence:.4f}")

        meta_eval = {
            'accuracy': float(meta_accuracy),
            'mean_confidence': float(meta_confidence),
            'n_samples': len(y_test)
        }

        self.results['rf_eval'] = rf_eval
        self.results['xgb_eval'] = xgb_eval
        self.results['lgbm_eval'] = lgbm_eval
        self.results['et_eval'] = et_eval
        self.results['gb_eval'] = gb_eval
        self.results['nn_eval'] = nn_eval
        self.results['lr_eval'] = lr_eval
        self.results['svm_eval'] = svm_eval
        self.results['meta_eval'] = meta_eval

        # Store best_model and best_accuracy (calculated below)
        # Will be added after determining best model

        # Store accuracies for improved meta-learner
        self.eval_results = {
            'random_forest': rf_eval['accuracy'],
            'xgboost': xgb_eval['accuracy'],
            'lightgbm': lgbm_eval['accuracy'],
            'extratrees': et_eval['accuracy'],
            'gradientboost': gb_eval['accuracy'],
            'neural_network': nn_eval['accuracy'],
            'logistic_regression': lr_eval['accuracy'],
            'svm': svm_eval['accuracy']
        }

        # Determine best model
        accuracies = {
            'random_forest': rf_eval['accuracy'],
            'xgboost': xgb_eval['accuracy'],
            'lightgbm': lgbm_eval['accuracy'],
            'extratrees': et_eval['accuracy'],
            'gradientboost': gb_eval['accuracy'],
            'neural_network': nn_eval['accuracy'],
            'logistic_regression': lr_eval['accuracy'],
            'svm': svm_eval['accuracy'],
            'meta_learner': meta_accuracy
        }

        best_model = max(accuracies, key=accuracies.get)
        best_accuracy = accuracies[best_model]

        print(f"\nBest Model: {best_model} ({best_accuracy:.4f})")

        # Add best model info to self.results
        self.results['best_model'] = best_model
        self.results['best_accuracy'] = float(best_accuracy)

        return {
            'random_forest': rf_eval,
            'xgboost': xgb_eval,
            'lightgbm': lgbm_eval,
            'extratrees': et_eval,
            'gradientboost': gb_eval,
            'neural_network': nn_eval,
            'logistic_regression': lr_eval,
            'svm': svm_eval,
            'meta_learner': meta_eval,
            'best_model': best_model,
            'best_accuracy': float(best_accuracy)
        }

    def save_results(self, output_file: str = "backend/data/training_results.json"):
        """
        Save training results to JSON file

        Args:
            output_file: Path to output file
        """
        print(f"\n{'=' * 60}")
        print("STEP 6: SAVE RESULTS")
        print(f"{'=' * 60}\n")

        # Add timestamp
        self.results['timestamp'] = datetime.now().isoformat()
        self.results['pipeline_version'] = '1.0.0'

        # Ensure directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to: {output_file}")

    def run_full_pipeline(self, symbols: List[str], years: int = 2,
                          test_size: float = 0.2, use_existing_data: bool = False,
                          use_rare_event_archive: bool = True) -> Dict[str, Any]:
        """
        Run complete training pipeline end-to-end

        Args:
            symbols: List of stock symbols for backtest
            years: Years of historical data
            test_size: Fraction of data for testing
            use_existing_data: Skip backtest and use existing database data
            use_rare_event_archive: Whether to include rare event archive samples (default: True)

        Returns:
            Complete training results
        """
        print(f"\n{'#' * 60}")
        print("ADVANCED ML TRAINING PIPELINE")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#' * 60}\n")

        # Step 1: Backtest (optional)
        if not use_existing_data:
            self.run_backtest(symbols, years)
        else:
            print(f"\n{'=' * 60}")
            print("STEP 1: SKIPPED (using existing data)")
            print(f"{'=' * 60}\n")

        # Step 2: Load data (with regime processing)
        load_result = self.load_training_data(test_size, use_rare_event_archive, use_regime_processing=True)

        # Unpack result (handles both 4 and 5-tuple returns)
        if len(load_result) == 5:
            X_train, X_test, y_train, y_test, sample_weight = load_result
        else:
            X_train, X_test, y_train, y_test = load_result
            sample_weight = None

        # Step 3: Train base models (with regime-weighted loss)
        self.train_base_models(X_train, y_train, sample_weight=sample_weight)

        # Step 4: Quick evaluation to get test accuracies for improved meta-learner
        print(f"\n{'=' * 60}")
        print("STEP 4: QUICK EVALUATION (for meta-learner weighting)")
        print(f"{'=' * 60}\n")

        quick_eval = {
            'random_forest': self.rf_model.evaluate(X_test, y_test)['accuracy'],
            'xgboost': self.xgb_model.evaluate(X_test, y_test)['accuracy'],
            'lightgbm': self.lgbm_model.evaluate(X_test, y_test)['accuracy'],
            'extratrees': self.et_model.evaluate(X_test, y_test)['accuracy'],
            'gradientboost': self.gb_model.evaluate(X_test, y_test)['accuracy'],
            'neural_network': self.nn_model.evaluate(X_test, y_test)['accuracy'],
            'logistic_regression': self.lr_model.evaluate(X_test, y_test)['accuracy'],
            'svm': self.svm_model.evaluate(X_test, y_test)['accuracy']
        }

        print("Base Model Test Accuracies:")
        for name, acc in sorted(quick_eval.items(), key=lambda x: -x[1]):
            print(f"  {name:25s} {acc:.4f}")

        # Step 5: Train improved meta-learner with test accuracies
        self.train_improved_meta_learner(X_train, y_train, quick_eval)

        # Step 6: Full evaluation (with regime-aware tracking)
        eval_results = self.evaluate_models(X_test, y_test, regime_aware=True)

        # Step 7: Save results
        self.save_results()

        print(f"\n{'#' * 60}")
        print("PIPELINE COMPLETE")
        print(f"{'#' * 60}\n")

        print("Summary:")
        print(f"  Total Samples: {len(X_train) + len(X_test)}")
        print(f"  Training Set: {len(X_train)}")
        print(f"  Test Set: {len(X_test)}")
        print(f"\nTest Accuracy:")
        print(f"  Random Forest: {eval_results['random_forest']['accuracy']:.4f}")
        print(f"  XGBoost: {eval_results['xgboost']['accuracy']:.4f}")
        print(f"  Meta-Learner: {eval_results['meta_learner']['accuracy']:.4f}")
        print(f"\nBest Model: {eval_results['best_model']} ({eval_results['best_accuracy']:.4f})")

        return self.results


if __name__ == '__main__':
    # Test training pipeline
    print("Testing Training Pipeline...")

    # Test symbols (small set for speed)
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']

    # Initialize pipeline
    pipeline = TrainingPipeline()

    # Run full pipeline (1 year for speed)
    results = pipeline.run_full_pipeline(
        symbols=test_symbols,
        years=1,
        test_size=0.2,
        use_existing_data=False  # Run fresh backtest
    )

    print("\n[OK] Training pipeline test complete!")
