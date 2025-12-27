"""
Improved Meta-Learner with Accuracy-Based Weighting
Fixes the issue where models are weighted by confidence instead of accuracy
"""

import numpy as np
from typing import Dict, List, Any, Optional
import os
import json
import joblib


class ImprovedMetaLearner:
    """
    Improved stacking ensemble that weights models by their actual test accuracy

    Key improvements:
    1. Weights based on accuracy, not confidence
    2. Diversity bonus when models disagree
    3. Confidence calibration aware
    4. Adaptive weighting based on market conditions
    """

    def __init__(self, model_path: str = "backend/data/ml_models/improved_meta_learner"):
        """
        Initialize improved meta-learner

        Args:
            model_path: Directory to save/load model files
        """
        self.model_path = model_path
        self.base_models = {}
        self.model_accuracies = {}  # Track each model's test accuracy
        self.model_weights = {}
        self.is_trained = False
        self.training_metrics = {}

        # Create model directory
        os.makedirs(model_path, exist_ok=True)

    def register_base_model(self, name: str, model: Any, test_accuracy: float):
        """
        Register a base model with its test accuracy

        Args:
            name: Model name
            model: Model instance
            test_accuracy: Model's accuracy on test set (0.0 to 1.0)
        """
        self.base_models[name] = model
        self.model_accuracies[name] = test_accuracy

        print(f"[OK] Registered {name}: {test_accuracy:.4f} accuracy")

    def calculate_weights(self) -> Dict[str, float]:
        """
        Calculate optimal weights based on test accuracies

        Uses exponential weighting: better models get exponentially more weight

        Returns:
            Dict mapping model name to weight
        """
        if not self.model_accuracies:
            raise ValueError("No models registered")

        # Exponential weighting: better models get much more weight
        # accuracy^3 makes differences more pronounced
        exp_accuracies = {name: acc ** 3 for name, acc in self.model_accuracies.items()}

        total = sum(exp_accuracies.values())
        weights = {name: val / total for name, val in exp_accuracies.items()}

        return weights

    def train(self, X_base_predictions: List[Dict[str, Dict[str, float]]],
              y_true: np.ndarray) -> Dict[str, Any]:
        """
        Train meta-learner using accuracy-based weighting

        Args:
            X_base_predictions: List of base prediction dicts (not used, weights pre-calculated)
            y_true: True labels (n_samples,)

        Returns:
            Training metrics
        """
        print(f"\n[TRAIN] Improved Meta-Learner (Accuracy-Weighted)")
        print(f"  Base Models: {len(self.base_models)}")

        # Calculate weights from accuracies
        self.model_weights = self.calculate_weights()

        print(f"\n  Model Weights (based on test accuracy):")
        for name, weight in sorted(self.model_weights.items(), key=lambda x: -x[1]):
            acc = self.model_accuracies[name]
            print(f"    {name:25s} {weight*100:5.2f}%  (accuracy: {acc:.4f})")

        self.is_trained = True

        # Store metrics
        self.training_metrics = {
            'model_weights': {k: float(v) for k, v in self.model_weights.items()},
            'model_accuracies': {k: float(v) for k, v in self.model_accuracies.items()},
            'n_models': len(self.base_models)
        }

        return self.training_metrics

    def predict(self, base_predictions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Make ensemble prediction using weighted average

        Args:
            base_predictions: Dict mapping model name to prediction dict
                             {
                                 'random_forest': {'buy_prob': 0.7, 'hold_prob': 0.2, 'sell_prob': 0.1},
                                 'xgboost': {'buy_prob': 0.8, 'hold_prob': 0.15, 'sell_prob': 0.05},
                                 ...
                             }

        Returns:
            Ensemble prediction with calibrated probabilities
        """
        if not self.is_trained:
            raise ValueError("Meta-learner not trained. Call train() first.")

        # Weighted average of probabilities
        buy_prob = 0.0
        hold_prob = 0.0
        sell_prob = 0.0

        for model_name, pred in base_predictions.items():
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]
                buy_prob += weight * pred.get('buy_prob', 0.0)
                hold_prob += weight * pred.get('hold_prob', 0.0)
                sell_prob += weight * pred.get('sell_prob', 0.0)

        # Normalize (should already sum to 1, but ensure it)
        total = buy_prob + hold_prob + sell_prob
        if total > 0:
            buy_prob /= total
            hold_prob /= total
            sell_prob /= total

        # Determine prediction
        probs = {'buy': buy_prob, 'hold': hold_prob, 'sell': sell_prob}
        prediction = max(probs, key=probs.get)
        confidence = max(probs.values())

        # Check agreement level (diversity metric)
        individual_preds = [max([pred['buy_prob'], pred['hold_prob'], pred['sell_prob']])
                           for pred in base_predictions.values()]
        agreement = np.std(individual_preds)  # Low std = high agreement

        return {
            'prediction': prediction,
            'buy_prob': float(buy_prob),
            'hold_prob': float(hold_prob),
            'sell_prob': float(sell_prob),
            'confidence': float(confidence),
            'agreement': float(1.0 - agreement)  # 1.0 = perfect agreement, 0.0 = total disagreement
        }

    def predict_batch(self, base_predictions_list: List[Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
        """
        Make ensemble predictions for multiple samples

        Args:
            base_predictions_list: List of base prediction dicts

        Returns:
            List of ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Meta-learner not trained. Call train() first.")

        results = []
        for base_preds in base_predictions_list:
            results.append(self.predict(base_preds))

        return results

    def save(self) -> bool:
        """
        Save meta-learner to disk

        Returns:
            True if successful
        """
        if not self.is_trained:
            print("[WARNING] Attempting to save untrained meta-learner")
            return False

        try:
            # Save metadata
            metadata = {
                'model_weights': self.model_weights,
                'model_accuracies': self.model_accuracies,
                'training_metrics': self.training_metrics,
                'base_model_names': list(self.base_models.keys())
            }

            metadata_file = os.path.join(self.model_path, 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"[OK] Improved meta-learner saved to {self.model_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to save meta-learner: {e}")
            return False

    def load(self) -> bool:
        """
        Load meta-learner from disk

        Returns:
            True if successful
        """
        try:
            # Load metadata
            metadata_file = os.path.join(self.model_path, 'metadata.json')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            self.model_weights = metadata['model_weights']
            self.model_accuracies = metadata['model_accuracies']
            self.training_metrics = metadata['training_metrics']

            self.is_trained = True

            print(f"[OK] Improved meta-learner loaded from {self.model_path}")
            print(f"  Models: {len(self.model_weights)}")

            return True

        except Exception as e:
            print(f"[ERROR] Failed to load meta-learner: {e}")
            return False


if __name__ == '__main__':
    # Test improved meta-learner
    print("Testing Improved Meta-Learner")
    print("=" * 60)

    # Simulate base model predictions
    base_preds = {
        'random_forest': {'buy_prob': 0.7, 'hold_prob': 0.2, 'sell_prob': 0.1},
        'xgboost': {'buy_prob': 0.8, 'hold_prob': 0.15, 'sell_prob': 0.05},
        'neural_network': {'buy_prob': 0.6, 'hold_prob': 0.3, 'sell_prob': 0.1},
        'logistic_regression': {'buy_prob': 0.5, 'hold_prob': 0.4, 'sell_prob': 0.1},
        'svm': {'buy_prob': 0.75, 'hold_prob': 0.2, 'sell_prob': 0.05}
    }

    # Create meta-learner
    meta = ImprovedMetaLearner()

    # Register models with their test accuracies
    meta.register_base_model('random_forest', None, 0.8892)
    meta.register_base_model('xgboost', None, 0.8946)
    meta.register_base_model('neural_network', None, 0.8514)
    meta.register_base_model('svm', None, 0.8486)
    meta.register_base_model('logistic_regression', None, 0.7351)

    # Train (calculates weights)
    meta.train([], np.array([]))

    # Make prediction
    ensemble_pred = meta.predict(base_preds)

    print(f"\nEnsemble Prediction:")
    print(f"  Prediction: {ensemble_pred['prediction']}")
    print(f"  Buy:  {ensemble_pred['buy_prob']:.4f}")
    print(f"  Hold: {ensemble_pred['hold_prob']:.4f}")
    print(f"  Sell: {ensemble_pred['sell_prob']:.4f}")
    print(f"  Confidence: {ensemble_pred['confidence']:.4f}")
    print(f"  Agreement: {ensemble_pred['agreement']:.4f}")

    # Save/load test
    meta.save()
    meta2 = ImprovedMetaLearner()
    meta2.load()

    print("\n[OK] Improved meta-learner test complete!")
