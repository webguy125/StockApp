"""
Regime-Weighted Loss

Applies per-sample loss weights based on market regime to emphasize rare and critical periods.

Loss Weights (from JSON spec):
- crash: 2.0x (highest priority - critical events)
- recovery: 1.5x (important transition periods)
- high_volatility: 1.3x (elevated importance)
- normal: 1.0x (baseline)
- low_volatility: 0.8x (slightly less emphasis)

Usage:
- Generate sample_weight array for training
- Compatible with scikit-learn models (sample_weight parameter)
- For custom models, multiply loss by weight
"""

import numpy as np
from typing import List, Dict, Any, Union


class RegimeWeightedLoss:
    """
    Generates sample weights based on market regime
    """

    def __init__(self, regime_weights: Dict[str, float] = None):
        """
        Initialize regime weighted loss

        Args:
            regime_weights: Dictionary of weights per regime
                           Defaults to JSON spec weights if None
        """
        self.version = "1.0.0"

        # Default weights from JSON spec
        self.regime_weights = regime_weights or {
            'crash': 2.0,
            'recovery': 1.5,
            'high_volatility': 1.3,
            'normal': 1.0,
            'low_volatility': 0.8
        }

    def get_sample_weights(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate sample weight array for training

        Args:
            samples: List of samples with 'regime' field

        Returns:
            Numpy array of sample weights (same length as samples)
        """
        weights = []

        for sample in samples:
            regime = sample.get('regime', 'normal')
            weight = self.regime_weights.get(regime, 1.0)
            weights.append(weight)

        return np.array(weights, dtype=np.float32)

    def get_sample_weight(self, sample: Dict[str, Any]) -> float:
        """
        Get weight for a single sample

        Args:
            sample: Sample dictionary with 'regime' field

        Returns:
            Weight value
        """
        regime = sample.get('regime', 'normal')
        return self.regime_weights.get(regime, 1.0)

    def calculate_effective_dataset_size(self, samples: List[Dict[str, Any]]) -> float:
        """
        Calculate effective dataset size after weighting

        Args:
            samples: List of samples

        Returns:
            Effective size (weighted sum)
        """
        weights = self.get_sample_weights(samples)
        return float(np.sum(weights))

    def get_regime_weight_distribution(self, samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate regime weight distribution statistics

        Args:
            samples: List of samples

        Returns:
            Dictionary with per-regime statistics
        """
        weights = self.get_sample_weights(samples)
        total_weight = np.sum(weights)

        # Group by regime
        regime_stats = {regime: {'count': 0, 'total_weight': 0.0} for regime in self.regime_weights.keys()}

        for sample, weight in zip(samples, weights):
            regime = sample.get('regime', 'normal')
            if regime in regime_stats:
                regime_stats[regime]['count'] += 1
                regime_stats[regime]['total_weight'] += weight

        # Calculate percentages
        for regime, stats in regime_stats.items():
            stats['weight_pct'] = (stats['total_weight'] / total_weight * 100) if total_weight > 0 else 0
            stats['count_pct'] = (stats['count'] / len(samples) * 100) if len(samples) > 0 else 0
            stats['avg_weight'] = self.regime_weights[regime]

        return regime_stats

    def print_weight_distribution(self, samples: List[Dict[str, Any]]):
        """
        Print regime weight distribution statistics

        Args:
            samples: List of samples
        """
        distribution = self.get_regime_weight_distribution(samples)
        total_weight = self.calculate_effective_dataset_size(samples)

        print("\nRegime Weight Distribution:")
        print(f"{'Regime':<18} {'Count':>6}  {'Weight':>6}  {'Count%':>7}  {'Weight%':>8}")
        print("-" * 62)

        for regime in ['crash', 'recovery', 'high_volatility', 'normal', 'low_volatility']:
            if regime in distribution:
                stats = distribution[regime]
                print(f"{regime:<18} {stats['count']:>6}  {stats['avg_weight']:>6.1f}x  "
                      f"{stats['count_pct']:>6.1f}%  {stats['weight_pct']:>7.1f}%")

        print("-" * 62)
        print(f"{'Total Samples:':<33} {len(samples)}")
        print(f"{'Effective Dataset Size:':<33} {total_weight:.1f}")
        print(f"{'Weighting Factor:':<33} {total_weight / len(samples):.2f}x")


def get_regime_weights(samples: List[Dict[str, Any]],
                      custom_weights: Dict[str, float] = None) -> np.ndarray:
    """
    Convenience function to get sample weights

    Args:
        samples: List of samples with 'regime' field
        custom_weights: Optional custom weight mapping

    Returns:
        Numpy array of sample weights
    """
    weighted_loss = RegimeWeightedLoss(custom_weights)
    return weighted_loss.get_sample_weights(samples)


class RegimeWeightedMetrics:
    """
    Calculate metrics with regime weighting applied
    """

    def __init__(self, regime_weights: Dict[str, float] = None):
        self.weighted_loss = RegimeWeightedLoss(regime_weights)

    def weighted_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, samples: List[Dict[str, Any]]) -> float:
        """
        Calculate weighted accuracy

        Args:
            y_true: True labels
            y_pred: Predicted labels
            samples: Samples (for regime weights)

        Returns:
            Weighted accuracy
        """
        weights = self.weighted_loss.get_sample_weights(samples)
        correct = (y_true == y_pred).astype(float)
        weighted_correct = correct * weights
        return float(np.sum(weighted_correct) / np.sum(weights))

    def weighted_mae(self, y_true: np.ndarray, y_pred: np.ndarray, samples: List[Dict[str, Any]]) -> float:
        """
        Calculate weighted mean absolute error

        Args:
            y_true: True values
            y_pred: Predicted values
            samples: Samples (for regime weights)

        Returns:
            Weighted MAE
        """
        weights = self.weighted_loss.get_sample_weights(samples)
        errors = np.abs(y_true - y_pred)
        weighted_errors = errors * weights
        return float(np.sum(weighted_errors) / np.sum(weights))


if __name__ == '__main__':
    # Test regime weighted loss
    print("Testing Regime Weighted Loss...")

    # Create test samples
    test_samples = []
    test_samples.extend([{'regime': 'crash'} for _ in range(100)])
    test_samples.extend([{'regime': 'recovery'} for _ in range(100)])
    test_samples.extend([{'regime': 'high_volatility'} for _ in range(200)])
    test_samples.extend([{'regime': 'normal'} for _ in range(400)])
    test_samples.extend([{'regime': 'low_volatility'} for _ in range(200)])

    print(f"\nTest dataset: {len(test_samples)} samples")

    weighted_loss = RegimeWeightedLoss()

    # Generate weights
    weights = weighted_loss.get_sample_weights(test_samples)

    print(f"Sample weights generated: {len(weights)}")
    print(f"  Min weight: {np.min(weights):.1f}x")
    print(f"  Max weight: {np.max(weights):.1f}x")
    print(f"  Mean weight: {np.mean(weights):.2f}x")

    # Print distribution
    weighted_loss.print_weight_distribution(test_samples)

    # Test weighted metrics
    print("\nTesting Weighted Metrics...")
    metrics = RegimeWeightedMetrics()

    # Simulate predictions
    y_true = np.array([0, 1, 2] * 333 + [0])  # 1000 samples
    y_pred = np.array([0, 1, 2] * 333 + [1])  # Last one is wrong

    weighted_acc = metrics.weighted_accuracy(y_true, y_pred, test_samples)
    print(f"  Weighted Accuracy: {weighted_acc*100:.2f}%")
