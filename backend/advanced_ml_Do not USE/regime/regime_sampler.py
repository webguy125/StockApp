"""
Regime-Balanced Sampling

Controls the percentage of each regime in the training dataset to ensure balanced representation.

Target Ratios (from JSON spec):
- low_volatility: 20%
- normal: 40%
- high_volatility: 20%
- crash: 10%
- recovery: 10%

Source Rules:
- low_volatility, normal, high_volatility: from rolling_5yr_window
- crash, recovery: from rare_event_archive

Balancing Strategy:
- Oversample minority regimes if necessary
- Undersample majority regimes
- Monitor for data leakage
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple
from collections import Counter


class RegimeSampler:
    """
    Balances samples across market regimes for training
    """

    def __init__(self, target_ratios: Dict[str, float] = None):
        """
        Initialize regime sampler

        Args:
            target_ratios: Dictionary of target percentages for each regime
                          Defaults to JSON spec ratios if None
        """
        self.version = "1.0.0"

        # Default ratios from JSON spec
        self.target_ratios = target_ratios or {
            'low_volatility': 0.20,
            'normal': 0.40,
            'high_volatility': 0.20,
            'crash': 0.10,
            'recovery': 0.10
        }

        # Validate ratios sum to 1.0
        total = sum(self.target_ratios.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Target ratios must sum to 1.0, got {total}")

    def balance_samples(self, samples: List[Dict[str, Any]], target_count: int = None) -> List[Dict[str, Any]]:
        """
        Balance samples to match target regime ratios

        Args:
            samples: List of samples with 'regime' field
            target_count: Target total number of samples (defaults to current count)

        Returns:
            Balanced list of samples
        """
        if not samples:
            return []

        # Use current count if no target specified
        if target_count is None:
            target_count = len(samples)

        # Separate samples by regime
        regime_samples = self._separate_by_regime(samples)

        # Calculate target counts per regime
        target_counts = {
            regime: int(target_count * ratio)
            for regime, ratio in self.target_ratios.items()
        }

        # Balance each regime
        balanced_samples = []
        for regime, ratio in self.target_ratios.items():
            regime_list = regime_samples.get(regime, [])
            target = target_counts[regime]

            if len(regime_list) == 0:
                print(f"[WARNING] No samples found for regime '{regime}' (target: {target})")
                continue

            # Oversample or undersample to reach target
            balanced_regime = self._sample_regime(regime_list, target)
            balanced_samples.extend(balanced_regime)

        # Shuffle to mix regimes
        random.shuffle(balanced_samples)

        return balanced_samples

    def _separate_by_regime(self, samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Separate samples into regime groups

        Args:
            samples: List of samples with 'regime' field

        Returns:
            Dictionary mapping regime -> list of samples
        """
        regime_samples = {regime: [] for regime in self.target_ratios.keys()}

        for sample in samples:
            regime = sample.get('regime', 'normal')
            if regime in regime_samples:
                regime_samples[regime].append(sample)
            else:
                # Unknown regime, assign to normal
                regime_samples['normal'].append(sample)

        return regime_samples

    def _sample_regime(self, samples: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """
        Sample (oversample or undersample) a regime to reach target count

        Args:
            samples: List of samples for this regime
            target_count: Target number of samples

        Returns:
            Sampled list (may contain duplicates if oversampled)
        """
        current_count = len(samples)

        if current_count == target_count:
            return samples.copy()

        elif current_count > target_count:
            # Undersample: randomly select subset
            return random.sample(samples, target_count)

        else:
            # Oversample: randomly sample with replacement
            # Use numpy for efficient random sampling with replacement
            indices = np.random.choice(current_count, size=target_count, replace=True)
            return [samples[i] for i in indices]

    def get_regime_distribution(self, samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate regime distribution statistics

        Args:
            samples: List of samples with 'regime' field

        Returns:
            Dictionary with regime counts, percentages, and target ratios
        """
        total = len(samples)
        if total == 0:
            return {}

        # Count samples per regime
        regime_counts = Counter(sample.get('regime', 'normal') for sample in samples)

        # Calculate statistics
        distribution = {}
        for regime in self.target_ratios.keys():
            count = regime_counts.get(regime, 0)
            actual_pct = (count / total) * 100 if total > 0 else 0
            target_pct = self.target_ratios[regime] * 100

            distribution[regime] = {
                'count': count,
                'actual_pct': round(actual_pct, 2),
                'target_pct': round(target_pct, 2),
                'delta_pct': round(actual_pct - target_pct, 2)
            }

        return distribution

    def validate_balance(self, samples: List[Dict[str, Any]], tolerance: float = 2.0) -> Tuple[bool, str]:
        """
        Validate that samples are balanced within tolerance

        Args:
            samples: List of samples to validate
            tolerance: Maximum allowed percentage point deviation from target

        Returns:
            Tuple of (is_valid, message)
        """
        distribution = self.get_regime_distribution(samples)

        violations = []
        for regime, stats in distribution.items():
            delta = abs(stats['delta_pct'])
            if delta > tolerance:
                violations.append(
                    f"{regime}: {stats['actual_pct']:.1f}% (target: {stats['target_pct']:.1f}%, off by {delta:.1f}%)"
                )

        if violations:
            message = "Balance validation failed:\n  " + "\n  ".join(violations)
            return False, message
        else:
            message = "Balance validation passed (all regimes within tolerance)"
            return True, message

    def print_distribution(self, samples: List[Dict[str, Any]]):
        """
        Print regime distribution statistics

        Args:
            samples: List of samples
        """
        distribution = self.get_regime_distribution(samples)

        print("\nRegime Distribution:")
        print(f"{'Regime':<18} {'Count':>6}  {'Actual%':>7}  {'Target%':>7}  {'Delta':>7}")
        print("-" * 60)

        for regime in ['low_volatility', 'normal', 'high_volatility', 'crash', 'recovery']:
            if regime in distribution:
                stats = distribution[regime]
                print(f"{regime:<18} {stats['count']:>6}  {stats['actual_pct']:>6.1f}%  "
                      f"{stats['target_pct']:>6.1f}%  {stats['delta_pct']:>+6.1f}%")


def balance_training_data(samples: List[Dict[str, Any]],
                          target_ratios: Dict[str, float] = None,
                          target_count: int = None) -> List[Dict[str, Any]]:
    """
    Convenience function to balance samples by regime

    Args:
        samples: List of samples with 'regime' field
        target_ratios: Optional custom target ratios
        target_count: Optional target total count

    Returns:
        Balanced list of samples
    """
    sampler = RegimeSampler(target_ratios)
    return sampler.balance_samples(samples, target_count)


if __name__ == '__main__':
    # Test regime sampler
    print("Testing Regime Sampler...")

    # Create test samples with imbalanced distribution
    test_samples = []

    # Create imbalanced dataset (heavy on normal, light on crash/recovery)
    test_samples.extend([{'regime': 'normal', 'value': i} for i in range(500)])
    test_samples.extend([{'regime': 'low_volatility', 'value': i} for i in range(100)])
    test_samples.extend([{'regime': 'high_volatility', 'value': i} for i in range(150)])
    test_samples.extend([{'regime': 'crash', 'value': i} for i in range(20)])
    test_samples.extend([{'regime': 'recovery', 'value': i} for i in range(30)])

    print(f"\nOriginal dataset: {len(test_samples)} samples")

    sampler = RegimeSampler()

    print("\nBefore balancing:")
    sampler.print_distribution(test_samples)

    # Balance to 1000 samples
    balanced = sampler.balance_samples(test_samples, target_count=1000)

    print(f"\nBalanced dataset: {len(balanced)} samples")
    print("\nAfter balancing:")
    sampler.print_distribution(balanced)

    # Validate
    is_valid, message = sampler.validate_balance(balanced, tolerance=2.0)
    print(f"\n{message}")
