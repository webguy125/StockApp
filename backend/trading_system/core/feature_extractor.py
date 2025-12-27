"""
Feature Extractor
Converts analyzer outputs into ML feature vectors
"""

import numpy as np
from typing import Dict, List, Any


class FeatureExtractor:
    """
    Extracts and normalizes features from analyzer outputs
    Converts to format suitable for ML models
    """

    def __init__(self):
        self.feature_names = []

    def extract_features(self, analyzer_results: Dict[str, Any]) -> np.ndarray:
        """
        Convert analyzer outputs to feature vector

        Args:
            analyzer_results: Dict from AnalyzerRegistry.analyze_symbol()
                {
                    'analyzer_name': {
                        'signal_strength': float,
                        'direction': str,
                        'confidence': float,
                        'metadata': dict
                    },
                    ...
                }

        Returns:
            numpy array of features
        """
        features = []
        self.feature_names = []

        for analyzer_name, result in sorted(analyzer_results.items()):
            # Base features
            features.append(result['signal_strength'])
            self.feature_names.append(f"{analyzer_name}_strength")

            features.append(result['confidence'])
            self.feature_names.append(f"{analyzer_name}_confidence")

            # Direction encoding (-1 = bearish, 0 = neutral, 1 = bullish)
            direction_map = {'bearish': -1.0, 'neutral': 0.0, 'bullish': 1.0}
            features.append(direction_map.get(result['direction'], 0.0))
            self.feature_names.append(f"{analyzer_name}_direction")

        return np.array(features)

    def extract_features_with_combinations(self, analyzer_results: Dict[str, Any]) -> np.ndarray:
        """
        Extract features including derived combinations

        Adds:
        - Raw analyzer values
        - Cross-indicator products
        - Confidence-weighted signals
        """
        # Get base features
        base_features = self.extract_features(analyzer_results)

        # Add derived features
        derived_features = []

        # Average signal strength
        strengths = [r['signal_strength'] for r in analyzer_results.values()]
        if strengths:
            derived_features.append(np.mean(strengths))
            derived_features.append(np.std(strengths))
            derived_features.append(np.max(strengths))
            derived_features.append(np.min(strengths))

        # Bullish consensus (how many analyzers are bullish?)
        bullish_count = sum(1 for r in analyzer_results.values() if r['direction'] == 'bullish')
        bearish_count = sum(1 for r in analyzer_results.values() if r['direction'] == 'bearish')
        total = len(analyzer_results)

        if total > 0:
            derived_features.append(bullish_count / total)
            derived_features.append(bearish_count / total)

        # Confidence-weighted average signal
        weighted_signals = [
            r['signal_strength'] * r['confidence']
            for r in analyzer_results.values()
        ]
        if weighted_signals:
            derived_features.append(np.mean(weighted_signals))

        # Combine all features
        all_features = np.concatenate([base_features, np.array(derived_features)])

        return all_features

    def create_feature_dict(self, analyzer_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Create a dict of features for easy inspection

        Returns:
            Dict mapping feature names to values
        """
        features = self.extract_features(analyzer_results)

        return dict(zip(self.feature_names, features))

    def get_feature_importance(self, analyzer_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate feature importance based on analyzer weights

        Args:
            analyzer_weights: Dict from AnalyzerRegistry.get_weights()

        Returns:
            Dict mapping feature names to importance scores
        """
        importance = {}

        for analyzer_name, weight in analyzer_weights.items():
            # Each analyzer contributes 3 features
            importance[f"{analyzer_name}_strength"] = weight
            importance[f"{analyzer_name}_confidence"] = weight * 0.8  # Slightly less important
            importance[f"{analyzer_name}_direction"] = weight * 0.9

        return importance

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to 0-1 range

        Note: This is a simple normalization. For production, use sklearn's StandardScaler
        """
        features_min = features.min()
        features_max = features.max()

        if features_max - features_min > 0:
            return (features - features_min) / (features_max - features_min)
        else:
            return features

    def __repr__(self):
        return f"<FeatureExtractor features={len(self.feature_names)}>"
