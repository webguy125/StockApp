"""
Drift Detection Module

Monitors three types of distribution drift:
1. Feature Drift - Technical indicators changing distribution
2. Regime Drift - Market regime composition shifting
3. Prediction Drift - Model output distribution changing

Uses Kolmogorov-Smirnov test for statistical drift detection.
"""

import numpy as np
import sqlite3
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from scipy import stats
from collections import Counter


class DriftDetector:
    """
    Monitors data distribution changes over time

    Strategy:
    - Store baseline distributions (training data)
    - Compare new data to baseline using KS test
    - Alert if drift exceeds threshold (15% by default)
    - Log all drift events to database for analysis
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: float = 0.15,
        db_path: str = "backend/data/advanced_ml_system.db"
    ):
        """
        Initialize drift detector

        Args:
            window_size: Number of samples for baseline comparison
            alert_threshold: KS statistic threshold (0.15 = 15% drift)
            db_path: Path to database for logging
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.db_path = db_path

        # Baseline distributions
        self.baseline_features = None  # Shape: (n_samples, n_features)
        self.baseline_regime_dist = None  # Dict: {'crash': 0.1, 'normal': 0.6, ...}
        self.baseline_predictions = None  # Shape: (n_samples,) - class predictions

        # Feature names for detailed reporting
        self.feature_names = []

    def set_baseline(
        self,
        features: np.ndarray,
        regimes: List[str],
        predictions: np.ndarray,
        feature_names: List[str] = None
    ):
        """
        Store baseline distributions from training data

        Args:
            features: Training features (n_samples, n_features)
            regimes: Regime labels for each sample
            predictions: Model predictions (n_samples,)
            feature_names: Optional feature names for reporting
        """
        # Store last window_size samples as baseline
        n_samples = min(len(features), self.window_size)

        self.baseline_features = features[-n_samples:]
        self.baseline_predictions = predictions[-n_samples:]

        # Calculate regime distribution
        regime_counts = Counter(regimes[-n_samples:])
        total = sum(regime_counts.values())
        self.baseline_regime_dist = {
            regime: count / total
            for regime, count in regime_counts.items()
        }

        # Store feature names
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        print(f"[DRIFT] Baseline set: {n_samples} samples, {features.shape[1]} features")
        print(f"[DRIFT] Regime distribution: {self.baseline_regime_dist}")

    def detect_feature_drift(
        self,
        new_features: np.ndarray,
        log_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Detect drift in feature distributions using KS test

        Args:
            new_features: New feature data (n_samples, n_features)
            log_to_db: Whether to log drift events to database

        Returns:
            Dict with drift detection results:
            {
                'drift_detected': bool,
                'ks_statistic': float,  # Max KS across features
                'drifted_features': List[int],  # Indices with drift
                'drifted_feature_names': List[str],  # Names with drift
                'feature_ks_scores': Dict[int, float],  # Per-feature KS
                'max_drift_feature': str,  # Feature with highest drift
                'n_drifted_features': int
            }
        """
        if self.baseline_features is None:
            return {
                'error': 'Baseline not set. Call set_baseline() first.',
                'drift_detected': False
            }

        # Ensure same number of features
        if new_features.shape[1] != self.baseline_features.shape[1]:
            return {
                'error': f'Feature count mismatch: {new_features.shape[1]} vs {self.baseline_features.shape[1]}',
                'drift_detected': False
            }

        n_features = new_features.shape[1]
        drifted_features = []
        feature_ks_scores = {}
        max_ks = 0.0
        max_ks_feature_idx = 0

        # Test each feature independently
        for i in range(n_features):
            baseline_feature = self.baseline_features[:, i]
            new_feature = new_features[:, i]

            # Skip if constant (no variance)
            if np.std(baseline_feature) == 0 or np.std(new_feature) == 0:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(baseline_feature, new_feature)
            feature_ks_scores[i] = ks_stat

            # Track maximum drift
            if ks_stat > max_ks:
                max_ks = ks_stat
                max_ks_feature_idx = i

            # Check if drifted
            if ks_stat > self.alert_threshold:
                drifted_features.append(i)

        # Determine overall drift
        drift_detected = len(drifted_features) > 0

        # Get feature names
        drifted_feature_names = [
            self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            for i in drifted_features
        ]

        max_drift_feature = (
            self.feature_names[max_ks_feature_idx]
            if max_ks_feature_idx < len(self.feature_names)
            else f"feature_{max_ks_feature_idx}"
        )

        result = {
            'drift_detected': drift_detected,
            'ks_statistic': float(max_ks),
            'drifted_features': drifted_features,
            'drifted_feature_names': drifted_feature_names,
            'feature_ks_scores': {int(k): float(v) for k, v in feature_ks_scores.items()},
            'max_drift_feature': max_drift_feature,
            'n_drifted_features': len(drifted_features),
            'total_features': n_features
        }

        # Log to database
        if log_to_db:
            self._log_drift_event('feature', result)

        return result

    def detect_regime_drift(
        self,
        new_regimes: List[str],
        log_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Detect drift in regime distribution

        Args:
            new_regimes: List of regime labels for new samples
            log_to_db: Whether to log drift events to database

        Returns:
            Dict with regime drift results:
            {
                'drift_detected': bool,
                'regime_shift': str,  # Description of shift
                'distribution_change': float,  # Max change in any regime %
                'new_distribution': Dict[str, float],
                'baseline_distribution': Dict[str, float],
                'regime_changes': Dict[str, float]  # Per-regime change
            }
        """
        if self.baseline_regime_dist is None:
            return {
                'error': 'Baseline not set. Call set_baseline() first.',
                'drift_detected': False
            }

        # Calculate new regime distribution
        regime_counts = Counter(new_regimes)
        total = sum(regime_counts.values())
        new_regime_dist = {
            regime: count / total
            for regime, count in regime_counts.items()
        }

        # Calculate distribution change for each regime
        all_regimes = set(list(self.baseline_regime_dist.keys()) + list(new_regime_dist.keys()))
        regime_changes = {}
        max_change = 0.0
        max_change_regime = ""

        for regime in all_regimes:
            baseline_pct = self.baseline_regime_dist.get(regime, 0.0)
            new_pct = new_regime_dist.get(regime, 0.0)
            change = abs(new_pct - baseline_pct)
            regime_changes[regime] = change

            if change > max_change:
                max_change = change
                max_change_regime = regime

        # Detect drift (>15% change in any regime)
        drift_detected = max_change > self.alert_threshold

        # Generate shift description
        if drift_detected:
            baseline_pct = self.baseline_regime_dist.get(max_change_regime, 0.0)
            new_pct = new_regime_dist.get(max_change_regime, 0.0)
            if new_pct > baseline_pct:
                regime_shift = f"Increased {max_change_regime} ({baseline_pct:.1%} to {new_pct:.1%})"
            else:
                regime_shift = f"Decreased {max_change_regime} ({baseline_pct:.1%} to {new_pct:.1%})"
        else:
            regime_shift = "Stable"

        result = {
            'drift_detected': drift_detected,
            'regime_shift': regime_shift,
            'distribution_change': float(max_change),
            'new_distribution': new_regime_dist,
            'baseline_distribution': self.baseline_regime_dist,
            'regime_changes': regime_changes
        }

        # Log to database
        if log_to_db:
            self._log_drift_event('regime', result)

        return result

    def detect_prediction_drift(
        self,
        new_predictions: np.ndarray,
        log_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Detect drift in model prediction distribution

        Args:
            new_predictions: New prediction labels (n_samples,)
            log_to_db: Whether to log drift events to database

        Returns:
            Dict with prediction drift results:
            {
                'drift_detected': bool,
                'ks_statistic': float,
                'prediction_shift': str,  # Description
                'new_distribution': Dict[int, float],  # % per class
                'baseline_distribution': Dict[int, float]
            }
        """
        if self.baseline_predictions is None:
            return {
                'error': 'Baseline not set. Call set_baseline() first.',
                'drift_detected': False
            }

        # KS test on prediction distributions
        ks_stat, p_value = stats.ks_2samp(
            self.baseline_predictions,
            new_predictions
        )

        # Calculate class distributions
        baseline_counts = Counter(self.baseline_predictions.tolist() if hasattr(self.baseline_predictions, 'tolist') else self.baseline_predictions)
        baseline_total = sum(baseline_counts.values())
        baseline_dist = {
            int(label): float(count / baseline_total)
            for label, count in baseline_counts.items()
        }

        new_counts = Counter(new_predictions.tolist() if hasattr(new_predictions, 'tolist') else new_predictions)
        new_total = sum(new_counts.values())
        new_dist = {
            int(label): float(count / new_total)
            for label, count in new_counts.items()
        }

        # Determine if drifted
        drift_detected = ks_stat > self.alert_threshold

        # Generate shift description
        label_names = {0: 'buy', 1: 'hold', 2: 'sell'}
        if drift_detected:
            # Find largest change
            all_labels = set(list(baseline_dist.keys()) + list(new_dist.keys()))
            max_change = 0.0
            max_label = 0
            for label in all_labels:
                change = abs(new_dist.get(label, 0.0) - baseline_dist.get(label, 0.0))
                if change > max_change:
                    max_change = change
                    max_label = label

            prediction_shift = f"Shift toward {label_names.get(max_label, max_label)}"
        else:
            prediction_shift = "Stable"

        result = {
            'drift_detected': drift_detected,
            'ks_statistic': float(ks_stat),
            'prediction_shift': prediction_shift,
            'new_distribution': new_dist,
            'baseline_distribution': baseline_dist
        }

        # Log to database
        if log_to_db:
            self._log_drift_event('prediction', result)

        return result

    def _log_drift_event(self, drift_type: str, result: Dict[str, Any]):
        """
        Log drift event to database

        Args:
            drift_type: 'feature', 'regime', or 'prediction'
            result: Drift detection result dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_monitoring (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    drift_type TEXT NOT NULL,
                    ks_statistic REAL,
                    drift_detected INTEGER,
                    details_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Extract KS statistic (if present)
            ks_stat = result.get('ks_statistic', result.get('distribution_change', 0.0))

            # Insert drift event
            cursor.execute('''
                INSERT INTO drift_monitoring
                (timestamp, drift_type, ks_statistic, drift_detected, details_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                drift_type,
                float(ks_stat),
                1 if result.get('drift_detected', False) else 0,
                json.dumps(result)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"[ERROR] Failed to log drift event: {e}")

    def get_drift_history(
        self,
        drift_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve drift monitoring history from database

        Args:
            drift_type: Filter by type ('feature', 'regime', 'prediction'), or None for all
            limit: Maximum records to return

        Returns:
            List of drift event dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if drift_type:
                cursor.execute('''
                    SELECT timestamp, drift_type, ks_statistic, drift_detected, details_json
                    FROM drift_monitoring
                    WHERE drift_type = ?
                    ORDER BY id DESC
                    LIMIT ?
                ''', (drift_type, limit))
            else:
                cursor.execute('''
                    SELECT timestamp, drift_type, ks_statistic, drift_detected, details_json
                    FROM drift_monitoring
                    ORDER BY id DESC
                    LIMIT ?
                ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

            # Parse results
            history = []
            for row in rows:
                event = {
                    'timestamp': row[0],
                    'drift_type': row[1],
                    'ks_statistic': row[2],
                    'drift_detected': bool(row[3]),
                    'details': json.loads(row[4]) if row[4] else {}
                }
                history.append(event)

            return history

        except Exception as e:
            print(f"[ERROR] Failed to retrieve drift history: {e}")
            return []

    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of drift monitoring

        Returns:
            Dict with drift summary:
            {
                'total_checks': int,
                'total_drifts': int,
                'drift_rate': float,
                'by_type': Dict[str, Dict]  # Stats per drift type
            }
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Overall stats
            cursor.execute('SELECT COUNT(*) FROM drift_monitoring')
            total_checks = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM drift_monitoring WHERE drift_detected = 1')
            total_drifts = cursor.fetchone()[0]

            # Stats by type
            cursor.execute('''
                SELECT drift_type,
                       COUNT(*) as checks,
                       SUM(drift_detected) as drifts,
                       AVG(ks_statistic) as avg_ks
                FROM drift_monitoring
                GROUP BY drift_type
            ''')

            by_type = {}
            for row in cursor.fetchall():
                by_type[row[0]] = {
                    'checks': row[1],
                    'drifts': row[2],
                    'drift_rate': row[2] / row[1] if row[1] > 0 else 0.0,
                    'avg_ks': row[3]
                }

            conn.close()

            return {
                'total_checks': total_checks,
                'total_drifts': total_drifts,
                'drift_rate': total_drifts / total_checks if total_checks > 0 else 0.0,
                'by_type': by_type
            }

        except Exception as e:
            print(f"[ERROR] Failed to get drift summary: {e}")
            return {
                'total_checks': 0,
                'total_drifts': 0,
                'drift_rate': 0.0,
                'by_type': {}
            }


if __name__ == '__main__':
    # Test drift detector
    print("Testing Drift Detector...\n")

    # Create detector
    detector = DriftDetector(window_size=100, alert_threshold=0.15)

    # Generate synthetic baseline data
    np.random.seed(42)
    baseline_features = np.random.randn(100, 10)  # 100 samples, 10 features
    baseline_regimes = ['normal'] * 60 + ['high_volatility'] * 30 + ['crash'] * 10
    baseline_predictions = np.random.choice([0, 1, 2], size=100)

    # Set baseline
    detector.set_baseline(
        baseline_features,
        baseline_regimes,
        baseline_predictions,
        feature_names=[f'rsi_{i}' for i in range(10)]
    )

    print("\n[TEST 1] No Drift - Same Distribution")
    print("=" * 50)
    new_features = np.random.randn(50, 10)
    result = detector.detect_feature_drift(new_features, log_to_db=False)
    print(f"Drift Detected: {result['drift_detected']}")
    print(f"Max KS Statistic: {result['ks_statistic']:.3f}")
    print(f"Drifted Features: {result['n_drifted_features']}/{result['total_features']}")

    print("\n[TEST 2] Feature Drift - Mean Shift")
    print("=" * 50)
    drifted_features = np.random.randn(50, 10) + 2.0  # Shift mean by 2
    result = detector.detect_feature_drift(drifted_features, log_to_db=False)
    print(f"Drift Detected: {result['drift_detected']}")
    print(f"Max KS Statistic: {result['ks_statistic']:.3f}")
    print(f"Drifted Features: {result['n_drifted_features']}/{result['total_features']}")
    print(f"Max Drift Feature: {result['max_drift_feature']}")

    print("\n[TEST 3] Regime Drift - Crash Spike")
    print("=" * 50)
    new_regimes = ['crash'] * 40 + ['normal'] * 10  # 80% crash (was 10%)
    result = detector.detect_regime_drift(new_regimes, log_to_db=False)
    print(f"Drift Detected: {result['drift_detected']}")
    print(f"Regime Shift: {result['regime_shift']}")
    print(f"Max Change: {result['distribution_change']:.1%}")

    print("\n[TEST 4] Prediction Drift - Sell Bias")
    print("=" * 50)
    new_predictions = np.array([2] * 45 + [1] * 5)  # 90% sell predictions
    result = detector.detect_prediction_drift(new_predictions, log_to_db=False)
    print(f"Drift Detected: {result['drift_detected']}")
    print(f"KS Statistic: {result['ks_statistic']:.3f}")
    print(f"Shift: {result['prediction_shift']}")

    print("\n" + "=" * 50)
    print("[OK] Drift Detector Tests Complete")
    print("=" * 50)
