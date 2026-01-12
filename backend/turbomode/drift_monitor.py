"""
TurboMode Drift Monitoring System
Detects data drift and model drift using PSI, KL Divergence, and KS Statistic

Per MASTER_MARKET_DATA_ARCHITECTURE.json v1.1:
- All drift thresholds loaded from turbomode_training_config.json
- Drift metrics logged to drift_monitoring table in TurboMode DB
- Alerts triggered when thresholds exceeded

Author: TurboMode System
Date: 2026-01-06
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

# TurboMode Config (all thresholds from JSON)
from turbomode.config.config_loader import get_config

# TurboMode DB
from turbomode.database_schema import TurboModeDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('drift_monitor')


class DriftMonitor:
    """
    Monitors data drift and model drift for TurboMode

    Features:
    - PSI (Population Stability Index) for feature drift
    - KL Divergence for distribution drift
    - KS Statistic for two-sample comparison
    - Config-driven thresholds
    - Alert system integrated with TurboMode DB
    """

    def __init__(self, turbomode_db_path: str = "backend/data/turbomode.db"):
        """
        Initialize drift monitor

        Args:
            turbomode_db_path: Path to TurboMode database
        """
        # Load config (all thresholds from JSON)
        self.config = get_config()
        self.drift_thresholds = self.config.get_drift_thresholds()
        logger.info(f"[INIT] Loaded drift thresholds from config")

        # Connect to TurboMode DB
        self.turbomode_db = TurboModeDB(db_path=turbomode_db_path)
        logger.info(f"[INIT] Connected to TurboMode DB")

    # ========================================================================
    # PSI (Population Stability Index)
    # ========================================================================

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)

        PSI measures how much a distribution has shifted
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change (action required)

        Args:
            expected: Expected (baseline) distribution
            actual: Actual (current) distribution
            bins: Number of bins for histogram

        Returns:
            PSI value
        """
        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        # Count frequencies in each bin
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        # Calculate PSI
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

        return psi_value

    # ========================================================================
    # KL Divergence
    # ========================================================================

    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Kullback-Leibler Divergence

        KL Divergence measures how one distribution differs from a reference distribution
        - Always non-negative
        - Higher values indicate more divergence

        Args:
            p: Reference distribution
            q: Comparison distribution
            bins: Number of bins

        Returns:
            KL divergence value
        """
        # Create histograms
        breakpoints = np.linspace(min(p.min(), q.min()), max(p.max(), q.max()), bins + 1)

        p_hist = np.histogram(p, bins=breakpoints)[0] / len(p)
        q_hist = np.histogram(q, bins=breakpoints)[0] / len(q)

        # Avoid zeros
        p_hist = np.where(p_hist == 0, 1e-10, p_hist)
        q_hist = np.where(q_hist == 0, 1e-10, q_hist)

        # Calculate KL divergence
        kl_div = np.sum(p_hist * np.log(p_hist / q_hist))

        return kl_div

    # ========================================================================
    # KS Statistic (Kolmogorov-Smirnov)
    # ========================================================================

    def calculate_ks_statistic(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov test statistic

        KS test measures maximum distance between two cumulative distributions
        - Returns (statistic, p-value)
        - Low p-value (< 0.05) indicates significant difference

        Args:
            baseline: Baseline distribution
            current: Current distribution

        Returns:
            Tuple of (KS statistic, p-value)
        """
        ks_statistic, p_value = stats.ks_2samp(baseline, current)

        return ks_statistic, p_value

    # ========================================================================
    # DRIFT DETECTION
    # ========================================================================

    def detect_feature_drift(self,
                            baseline_features: pd.DataFrame,
                            current_features: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect drift in feature distributions

        Args:
            baseline_features: Baseline feature DataFrame
            current_features: Current feature DataFrame

        Returns:
            Dictionary of feature -> drift metrics
        """
        logger.info("[DRIFT DETECTION] Analyzing feature drift...")

        drift_results = {}
        psi_threshold = self.drift_thresholds.get('psi_threshold', 0.10)
        kl_threshold = self.drift_thresholds.get('kl_divergence_threshold', 0.15)

        # Analyze each feature
        for feature in baseline_features.columns:
            if feature not in current_features.columns:
                continue

            baseline_values = baseline_features[feature].dropna().values
            current_values = current_features[feature].dropna().values

            if len(baseline_values) < 100 or len(current_values) < 100:
                continue

            # Calculate drift metrics
            psi = self.calculate_psi(baseline_values, current_values)
            kl_div = self.calculate_kl_divergence(baseline_values, current_values)
            ks_stat, ks_pval = self.calculate_ks_statistic(baseline_values, current_values)

            # Check thresholds
            psi_alert = psi > psi_threshold
            kl_alert = kl_div > kl_threshold
            ks_alert = ks_pval < 0.05

            drift_results[feature] = {
                'psi_score': float(psi),
                'kl_divergence': float(kl_div),
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'psi_alert': psi_alert,
                'kl_alert': kl_alert,
                'ks_alert': ks_alert,
                'any_alert': psi_alert or kl_alert or ks_alert
            }

        # Count alerts
        num_alerts = sum(1 for v in drift_results.values() if v['any_alert'])

        logger.info(f"[DRIFT DETECTION] Analyzed {len(drift_results)} features")
        logger.info(f"                  {num_alerts} features with drift alerts")

        return drift_results

    def detect_prediction_drift(self,
                               baseline_predictions: np.ndarray,
                               current_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in model predictions

        Args:
            baseline_predictions: Baseline prediction probabilities
            current_predictions: Current prediction probabilities

        Returns:
            Drift metrics dictionary
        """
        logger.info("[DRIFT DETECTION] Analyzing prediction drift...")

        psi = self.calculate_psi(baseline_predictions, current_predictions)
        kl_div = self.calculate_kl_divergence(baseline_predictions, current_predictions)
        ks_stat, ks_pval = self.calculate_ks_statistic(baseline_predictions, current_predictions)

        psi_threshold = self.drift_thresholds.get('psi_threshold', 0.10)
        kl_threshold = self.drift_thresholds.get('kl_divergence_threshold', 0.15)

        results = {
            'psi_score': float(psi),
            'kl_divergence': float(kl_div),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'psi_alert': psi > psi_threshold,
            'kl_alert': kl_div > kl_threshold,
            'ks_alert': ks_pval < 0.05
        }

        results['any_alert'] = results['psi_alert'] or results['kl_alert'] or results['ks_alert']

        if results['any_alert']:
            logger.warning(f"[ALERT] Prediction drift detected! PSI={psi:.3f}, KL={kl_div:.3f}")
        else:
            logger.info(f"[OK] No significant prediction drift")

        return results

    # ========================================================================
    # LOGGING TO DATABASE
    # ========================================================================

    def log_drift_to_db(self, drift_type: str, metric_name: str, metric_value: float,
                       threshold: float, alert_triggered: bool,
                       symbol: Optional[str] = None, feature_name: Optional[str] = None,
                       **kwargs):
        """
        Log drift metrics to TurboMode DB drift_monitoring table

        Args:
            drift_type: Type of drift ('feature_drift', 'prediction_drift', 'performance_drift')
            metric_name: Name of metric ('psi', 'kl_divergence', 'ks_statistic')
            metric_value: Value of metric
            threshold: Threshold value
            alert_triggered: Whether alert was triggered
            symbol: Optional stock symbol
            feature_name: Optional feature name
            **kwargs: Additional metrics (psi_score, kl_divergence, ks_statistic)
        """
        conn = self.turbomode_db.conn
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO drift_monitoring (
                drift_type, metric_name, metric_value, threshold, alert_triggered,
                symbol, feature_name, psi_score, kl_divergence, ks_statistic,
                detection_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            drift_type,
            metric_name,
            metric_value,
            threshold,
            1 if alert_triggered else 0,
            symbol,
            feature_name,
            kwargs.get('psi_score'),
            kwargs.get('kl_divergence'),
            kwargs.get('ks_statistic'),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))

        conn.commit()

    def log_feature_drift_batch(self, drift_results: Dict[str, Dict]):
        """
        Log batch of feature drift results to database

        Args:
            drift_results: Dictionary of feature -> drift metrics
        """
        logger.info("[LOGGING] Writing drift metrics to TurboMode DB...")

        psi_threshold = self.drift_thresholds.get('psi_threshold', 0.10)
        kl_threshold = self.drift_thresholds.get('kl_divergence_threshold', 0.15)

        for feature_name, metrics in drift_results.items():
            # Log PSI
            self.log_drift_to_db(
                drift_type='feature_drift',
                metric_name='psi',
                metric_value=metrics['psi_score'],
                threshold=psi_threshold,
                alert_triggered=metrics['psi_alert'],
                feature_name=feature_name,
                **metrics
            )

        logger.info(f"[LOGGING] ✓ Logged {len(drift_results)} feature drift records")

    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================

    def run_drift_monitoring(self,
                            baseline_features: pd.DataFrame,
                            current_features: pd.DataFrame,
                            baseline_predictions: Optional[np.ndarray] = None,
                            current_predictions: Optional[np.ndarray] = None):
        """
        Run complete drift monitoring pipeline

        Args:
            baseline_features: Baseline feature DataFrame
            current_features: Current feature DataFrame
            baseline_predictions: Optional baseline predictions
            current_predictions: Optional current predictions
        """
        logger.info("=" * 80)
        logger.info("TURBOMODE DRIFT MONITORING")
        logger.info("=" * 80)

        # Feature drift
        feature_drift = self.detect_feature_drift(baseline_features, current_features)
        self.log_feature_drift_batch(feature_drift)

        # Prediction drift (if provided)
        if baseline_predictions is not None and current_predictions is not None:
            pred_drift = self.detect_prediction_drift(baseline_predictions, current_predictions)

            self.log_drift_to_db(
                drift_type='prediction_drift',
                metric_name='psi',
                metric_value=pred_drift['psi_score'],
                threshold=self.drift_thresholds.get('psi_threshold', 0.10),
                alert_triggered=pred_drift['any_alert'],
                **pred_drift
            )

        logger.info("=" * 80)
        logger.info("DRIFT MONITORING COMPLETE")
        logger.info("=" * 80)


if __name__ == '__main__':
    print("=" * 80)
    print("TURBOMODE DRIFT MONITORING SYSTEM")
    print("=" * 80)

    # Test with synthetic data
    np.random.seed(42)

    # Baseline data
    baseline = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000),
        'feature_3': np.random.uniform(0, 10, 1000)
    })

    # Current data with drift
    current = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, 1000),  # Slight drift
        'feature_2': np.random.normal(5, 2, 1000),      # No drift
        'feature_3': np.random.uniform(2, 12, 1000)     # Significant drift
    })

    # Run drift monitoring
    monitor = DriftMonitor()
    monitor.run_drift_monitoring(baseline, current)

    print("\n[OK] Drift monitoring test complete!")
