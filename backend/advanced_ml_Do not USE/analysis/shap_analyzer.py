"""
SHAP Feature Analysis Module

Provides model interpretability using SHAP values.

Strategy:
- Calculate SHAP values for each prediction
- Identify top contributing features
- Store feature importance by regime
- Generate explanation reports

SHAP (SHapley Additive exPlanations) provides:
- Model-agnostic explanations
- Consistent feature attribution
- Local and global interpretability
"""

import sqlite3
import json
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARNING] SHAP library not installed. Run: pip install shap")


class SHAPAnalyzer:
    """
    SHAP-based model interpretability

    Provides:
    - Per-prediction explanations
    - Global feature importance
    - Regime-specific feature rankings
    - Feature interaction analysis
    """

    def __init__(self, db_path: str = "backend/data/advanced_ml_system.db"):
        """
        Initialize SHAP analyzer

        Args:
            db_path: Path to ML database
        """
        self.db_path = db_path

        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed. Run: pip install shap")

        self._init_table()

    def _init_table(self):
        """Create shap_analysis table if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shap_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                regime TEXT,
                feature_name TEXT NOT NULL,
                importance_score REAL NOT NULL,
                rank INTEGER,
                analysis_date TEXT DEFAULT CURRENT_TIMESTAMP,
                sample_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_shap_model_regime
            ON shap_analysis(model_version, regime)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_shap_feature
            ON shap_analysis(feature_name)
        ''')

        conn.commit()
        conn.close()

    def explain_prediction(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Get SHAP values for a single prediction

        Args:
            model: Trained model (XGBoost, RandomForest, etc.)
            features: Feature vector (1D array)
            feature_names: List of feature names
            background_data: Background dataset for SHAP explainer

        Returns:
            Dictionary with SHAP values and top features
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed")

        try:
            # Use TreeExplainer for tree-based models (faster and more accurate)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values_raw = explainer.shap_values(features.reshape(1, -1))
                base_value_raw = explainer.expected_value
            except:
                # Fall back to general explainer
                if background_data is not None:
                    explainer = shap.Explainer(model, background_data[:100])
                else:
                    explainer = shap.Explainer(model)
                shap_output = explainer(features.reshape(1, -1))
                shap_values_raw = shap_output.values[0] if hasattr(shap_output, 'values') else shap_output[0]
                base_value_raw = shap_output.base_values[0] if hasattr(shap_output, 'base_values') else explainer.expected_value

            # Handle different output formats
            # TreeExplainer with binary classification returns list of 2 arrays
            if isinstance(shap_values_raw, list):
                # Binary classification: use positive class (index 1)
                values = shap_values_raw[1][0] if isinstance(shap_values_raw[1], np.ndarray) else shap_values_raw[1]
                base_value = base_value_raw[1] if isinstance(base_value_raw, (list, np.ndarray)) and len(base_value_raw) > 1 else base_value_raw
            elif isinstance(shap_values_raw, np.ndarray):
                if shap_values_raw.ndim == 2:
                    # Shape: (classes, features) - take positive class
                    values = shap_values_raw[1] if shap_values_raw.shape[0] == 2 else shap_values_raw[0]
                else:
                    values = shap_values_raw
                base_value = base_value_raw
            else:
                values = shap_values_raw
                base_value = base_value_raw

            # Ensure base_value is a scalar
            if isinstance(base_value, (list, np.ndarray)):
                if hasattr(base_value, '__len__') and len(base_value) > 0:
                    base_value = base_value[1] if len(base_value) == 2 else base_value[0]
                    if isinstance(base_value, np.ndarray):
                        base_value = base_value.item() if base_value.size == 1 else float(base_value[0])
            base_value = float(base_value) if not isinstance(base_value, float) else base_value

            # Create feature importance dict
            feature_importance = {
                feature_names[i]: float(values[i])
                for i in range(len(feature_names))
            }

            # Get top contributing features
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            top_features = sorted_features[:10]

            # Separate positive and negative contributions
            positive_features = [(name, val) for name, val in sorted_features if val > 0][:5]
            negative_features = [(name, val) for name, val in sorted_features if val < 0][:5]

            return {
                'shap_values': feature_importance,
                'base_value': float(base_value),
                'top_features': top_features,
                'positive_contributors': positive_features,
                'negative_contributors': negative_features,
                'prediction_explanation': self._format_explanation(top_features, base_value)
            }

        except Exception as e:
            print(f"[ERROR] SHAP explanation failed: {e}")
            return {
                'error': str(e),
                'shap_values': {},
                'top_features': []
            }

    def _format_explanation(self, top_features: List[Tuple[str, float]], base_value: float) -> str:
        """Format human-readable explanation"""
        explanation = f"Base prediction: {base_value:.4f}\n\n"
        explanation += "Top contributing features:\n"

        for i, (feature, value) in enumerate(top_features, 1):
            direction = "↑" if value > 0 else "↓"
            explanation += f"{i}. {feature}: {value:+.4f} {direction}\n"

        return explanation

    def get_feature_importance(
        self,
        model: Any,
        X_test: np.ndarray,
        feature_names: List[str],
        model_version: str,
        regime: Optional[str] = None,
        sample_size: int = 100
    ) -> Dict[str, float]:
        """
        Calculate global feature importance using SHAP

        Args:
            model: Trained model
            X_test: Test dataset
            feature_names: List of feature names
            model_version: Model version identifier
            regime: Market regime (optional)
            sample_size: Number of samples to use for SHAP

        Returns:
            Dictionary of feature importances
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed")

        try:
            print(f"\n[SHAP] Calculating global feature importance...")
            print(f"  Model: {model_version}")
            if regime:
                print(f"  Regime: {regime}")
            print(f"  Samples: {min(sample_size, len(X_test))}")

            # Sample data if too large
            if len(X_test) > sample_size:
                indices = np.random.choice(len(X_test), sample_size, replace=False)
                X_sample = X_test[indices]
            else:
                X_sample = X_test

            # Use TreeExplainer for tree-based models (faster and more accurate)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values_raw = explainer.shap_values(X_sample)
            except:
                # Fall back to general explainer
                explainer = shap.Explainer(model, X_sample[:50])
                shap_output = explainer(X_sample)
                shap_values_raw = shap_output.values if hasattr(shap_output, 'values') else shap_output

            # Handle different output formats
            # TreeExplainer with binary classification returns list of 2 arrays
            if isinstance(shap_values_raw, list):
                # Binary classification: use positive class (index 1)
                # Shape: list of 2 arrays, each (n_samples, n_features)
                values = shap_values_raw[1]
            elif isinstance(shap_values_raw, np.ndarray):
                if shap_values_raw.ndim == 3:
                    # Shape: (n_samples, n_classes, n_features)
                    values = shap_values_raw[:, 1, :] if shap_values_raw.shape[1] == 2 else shap_values_raw[:, 0, :]
                elif shap_values_raw.ndim == 2:
                    # Shape: (n_samples, n_features) - already correct
                    values = shap_values_raw
                else:
                    values = shap_values_raw
            else:
                values = shap_values_raw

            # Calculate mean absolute SHAP value for each feature
            mean_abs_shap = np.abs(values).mean(axis=0)

            # Create feature importance dict
            feature_importance = {
                feature_names[i]: float(mean_abs_shap[i])
                for i in range(len(feature_names))
            }

            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            print(f"\n  Top 10 Features:")
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"    {i}. {feature}: {importance:.6f}")

            # Save to database
            self._save_feature_importance(
                model_version=model_version,
                regime=regime,
                feature_importance=dict(sorted_features),
                sample_count=len(X_sample)
            )

            return dict(sorted_features)

        except Exception as e:
            print(f"[ERROR] Global feature importance failed: {e}")
            return {}

    def analyze_by_regime(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        regimes: np.ndarray,
        feature_names: List[str],
        model_version: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Get regime-specific feature rankings

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            regimes: Regime labels for each sample
            feature_names: List of feature names
            model_version: Model version identifier

        Returns:
            Dictionary mapping regime to feature importance
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed")

        print(f"\n[SHAP] Analyzing by regime...")

        regime_importance = {}
        unique_regimes = np.unique(regimes)

        for regime in unique_regimes:
            print(f"\n  Regime: {regime}")

            # Filter data for this regime
            regime_mask = regimes == regime
            X_regime = X_test[regime_mask]

            if len(X_regime) == 0:
                print(f"    [SKIP] No samples for regime {regime}")
                continue

            # Calculate feature importance for this regime
            importance = self.get_feature_importance(
                model=model,
                X_test=X_regime,
                feature_names=feature_names,
                model_version=model_version,
                regime=regime,
                sample_size=min(100, len(X_regime))
            )

            regime_importance[regime] = importance

        return regime_importance

    def _save_feature_importance(
        self,
        model_version: str,
        feature_importance: Dict[str, float],
        regime: Optional[str] = None,
        sample_count: int = 0
    ):
        """
        Save feature importance to database

        Args:
            model_version: Model version identifier
            feature_importance: Dictionary of feature importances
            regime: Market regime (optional)
            sample_count: Number of samples analyzed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete existing entries for this model/regime
        if regime:
            cursor.execute('''
                DELETE FROM shap_analysis
                WHERE model_version = ? AND regime = ?
            ''', (model_version, regime))
        else:
            cursor.execute('''
                DELETE FROM shap_analysis
                WHERE model_version = ? AND regime IS NULL
            ''', (model_version,))

        # Insert new entries
        for rank, (feature_name, importance_score) in enumerate(feature_importance.items(), 1):
            cursor.execute('''
                INSERT INTO shap_analysis
                (model_version, regime, feature_name, importance_score, rank, analysis_date, sample_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_version,
                regime,
                feature_name,
                float(importance_score),
                rank,
                datetime.now().isoformat(),
                sample_count
            ))

        conn.commit()
        conn.close()

        print(f"[OK] Saved {len(feature_importance)} feature importances to database")

    def get_top_features_by_regime(
        self,
        model_version: str,
        regime: Optional[str] = None,
        limit: int = 10
    ) -> List[Tuple[str, float, int]]:
        """
        Get top features from database

        Args:
            model_version: Model version identifier
            regime: Market regime (optional)
            limit: Number of top features to return

        Returns:
            List of (feature_name, importance_score, rank) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if regime:
            cursor.execute('''
                SELECT feature_name, importance_score, rank
                FROM shap_analysis
                WHERE model_version = ? AND regime = ?
                ORDER BY rank ASC
                LIMIT ?
            ''', (model_version, regime, limit))
        else:
            cursor.execute('''
                SELECT feature_name, importance_score, rank
                FROM shap_analysis
                WHERE model_version = ? AND regime IS NULL
                ORDER BY rank ASC
                LIMIT ?
            ''', (model_version, limit))

        results = cursor.fetchall()
        conn.close()

        return results

    def compare_regime_features(
        self,
        model_version: str,
        limit: int = 5
    ) -> Dict[str, List[str]]:
        """
        Compare top features across regimes

        Args:
            model_version: Model version identifier
            limit: Number of top features per regime

        Returns:
            Dictionary mapping regime to top feature names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all regimes
        cursor.execute('''
            SELECT DISTINCT regime
            FROM shap_analysis
            WHERE model_version = ? AND regime IS NOT NULL
        ''', (model_version,))

        regimes = [row[0] for row in cursor.fetchall()]
        conn.close()

        comparison = {}
        for regime in regimes:
            top_features = self.get_top_features_by_regime(model_version, regime, limit)
            comparison[regime] = [name for name, _, _ in top_features]

        return comparison

    def generate_report(self, model_version: str) -> str:
        """
        Generate SHAP analysis report

        Args:
            model_version: Model version identifier

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append(f"SHAP Feature Analysis Report")
        report.append(f"Model: {model_version}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")

        # Global feature importance
        report.append("Global Feature Importance (Top 15):")
        report.append("-" * 80)
        top_features = self.get_top_features_by_regime(model_version, regime=None, limit=15)
        for i, (feature, importance, rank) in enumerate(top_features, 1):
            report.append(f"  {i:2d}. {feature:40s} {importance:.6f}")
        report.append("")

        # Regime-specific comparison
        regime_comparison = self.compare_regime_features(model_version, limit=5)
        if regime_comparison:
            report.append("Regime-Specific Top Features:")
            report.append("-" * 80)
            for regime, features in regime_comparison.items():
                report.append(f"\n  {regime.upper()}:")
                for i, feature in enumerate(features, 1):
                    report.append(f"    {i}. {feature}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


if __name__ == '__main__':
    # Test SHAP analyzer
    print("Testing SHAP Analyzer...\n")

    if not SHAP_AVAILABLE:
        print("[ERROR] SHAP library not installed. Run: pip install shap")
        print("Skipping SHAP tests.")
        sys.exit(1)

    # Create analyzer
    analyzer = SHAPAnalyzer()

    print("[TEST 1] Create synthetic model and data")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier

    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(1000, 50)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    X_test = np.random.randn(200, 50)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    feature_names = [f'feature_{i}' for i in range(50)]

    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_names)}")
    print()

    print("[TEST 2] Explain single prediction")
    print("=" * 60)
    explanation = analyzer.explain_prediction(
        model=model,
        features=X_test[0],
        feature_names=feature_names,
        background_data=X_train[:100]
    )

    if 'error' not in explanation:
        print(explanation['prediction_explanation'])
        print()

    print("[TEST 3] Global feature importance")
    print("=" * 60)
    importance = analyzer.get_feature_importance(
        model=model,
        X_test=X_test,
        feature_names=feature_names,
        model_version="test_v1",
        sample_size=100
    )
    print()

    print("[TEST 4] Regime-specific analysis")
    print("=" * 60)
    regimes = np.random.choice(['normal', 'crash', 'recovery'], size=len(X_test))
    regime_importance = analyzer.analyze_by_regime(
        model=model,
        X_test=X_test,
        y_test=y_test,
        regimes=regimes,
        feature_names=feature_names,
        model_version="test_v1"
    )
    print()

    print("[TEST 5] Generate report")
    print("=" * 60)
    report = analyzer.generate_report("test_v1")
    print(report)
    print()

    print("=" * 60)
    print("[OK] SHAP Analyzer Tests Complete")
    print("=" * 60)
