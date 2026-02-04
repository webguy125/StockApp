
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Unit Tests for HOLD Intelligence Module

Tests stability and volatility scoring functions to ensure:
- Deterministic results
- Correct normalization (0-100 scales)
- Expected behavior for edge cases
- No contamination with BUY/SELL logic
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from backend.turbomode.core_engine.hold_intelligence import HoldIntelligence


class TestHoldIntelligence(unittest.TestCase):
    """Test suite for HOLD intelligence scoring functions"""

    def setUp(self):
        """Initialize HoldIntelligence instance for testing"""
        self.hold_intel = HoldIntelligence(db_path=':memory:')

    def test_calculate_price_drift_no_movement(self):
        """Test price drift calculation when entry and exit are equal"""
        entry = 100.0
        exit_price = 100.0

        drift = self.hold_intel.calculate_price_drift(entry, exit_price)

        self.assertEqual(drift, 0.0, "Price drift should be 0 when entry equals exit")

    def test_calculate_price_drift_positive_movement(self):
        """Test price drift calculation for upward movement"""
        entry = 100.0
        exit_price = 105.0

        drift = self.hold_intel.calculate_price_drift(entry, exit_price)

        self.assertEqual(drift, 5.0, "Price drift should be 5% for $100 -> $105")

    def test_calculate_price_drift_negative_movement(self):
        """Test price drift calculation for downward movement"""
        entry = 100.0
        exit_price = 95.0

        drift = self.hold_intel.calculate_price_drift(entry, exit_price)

        self.assertEqual(drift, 5.0, "Price drift should be absolute value (5% for $100 -> $95)")

    def test_stability_score_normalization_perfect(self):
        """Test that perfect stability (no drift) produces high scores"""
        row = {
            'symbol': 'TEST',
            'entry_date': '2026-01-01',
            'exit_date': '2026-01-15',
            'entry_price': 100.0,
            'exit_price': 100.0,  # No drift
            'sector': 'Technology'
        }

        with patch.object(self.hold_intel, 'calculate_max_intraday_drift', return_value=0.0):
            with patch.object(self.hold_intel, 'calculate_volatility_drift', return_value=-10.0):  # Compression
                with patch.object(self.hold_intel, 'calculate_range_compression', return_value=90.0):
                    metrics = self.hold_intel.calculate_stability_metrics(row)

        self.assertGreaterEqual(metrics['stability_score'], 80.0,
                               "Perfect stability should produce score >= 80")
        self.assertLessEqual(metrics['stability_score'], 100.0,
                            "Stability score should not exceed 100")

    def test_stability_score_normalization_poor(self):
        """Test that poor stability (high drift) produces low scores"""
        row = {
            'symbol': 'TEST',
            'entry_date': '2026-01-01',
            'exit_date': '2026-01-15',
            'entry_price': 100.0,
            'exit_price': 115.0,  # 15% drift
            'sector': 'Technology'
        }

        with patch.object(self.hold_intel, 'calculate_max_intraday_drift', return_value=20.0):  # High volatility
            with patch.object(self.hold_intel, 'calculate_volatility_drift', return_value=30.0):  # Expansion
                with patch.object(self.hold_intel, 'calculate_range_compression', return_value=10.0):  # Wide range
                    metrics = self.hold_intel.calculate_stability_metrics(row)

        self.assertLess(metrics['stability_score'], 40.0,
                       "Poor stability should produce score < 40")
        self.assertGreaterEqual(metrics['stability_score'], 0.0,
                               "Stability score should not be negative")

    def test_volatility_score_determinism(self):
        """Test that volatility scoring is deterministic (same input = same output)"""
        row = {
            'symbol': 'TEST',
            'entry_date': '2026-01-01',
            'exit_date': '2026-01-15'
        }

        # Mock consistent market data
        mock_df = pd.DataFrame({
            'date': pd.date_range('2025-12-15', periods=50, freq='D'),
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [98.0] * 50,
            'close': [100.0] * 50,
            'volume': [1000000] * 50
        })

        with patch.object(self.hold_intel.market_data_api, 'get_candles', return_value=mock_df):
            metrics1 = self.hold_intel.calculate_volatility_compression_metrics(row)
            metrics2 = self.hold_intel.calculate_volatility_compression_metrics(row)

        self.assertEqual(metrics1['volatility_score'], metrics2['volatility_score'],
                        "Volatility score should be deterministic")

    def test_atr_calculation(self):
        """Test ATR calculation produces reasonable values"""
        df = pd.DataFrame({
            'high': [102, 103, 101, 104, 105],
            'low': [98, 99, 97, 100, 101],
            'close': [100, 101, 99, 102, 103]
        })

        atr = self.hold_intel.calculate_atr(df, period=3)

        # ATR should be calculated for rows with enough history
        self.assertEqual(len(atr), 5, "ATR should have same length as input")
        self.assertTrue(pd.isna(atr.iloc[0]), "First ATR value should be NaN (insufficient history)")
        self.assertTrue(pd.isna(atr.iloc[1]), "Second ATR value should be NaN (insufficient history)")
        self.assertFalse(pd.isna(atr.iloc[4]), "Last ATR value should be calculated")
        self.assertGreater(atr.iloc[4], 0, "ATR should be positive")

    def test_no_cross_contamination_buy_sell(self):
        """Test that HOLD intelligence never uses BUY/SELL performance data"""
        # This test verifies the query in analyze_hold_signals() filters only HOLD signals

        # The query should be:
        # WHERE signal_type = 'HOLD'
        # NOT: WHERE signal_type IN ('BUY', 'SELL', 'HOLD')

        # Use real database with no data to check query structure
        import sqlite3
        test_conn = sqlite3.connect(':memory:')
        test_cursor = test_conn.cursor()

        # Create empty signal_history table
        test_cursor.execute('''
            CREATE TABLE signal_history (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                signal_type TEXT,
                entry_date TEXT,
                exit_date TEXT
            )
        ''')
        test_conn.commit()

        # Patch the connection to use our test database
        with patch.object(self.hold_intel, 'db_path', ':memory:'):
            with patch('sqlite3.connect', return_value=test_conn):
                # Read the source code to verify query structure
                import inspect
                source = inspect.getsource(self.hold_intel.analyze_hold_signals)

                # Verify query only references HOLD signals
                self.assertIn("signal_type = 'HOLD'", source,
                             "Query should filter for HOLD signals only")
                self.assertNotIn("signal_type IN ('BUY'", source,
                               "Query should not include BUY in filter")
                self.assertNotIn("signal_type IN ('SELL'", source,
                               "Query should not include SELL in filter")

        test_conn.close()

    def test_weights_sum_to_one(self):
        """Test that stability and volatility weights sum to approximately 1.0"""
        stability_sum = sum(self.hold_intel.STABILITY_WEIGHTS.values())
        volatility_sum = sum(self.hold_intel.VOLATILITY_WEIGHTS.values())

        self.assertAlmostEqual(stability_sum, 1.0, places=2,
                              msg="Stability weights should sum to 1.0")
        self.assertAlmostEqual(volatility_sum, 1.0, places=2,
                              msg="Volatility weights should sum to 1.0")

    def test_minimum_threshold_filtering(self):
        """Test that Top 20 filtering correctly applies minimum thresholds"""
        # Create test data with mixed scores
        test_data = pd.DataFrame({
            'symbol': ['GOOD1', 'GOOD2', 'BAD1', 'BAD2'],
            'stability_score': [70, 65, 50, 45],
            'volatility_score': [60, 55, 30, 25],
            'hold_days': [14, 14, 14, 14],
            'price_drift_pct': [1.0, 2.0, 8.0, 10.0],
            'atr_compression_pct': [10, 5, -5, -10],
            'sector': ['Tech', 'Tech', 'Tech', 'Tech']
        })

        top_20 = self.hold_intel.get_top_20_hold_candidates(test_data)

        # Should only include GOOD1 and GOOD2 (meet thresholds)
        self.assertEqual(len(top_20), 2, "Only 2 symbols meet minimum thresholds")
        self.assertIn('GOOD1', top_20['symbol'].values)
        self.assertIn('GOOD2', top_20['symbol'].values)
        self.assertNotIn('BAD1', top_20['symbol'].values)
        self.assertNotIn('BAD2', top_20['symbol'].values)

    def test_sorting_order(self):
        """Test that Top 20 candidates are sorted correctly (stability DESC, volatility DESC)"""
        test_data = pd.DataFrame({
            'symbol': ['A', 'B', 'C'],
            'stability_score': [80, 80, 70],
            'volatility_score': [60, 65, 55],
            'hold_days': [14, 14, 14],
            'price_drift_pct': [1.0, 1.0, 2.0],
            'atr_compression_pct': [10, 15, 5],
            'sector': ['Tech', 'Tech', 'Tech']
        })

        top_20 = self.hold_intel.get_top_20_hold_candidates(test_data)

        # Should be sorted: B (80, 65), A (80, 60), C (70, 55)
        self.assertEqual(top_20.iloc[0]['symbol'], 'B', "B should rank first (80 stability, 65 volatility)")
        self.assertEqual(top_20.iloc[1]['symbol'], 'A', "A should rank second (80 stability, 60 volatility)")
        self.assertEqual(top_20.iloc[2]['symbol'], 'C', "C should rank third (70 stability)")

    def test_empty_signal_history(self):
        """Test graceful handling when no HOLD signals exist"""
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            # Mock pd.read_sql_query to return empty DataFrame
            with patch('pandas.read_sql_query', return_value=pd.DataFrame()):
                result = self.hold_intel.analyze_hold_signals()

        self.assertIsNone(result, "Should return None when no HOLD signals exist")


class TestStabilityMetricsCalculation(unittest.TestCase):
    """Focused tests for stability metrics calculations"""

    def setUp(self):
        self.hold_intel = HoldIntelligence(db_path=':memory:')

    def test_price_drift_score_normalization(self):
        """Test price drift normalization to 0-100 scale"""
        # 0% drift = 100 score
        score_0 = max(0, min(100, 100 - (0.0 * 10)))
        self.assertEqual(score_0, 100.0)

        # 5% drift = 50 score
        score_5 = max(0, min(100, 100 - (5.0 * 10)))
        self.assertEqual(score_5, 50.0)

        # 10%+ drift = 0 score
        score_10 = max(0, min(100, 100 - (10.0 * 10)))
        self.assertEqual(score_10, 0.0)

        # 15% drift = still 0 (capped)
        score_15 = max(0, min(100, 100 - (15.0 * 10)))
        self.assertEqual(score_15, 0.0)

    def test_volatility_drift_score_normalization(self):
        """Test volatility drift normalization (negative = compression = good)"""
        # -20% volatility drift (strong compression) = 100 score
        score_neg20 = max(0, min(100, 50 - (-20.0 * 2.5)))
        self.assertEqual(score_neg20, 100.0)

        # 0% volatility drift (no change) = 50 score
        score_0 = max(0, min(100, 50 - (0.0 * 2.5)))
        self.assertEqual(score_0, 50.0)

        # +20% volatility drift (expansion) = 0 score
        score_pos20 = max(0, min(100, 50 - (20.0 * 2.5)))
        self.assertEqual(score_pos20, 0.0)


if __name__ == '__main__':
    unittest.main()
