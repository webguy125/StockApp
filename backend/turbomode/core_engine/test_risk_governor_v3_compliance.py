#!/usr/bin/env python
"""
Risk Governor v3.0.0 Compliance Test Suite

Tests the complete v3.0.0 specification including:
- "at least 2 of 4 metrics" transition logic with time windows
- Sustained recovery requirements
- Exact threshold alignment
- Action flag verification (freeze_new_entries, ATR multipliers, confidence penalty)

Author: TurboMode Production Team
Date: 2026-01-25
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta
from backend.turbomode.core_engine.risk_governor_daemon import (
    RiskGovernorDB,
    RiskGovernorConfig,
    StateMachineEvaluator,
    MetricHistory
)


class V3ComplianceTestSuite:
    """Test suite for v3.0.0 specification compliance."""

    def __init__(self):
        self.db = RiskGovernorDB(RiskGovernorConfig.DB_PATH)
        self.evaluator = StateMachineEvaluator(self.db)
        self.passed = 0
        self.failed = 0

    def assert_equal(self, actual, expected, message):
        """Assert two values are equal."""
        if actual == expected:
            print(f"  [PASS] {message}: {actual}")
            self.passed += 1
        else:
            print(f"  [FAIL] {message}: Expected {expected}, got {actual}")
            self.failed += 1
            raise AssertionError(f"{message}: Expected {expected}, got {actual}")

    def assert_true(self, condition, message):
        """Assert condition is true."""
        if condition:
            print(f"  [PASS] {message}")
            self.passed += 1
        else:
            print(f"  [FAIL] {message} (failed)")
            self.failed += 1
            raise AssertionError(message)

    # ========================================================================
    # TEST 1: NORMAL -> CAUTION requires 2 of 4 metrics within 10 min
    # ========================================================================

    def test_1_normal_to_caution_with_2_metrics(self):
        """Test NORMAL -> CAUTION with exactly 2 of 4 metrics elevated."""
        print("\n" + "=" * 80)
        print("TEST 1: NORMAL -> CAUTION with 2 of 4 metrics (v3.0.0)")
        print("=" * 80)

        # Create fresh metric history
        self.evaluator.metric_history = MetricHistory(max_window_sec=3600)

        # Add samples with 2 metrics above CAUTION thresholds
        metrics = {
            'volatility_ratio': 2.0,  # ABOVE CAUTION (1.8) [OK]
            'confidence_collapse_pct': 0.25,  # ABOVE CAUTION (0.20) [OK]
            'liquidity_stress': 0.003,  # BELOW CAUTION (0.004) [X]
            'portfolio_drawdown': 0.01  # BELOW CAUTION (0.03) [X]
        }

        print("\nSimulated metrics (2 of 4 above CAUTION):")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Add sample to history
        self.evaluator.metric_history.add_sample(metrics)

        # Evaluate from NORMAL state
        evaluation = self.evaluator.evaluate_transition(
            current_state=RiskGovernorConfig.STATE_NORMAL,
            metrics=metrics,
            entered_at=None
        )

        print(f"\nTransition evaluation:")
        print(f"  new_state: {evaluation['new_state']}")
        print(f"  reason: {evaluation['reason']}")

        # Assertions per v3.0.0
        self.assert_equal(
            evaluation['new_state'],
            RiskGovernorConfig.STATE_CAUTION,
            "State transitioned to CAUTION"
        )
        self.assert_equal(
            evaluation['freeze_new_entries'],
            False,
            "freeze_new_entries = False (v3.0.0: CAUTION allows entries)"
        )
        self.assert_equal(
            evaluation['tighten_exits'],
            True,
            "tighten_exits = True"
        )
        self.assert_equal(
            evaluation['atr_multiplier'],
            0.8,
            "atr_multiplier = 0.8 (v3.0.0 CAUTION)"
        )

        print("\n[PASS] Test 1: NORMAL -> CAUTION with 2 of 4 metrics")

    # ========================================================================
    # TEST 2: NORMAL stays NORMAL with only 1 metric elevated
    # ========================================================================

    def test_2_normal_stays_normal_with_1_metric(self):
        """Test NORMAL stays NORMAL with only 1 of 4 metrics elevated."""
        print("\n" + "=" * 80)
        print("TEST 2: NORMAL stays NORMAL with 1 of 4 metrics (v3.0.0)")
        print("=" * 80)

        # Create fresh metric history
        self.evaluator.metric_history = MetricHistory(max_window_sec=3600)

        # Add samples with only 1 metric above CAUTION threshold
        metrics = {
            'volatility_ratio': 2.0,  # ABOVE CAUTION (1.8) [OK]
            'confidence_collapse_pct': 0.10,  # BELOW CAUTION (0.20) [X]
            'liquidity_stress': 0.002,  # BELOW CAUTION (0.004) [X]
            'portfolio_drawdown': 0.01  # BELOW CAUTION (0.03) [X]
        }

        print("\nSimulated metrics (1 of 4 above CAUTION):")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Add sample to history
        self.evaluator.metric_history.add_sample(metrics)

        # Evaluate from NORMAL state
        evaluation = self.evaluator.evaluate_transition(
            current_state=RiskGovernorConfig.STATE_NORMAL,
            metrics=metrics,
            entered_at=None
        )

        print(f"\nTransition evaluation:")
        print(f"  new_state: {evaluation['new_state']}")
        print(f"  reason: {evaluation['reason']}")

        # Assertions per v3.0.0
        self.assert_equal(
            evaluation['new_state'],
            RiskGovernorConfig.STATE_NORMAL,
            "State remains NORMAL (< 2 metrics elevated)"
        )
        self.assert_equal(
            evaluation['freeze_new_entries'],
            False,
            "freeze_new_entries = False"
        )
        self.assert_equal(
            evaluation['tighten_exits'],
            False,
            "tighten_exits = False"
        )

        print("\n[PASS] Test 2: NORMAL stays NORMAL with 1 of 4 metrics")

    # ========================================================================
    # TEST 3: CAUTION -> CRITICAL with 2 of 4 metrics within 5 min
    # ========================================================================

    def test_3_caution_to_critical_with_2_metrics(self):
        """Test CAUTION -> CRITICAL with exactly 2 of 4 metrics at CRITICAL level."""
        print("\n" + "=" * 80)
        print("TEST 3: CAUTION -> CRITICAL with 2 of 4 metrics (v3.0.0)")
        print("=" * 80)

        # Create fresh metric history
        self.evaluator.metric_history = MetricHistory(max_window_sec=3600)

        # Add samples with 2 metrics above CRITICAL thresholds
        metrics = {
            'volatility_ratio': 2.6,  # ABOVE CRITICAL (2.4) [OK]
            'confidence_collapse_pct': 0.40,  # ABOVE CRITICAL (0.35) [OK]
            'liquidity_stress': 0.005,  # BELOW CRITICAL (0.008) [X]
            'portfolio_drawdown': 0.04  # BELOW CRITICAL (0.06) [X]
        }

        print("\nSimulated metrics (2 of 4 above CRITICAL):")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Add sample to history
        self.evaluator.metric_history.add_sample(metrics)

        # Evaluate from CAUTION state
        evaluation = self.evaluator.evaluate_transition(
            current_state=RiskGovernorConfig.STATE_CAUTION,
            metrics=metrics,
            entered_at=None
        )

        print(f"\nTransition evaluation:")
        print(f"  new_state: {evaluation['new_state']}")
        print(f"  reason: {evaluation['reason']}")

        # Assertions per v3.0.0
        self.assert_equal(
            evaluation['new_state'],
            RiskGovernorConfig.STATE_CRITICAL,
            "State transitioned to CRITICAL"
        )
        self.assert_equal(
            evaluation['freeze_new_entries'],
            True,
            "freeze_new_entries = True (v3.0.0: CRITICAL freezes entries)"
        )
        self.assert_equal(
            evaluation['tighten_exits'],
            True,
            "tighten_exits = True"
        )
        self.assert_equal(
            evaluation['atr_multiplier'],
            0.5,
            "atr_multiplier = 0.5 (v3.0.0 CRITICAL)"
        )
        self.assert_equal(
            evaluation['force_deleveraging'],
            True,
            "force_deleveraging = True"
        )
        self.assert_equal(
            evaluation['confidence_penalty_active'],
            True,
            "confidence_penalty_active = True"
        )
        self.assert_equal(
            evaluation['confidence_penalty_factor'],
            0.60,
            "confidence_penalty_factor = 0.60 (v3.0.0)"
        )

        print("\n[PASS] Test 3: CAUTION -> CRITICAL with 2 of 4 metrics")

    # ========================================================================
    # TEST 4: CAUTION stays CAUTION with only 1 CRITICAL metric
    # ========================================================================

    def test_4_caution_stays_caution_with_1_critical(self):
        """Test CAUTION stays CAUTION with only 1 of 4 metrics at CRITICAL."""
        print("\n" + "=" * 80)
        print("TEST 4: CAUTION stays CAUTION with 1 of 4 CRITICAL metrics (v3.0.0)")
        print("=" * 80)

        # Create fresh metric history
        self.evaluator.metric_history = MetricHistory(max_window_sec=3600)

        # Add samples with only 1 metric above CRITICAL threshold
        metrics = {
            'volatility_ratio': 2.6,  # ABOVE CRITICAL (2.4) [OK]
            'confidence_collapse_pct': 0.25,  # BELOW CRITICAL (0.35) [X]
            'liquidity_stress': 0.005,  # BELOW CRITICAL (0.008) [X]
            'portfolio_drawdown': 0.04  # BELOW CRITICAL (0.06) [X]
        }

        print("\nSimulated metrics (1 of 4 above CRITICAL):")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Add sample to history
        self.evaluator.metric_history.add_sample(metrics)

        # Evaluate from CAUTION state
        evaluation = self.evaluator.evaluate_transition(
            current_state=RiskGovernorConfig.STATE_CAUTION,
            metrics=metrics,
            entered_at=None
        )

        print(f"\nTransition evaluation:")
        print(f"  new_state: {evaluation['new_state']}")
        print(f"  reason: {evaluation['reason']}")

        # Assertions per v3.0.0
        self.assert_equal(
            evaluation['new_state'],
            RiskGovernorConfig.STATE_CAUTION,
            "State remains CAUTION (< 2 metrics at CRITICAL)"
        )
        self.assert_equal(
            evaluation['freeze_new_entries'],
            False,
            "freeze_new_entries = False (CAUTION allows entries)"
        )
        self.assert_equal(
            evaluation['atr_multiplier'],
            0.8,
            "atr_multiplier = 0.8 (CAUTION)"
        )

        print("\n[PASS] Test 4: CAUTION stays CAUTION with 1 of 4 CRITICAL metrics")

    # ========================================================================
    # TEST 5: CAUTION -> NORMAL recovery after 30 minutes
    # ========================================================================

    def test_5_caution_to_normal_recovery_30min(self):
        """Test CAUTION -> NORMAL requires all metrics below for 30 minutes."""
        print("\n" + "=" * 80)
        print("TEST 5: CAUTION -> NORMAL recovery (30 min sustained, v3.0.0)")
        print("=" * 80)

        # Create fresh metric history
        self.evaluator.metric_history = MetricHistory(max_window_sec=3600)

        # Define recovery metrics (all below recovery thresholds)
        recovery_metrics = {
            'volatility_ratio': 1.4,  # BELOW recovery (1.5)
            'confidence_collapse_pct': 0.10,  # BELOW recovery (0.15)
            'liquidity_stress': 0.002,  # BELOW recovery (0.003)
            'portfolio_drawdown': 0.01  # BELOW recovery (0.02)
        }

        print("\nSimulating 30 minutes of recovery metrics:")
        for key, value in recovery_metrics.items():
            print(f"  {key}: {value}")

        # Add samples every 5 seconds for 30 minutes (360 samples)
        # Simulate by adding samples with timestamps spanning 30+ minutes
        now = datetime.utcnow()

        # Add samples spanning 31 minutes (to exceed 30 min requirement)
        for i in range(372):  # 31 min * 60 sec / 5 sec = 372 samples
            sample_time = now - timedelta(seconds=(372 - i - 1) * 5)
            self.evaluator.metric_history.history.append((sample_time, recovery_metrics.copy()))

        print(f"  Added {len(self.evaluator.metric_history.history)} samples spanning 31 minutes")

        # Evaluate from CAUTION state
        evaluation = self.evaluator.evaluate_transition(
            current_state=RiskGovernorConfig.STATE_CAUTION,
            metrics=recovery_metrics,
            entered_at=(now - timedelta(minutes=31)).isoformat()
        )

        print(f"\nTransition evaluation:")
        print(f"  new_state: {evaluation['new_state']}")
        print(f"  reason: {evaluation['reason']}")

        # Assertions per v3.0.0
        self.assert_equal(
            evaluation['new_state'],
            RiskGovernorConfig.STATE_NORMAL,
            "State transitioned to NORMAL after 30 min stability"
        )
        self.assert_equal(
            evaluation['freeze_new_entries'],
            False,
            "freeze_new_entries = False"
        )
        self.assert_equal(
            evaluation['tighten_exits'],
            False,
            "tighten_exits = False"
        )

        print("\n[PASS] Test 5: CAUTION -> NORMAL recovery after 30 min")

    # ========================================================================
    # TEST 6: CAUTION -> NORMAL blocked if < 30 minutes elapsed
    # ========================================================================

    def test_6_caution_to_normal_blocked_before_30min(self):
        """Test CAUTION -> NORMAL is blocked if < 30 minutes of stability."""
        print("\n" + "=" * 80)
        print("TEST 6: CAUTION -> NORMAL blocked before 30 min (v3.0.0)")
        print("=" * 80)

        # Create fresh metric history
        self.evaluator.metric_history = MetricHistory(max_window_sec=3600)

        # Define recovery metrics (all below recovery thresholds)
        recovery_metrics = {
            'volatility_ratio': 1.4,
            'confidence_collapse_pct': 0.10,
            'liquidity_stress': 0.002,
            'portfolio_drawdown': 0.01
        }

        print("\nSimulating only 15 minutes of recovery metrics:")

        # Add samples spanning only 15 minutes (not enough for recovery)
        now = datetime.utcnow()
        for i in range(180):  # 15 min * 60 sec / 5 sec = 180 samples
            sample_time = now - timedelta(seconds=(180 - i - 1) * 5)
            self.evaluator.metric_history.history.append((sample_time, recovery_metrics.copy()))

        print(f"  Added {len(self.evaluator.metric_history.history)} samples spanning 15 minutes")

        # Evaluate from CAUTION state
        evaluation = self.evaluator.evaluate_transition(
            current_state=RiskGovernorConfig.STATE_CAUTION,
            metrics=recovery_metrics,
            entered_at=(now - timedelta(minutes=15)).isoformat()
        )

        print(f"\nTransition evaluation:")
        print(f"  new_state: {evaluation['new_state']}")
        print(f"  reason: {evaluation['reason']}")

        # Assertions per v3.0.0
        self.assert_equal(
            evaluation['new_state'],
            RiskGovernorConfig.STATE_CAUTION,
            "State remains CAUTION (< 30 min stability)"
        )

        print("\n[PASS] Test 6: CAUTION -> NORMAL blocked before 30 min")

    # ========================================================================
    # TEST 7: CRITICAL -> CAUTION recovery after 60 minutes
    # ========================================================================

    def test_7_critical_to_caution_recovery_60min(self):
        """Test CRITICAL -> CAUTION requires all metrics below CRITICAL for 60 minutes."""
        print("\n" + "=" * 80)
        print("TEST 7: CRITICAL -> CAUTION recovery (60 min sustained, v3.0.0)")
        print("=" * 80)

        # Create fresh metric history
        self.evaluator.metric_history = MetricHistory(max_window_sec=3600)

        # Define recovery metrics (all below CRITICAL thresholds)
        recovery_metrics = {
            'volatility_ratio': 2.3,  # BELOW CRITICAL (2.4)
            'confidence_collapse_pct': 0.30,  # BELOW CRITICAL (0.35)
            'liquidity_stress': 0.006,  # BELOW CRITICAL (0.008)
            'portfolio_drawdown': 0.05  # BELOW CRITICAL (0.06)
        }

        print("\nSimulating 60 minutes of metrics below CRITICAL:")
        for key, value in recovery_metrics.items():
            print(f"  {key}: {value}")

        # Add samples every 5 seconds for 60 minutes
        # Need 721 samples to span 3600 seconds: (721-1) * 5 = 3600
        now = datetime.utcnow()

        # Add 721 samples spanning exactly 60 minutes
        for i in range(721):  # (721-1) * 5 = 3600 seconds
            sample_time = now - timedelta(seconds=(721 - i - 1) * 5)
            self.evaluator.metric_history.history.append((sample_time, recovery_metrics.copy()))

        print(f"  Added {len(self.evaluator.metric_history.history)} samples spanning 60 minutes")

        # Evaluate from CRITICAL state (will add 1 more sample internally)
        evaluation = self.evaluator.evaluate_transition(
            current_state=RiskGovernorConfig.STATE_CRITICAL,
            metrics=recovery_metrics,
            entered_at=(now - timedelta(minutes=60)).isoformat()
        )

        print(f"\nTransition evaluation:")
        print(f"  new_state: {evaluation['new_state']}")
        print(f"  reason: {evaluation['reason']}")

        # Assertions per v3.0.0
        self.assert_equal(
            evaluation['new_state'],
            RiskGovernorConfig.STATE_CAUTION,
            "State transitioned to CAUTION after 60 min stability"
        )
        self.assert_equal(
            evaluation['freeze_new_entries'],
            False,
            "freeze_new_entries = False (CAUTION allows entries)"
        )
        self.assert_equal(
            evaluation['atr_multiplier'],
            0.8,
            "atr_multiplier = 0.8 (CAUTION)"
        )

        print("\n[PASS] Test 7: CRITICAL -> CAUTION recovery after 60 min")

    # ========================================================================
    # TEST 8: CRITICAL -> CAUTION blocked if < 60 minutes elapsed
    # ========================================================================

    def test_8_critical_to_caution_blocked_before_60min(self):
        """Test CRITICAL -> CAUTION is blocked if < 60 minutes of stability."""
        print("\n" + "=" * 80)
        print("TEST 8: CRITICAL -> CAUTION blocked before 60 min (v3.0.0)")
        print("=" * 80)

        # Create fresh metric history
        self.evaluator.metric_history = MetricHistory(max_window_sec=3600)

        # Define recovery metrics (all below CRITICAL thresholds)
        recovery_metrics = {
            'volatility_ratio': 2.3,
            'confidence_collapse_pct': 0.30,
            'liquidity_stress': 0.006,
            'portfolio_drawdown': 0.05
        }

        print("\nSimulating only 30 minutes of metrics below CRITICAL:")

        # Add samples spanning only 30 minutes (not enough for recovery)
        now = datetime.utcnow()
        for i in range(360):  # 30 min * 60 sec / 5 sec = 360 samples
            sample_time = now - timedelta(seconds=(360 - i - 1) * 5)
            self.evaluator.metric_history.history.append((sample_time, recovery_metrics.copy()))

        print(f"  Added {len(self.evaluator.metric_history.history)} samples spanning 30 minutes")

        # Evaluate from CRITICAL state
        evaluation = self.evaluator.evaluate_transition(
            current_state=RiskGovernorConfig.STATE_CRITICAL,
            metrics=recovery_metrics,
            entered_at=(now - timedelta(minutes=30)).isoformat()
        )

        print(f"\nTransition evaluation:")
        print(f"  new_state: {evaluation['new_state']}")
        print(f"  reason: {evaluation['reason']}")

        # Assertions per v3.0.0
        self.assert_equal(
            evaluation['new_state'],
            RiskGovernorConfig.STATE_CRITICAL,
            "State remains CRITICAL (< 60 min stability)"
        )
        self.assert_equal(
            evaluation['freeze_new_entries'],
            True,
            "freeze_new_entries = True (CRITICAL)"
        )
        self.assert_equal(
            evaluation['atr_multiplier'],
            0.5,
            "atr_multiplier = 0.5 (CRITICAL)"
        )

        print("\n[PASS] Test 8: CRITICAL -> CAUTION blocked before 60 min")

    # ========================================================================
    # TEST 9: MetricHistory time window boundary (10 min)
    # ========================================================================

    def test_9_metric_history_10min_window(self):
        """Test MetricHistory correctly evaluates 10-minute time window."""
        print("\n" + "=" * 80)
        print("TEST 9: MetricHistory 10-minute time window (v3.0.0)")
        print("=" * 80)

        # Create fresh metric history
        history = MetricHistory(max_window_sec=3600)

        now = datetime.utcnow()

        # Add old sample (15 minutes ago) with elevated metrics
        old_metrics = {
            'volatility_ratio': 2.0,
            'confidence_collapse_pct': 0.25,
            'liquidity_stress': 0.005,
            'portfolio_drawdown': 0.04
        }
        history.history.append((now - timedelta(minutes=15), old_metrics))

        # Add recent sample (5 minutes ago) with normal metrics
        recent_metrics = {
            'volatility_ratio': 1.5,
            'confidence_collapse_pct': 0.10,
            'liquidity_stress': 0.002,
            'portfolio_drawdown': 0.01
        }
        history.history.append((now - timedelta(minutes=5), recent_metrics))

        print("\nAdded samples:")
        print(f"  15 min ago: 2 metrics elevated")
        print(f"  5 min ago: all metrics normal")

        # Test 10-minute window (should only see recent sample)
        thresholds = {
            'volatility_ratio': 1.8,
            'confidence_collapse_pct': 0.20,
            'liquidity_stress': 0.004,
            'portfolio_drawdown': 0.03
        }

        count = history.count_metrics_above_threshold(thresholds, window_sec=600)

        print(f"\ncount_metrics_above_threshold(window=10min): {count}")

        self.assert_equal(
            count,
            0,
            "10-min window ignores 15-min-old sample"
        )

        print("\n[PASS] Test 9: MetricHistory 10-minute time window")

    # ========================================================================
    # TEST 10: MetricHistory recovery validation (all samples below)
    # ========================================================================

    def test_10_metric_history_all_below_validation(self):
        """Test MetricHistory validates ALL samples below threshold."""
        print("\n" + "=" * 80)
        print("TEST 10: MetricHistory all-below validation (v3.0.0)")
        print("=" * 80)

        # Create fresh metric history
        history = MetricHistory(max_window_sec=3600)

        now = datetime.utcnow()

        # Add 100 samples with all metrics below threshold
        good_metrics = {
            'volatility_ratio': 1.4,
            'confidence_collapse_pct': 0.10,
            'liquidity_stress': 0.002,
            'portfolio_drawdown': 0.01
        }

        for i in range(100):
            sample_time = now - timedelta(seconds=(100 - i) * 5)
            history.history.append((sample_time, good_metrics.copy()))

        # Add 1 sample with elevated metric (should fail all-below check)
        bad_metrics = good_metrics.copy()
        bad_metrics['volatility_ratio'] = 1.6  # Above threshold
        history.history.append((now - timedelta(seconds=50 * 5), bad_metrics))

        # Add 50 more good samples
        for i in range(50):
            sample_time = now - timedelta(seconds=(50 - i) * 5)
            history.history.append((sample_time, good_metrics.copy()))

        print("\nAdded samples:")
        print(f"  150 samples total spanning ~12.5 minutes")
        print(f"  1 sample (at t=-4:10) has volatility_ratio = 1.6 (above 1.5)")

        thresholds = {
            'volatility_ratio': 1.5,
            'confidence_collapse_pct': 0.15,
            'liquidity_stress': 0.003,
            'portfolio_drawdown': 0.02
        }

        result = history.all_metrics_below_threshold(thresholds, duration_sec=600)

        print(f"\nall_metrics_below_threshold(duration=10min): {result}")

        self.assert_equal(
            result,
            False,
            "all-below fails when ANY sample has elevated metric"
        )

        print("\n[PASS] Test 10: MetricHistory all-below validation")

    # ========================================================================
    # Run all tests
    # ========================================================================

    def run_all_tests(self):
        """Execute all v3.0.0 compliance tests."""
        print("\n")
        print("=" * 80)
        print("RISK GOVERNOR v3.0.0 COMPLIANCE TEST SUITE")
        print("=" * 80)
        print("\nTesting against:")
        print("  backend/turbomode/real_time_risk_governor_v3.json")
        print("  backend/turbomode/core_engine/risk_governor_daemon.py")
        print()

        tests = [
            self.test_1_normal_to_caution_with_2_metrics,
            self.test_2_normal_stays_normal_with_1_metric,
            self.test_3_caution_to_critical_with_2_metrics,
            self.test_4_caution_stays_caution_with_1_critical,
            self.test_5_caution_to_normal_recovery_30min,
            self.test_6_caution_to_normal_blocked_before_30min,
            self.test_7_critical_to_caution_recovery_60min,
            self.test_8_critical_to_caution_blocked_before_60min,
            self.test_9_metric_history_10min_window,
            self.test_10_metric_history_all_below_validation
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"\n[FAIL] {test.__name__}: {e}")
                import traceback
                traceback.print_exc()
                self.failed += 1

        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"  Total assertions: {self.passed + self.failed}")
        print(f"  Passed: {self.passed}")
        print(f"  Failed: {self.failed}")

        if self.failed == 0:
            print("\n[OK] ALL TESTS PASSED - v3.0.0 COMPLIANCE VERIFIED")
            print("=" * 80)
            print()
            return 0
        else:
            print(f"\n[X] {self.failed} TESTS FAILED - v3.0.0 COMPLIANCE NOT MET")
            print("=" * 80)
            print()
            return 1


def main():
    """Run v3.0.0 compliance test suite."""
    suite = V3ComplianceTestSuite()
    return suite.run_all_tests()


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
