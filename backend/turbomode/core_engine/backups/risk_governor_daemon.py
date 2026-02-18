#!/usr/bin/env python
"""
Real-Time Risk Governor Daemon v3.0.0

Implements the state machine and monitoring logic exactly as specified
in real_time_risk_governor.json v3.0.0.

This daemon:
- Runs every 5 seconds during market hours
- Computes derived metrics from real-time inputs
- Evaluates state machine transitions
- Writes state changes to database
- Emits WebSocket events

Author: TurboMode Production Team
Date: 2026-01-25
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback


# ============================================================================
# CONFIGURATION
# ============================================================================

class RiskGovernorConfig:
    """Configuration constants from JSON specification."""

    # Runtime config
    LOOP_INTERVAL_MS = 5000
    LOOP_INTERVAL_SEC = LOOP_INTERVAL_MS / 1000.0

    # Database path
    DB_PATH = project_root / "backend" / "data" / "turbomode.db"

    # State machine states
    STATE_NORMAL = "NORMAL"
    STATE_CAUTION = "CAUTION"
    STATE_CRITICAL = "CRITICAL"

    # Volatility thresholds (v3.0.0)
    CAUTION_VOLATILITY_RATIO_MIN = 1.8
    CRITICAL_VOLATILITY_RATIO_MIN = 2.4  # v3.0.0: 2.4 (not 2.5)

    # Confidence thresholds (v3.0.0)
    CAUTION_CONFIDENCE_COLLAPSE_MIN = 0.20  # v3.0.0: 0.20 (not 0.25)
    CRITICAL_CONFIDENCE_COLLAPSE_MIN = 0.35  # v3.0.0: 0.35 (not 0.40)

    # Liquidity thresholds (v3.0.0)
    CAUTION_LIQUIDITY_STRESS_MIN = 0.004  # v3.0.0: 0.004 (not 0.015)
    CRITICAL_LIQUIDITY_STRESS_MIN = 0.008  # v3.0.0: 0.008 (not 0.030)

    # Drawdown thresholds (v3.0.0)
    CAUTION_PORTFOLIO_DRAWDOWN_MIN = 0.03
    CRITICAL_PORTFOLIO_DRAWDOWN_MIN = 0.06  # v3.0.0: 0.06 (not 0.05)

    # Recovery thresholds (v3.0.0)
    RECOVERY_VOLATILITY_RATIO_MAX = 1.5
    RECOVERY_CONFIDENCE_COLLAPSE_MAX = 0.15
    RECOVERY_LIQUIDITY_STRESS_MAX = 0.003  # v3.0.0: 0.003 (not 0.010)
    RECOVERY_PORTFOLIO_DRAWDOWN_MAX = 0.02

    # Exit tightening config (v3.0.0)
    CAUTION_ATR_MULTIPLIER = 0.8  # v3.0.0: 0.8 for CAUTION
    CRITICAL_ATR_MULTIPLIER = 0.5  # v3.0.0: 0.5 for CRITICAL

    # Deleveraging config (v3.0.0)
    DELEVERAGING_NOTIONAL_REDUCTION_PCT = 0.30

    # Confidence penalty config (v3.0.0)
    CONFIDENCE_PENALTY_FACTOR = 0.60  # v3.0.0: 0.60 (not 0.50)

    # Transition time windows (v3.0.0)
    NORMAL_TO_CAUTION_TIME_WINDOW_SEC = 600  # v3.0.0: 10 minutes
    CAUTION_TO_CRITICAL_TIME_WINDOW_SEC = 300  # v3.0.0: 5 minutes

    # Recovery timing (v3.0.0)
    CRITICAL_TO_CAUTION_STABLE_DURATION_SEC = 3600  # v3.0.0: 60 minutes (not 5)
    CAUTION_TO_NORMAL_STABLE_DURATION_SEC = 1800  # v3.0.0: 30 minutes (not 10)


# ============================================================================
# DATA ACCESS LAYER
# ============================================================================

class RiskGovernorDB:
    """Database access layer for Risk Governor."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(str(self.db_path))

    def get_current_state(self) -> Dict[str, Any]:
        """
        Read current risk state from global_risk_state table.

        Returns:
            Dict with current state fields
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                current_state,
                previous_state,
                freeze_new_entries,
                tighten_exits,
                force_deleveraging,
                confidence_penalty_active,
                confidence_penalty_factor,
                reason,
                details,
                entered_at,
                updated_at
            FROM global_risk_state
            WHERE id = 1
        """)

        row = cursor.fetchone()
        conn.close()

        if not row:
            # Should never happen (initialized in migration)
            return {
                'current_state': RiskGovernorConfig.STATE_NORMAL,
                'previous_state': None,
                'freeze_new_entries': False,
                'tighten_exits': False,
                'force_deleveraging': False,
                'confidence_penalty_active': False,
                'confidence_penalty_factor': None,
                'reason': 'Uninitialized',
                'details': {},
                'entered_at': None,
                'updated_at': None
            }

        return {
            'current_state': row[0],
            'previous_state': row[1],
            'freeze_new_entries': bool(row[2]),
            'tighten_exits': bool(row[3]),
            'force_deleveraging': bool(row[4]),
            'confidence_penalty_active': bool(row[5]),
            'confidence_penalty_factor': row[6],
            'reason': row[7],
            'details': json.loads(row[8]) if row[8] else {},
            'entered_at': row[9],
            'updated_at': row[10]
        }

    def update_state(self, new_state: str, reason: str, details: Dict[str, Any],
                     freeze_new_entries: bool, tighten_exits: bool, force_deleveraging: bool,
                     confidence_penalty_active: bool, confidence_penalty_factor: Optional[float]):
        """
        Update global risk state (state transition).

        Args:
            new_state: New state (NORMAL/CAUTION/CRITICAL)
            reason: Human-readable reason for transition
            details: JSON-serializable dict with computed metrics
            freeze_new_entries: Block new position entries
            tighten_exits: Tighten stop losses
            force_deleveraging: Force position reduction
            confidence_penalty_active: Enable confidence penalty
            confidence_penalty_factor: Confidence penalty multiplier
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Get current state for transition tracking
        current = self.get_current_state()
        previous_state = current['current_state']

        now = datetime.utcnow().isoformat()

        # Determine if state changed (to update entered_at)
        state_changed = (previous_state != new_state)
        entered_at = now if state_changed else current['entered_at']

        cursor.execute("""
            UPDATE global_risk_state
            SET
                current_state = ?,
                previous_state = ?,
                freeze_new_entries = ?,
                tighten_exits = ?,
                force_deleveraging = ?,
                confidence_penalty_active = ?,
                confidence_penalty_factor = ?,
                reason = ?,
                details = ?,
                entered_at = ?,
                updated_at = ?
            WHERE id = 1
        """, (
            new_state,
            previous_state,
            int(freeze_new_entries),
            int(tighten_exits),
            int(force_deleveraging),
            int(confidence_penalty_active),
            confidence_penalty_factor,
            reason,
            json.dumps(details),
            entered_at,
            now
        ))

        conn.commit()
        conn.close()

        # If state changed, log risk event
        if state_changed:
            self.log_risk_event(
                event_type='state_transition',
                payload={
                    'old_state': previous_state,
                    'new_state': new_state,
                    'reason': reason,
                    'timestamp': now
                }
            )

    def log_risk_event(self, event_type: str, payload: Dict[str, Any]):
        """
        Append risk event to audit log.

        Args:
            event_type: Event type (e.g., "state_transition", "symbol_flag_raised")
            payload: JSON-serializable event payload
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO risk_events (event_type, payload, created_at)
            VALUES (?, ?, ?)
        """, (
            event_type,
            json.dumps(payload),
            datetime.utcnow().isoformat()
        ))

        conn.commit()
        conn.close()

    def raise_symbol_flag(self, symbol: str, risk_level: str, reason: str, details: Dict[str, Any]):
        """
        Raise a risk flag for a symbol.

        Args:
            symbol: Stock symbol
            risk_level: Risk level (NORMAL/CAUTION/CRITICAL)
            reason: Flag reason (e.g., "volatility_spike")
            details: JSON-serializable details
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()

        cursor.execute("""
            INSERT INTO symbol_risk_flags (symbol, risk_level, reason, details, flagged_at, cleared_at)
            VALUES (?, ?, ?, ?, ?, NULL)
        """, (
            symbol,
            risk_level,
            reason,
            json.dumps(details),
            now
        ))

        conn.commit()
        conn.close()

        # Log event
        self.log_risk_event(
            event_type='symbol_flag_raised',
            payload={
                'symbol': symbol,
                'risk_level': risk_level,
                'reason': reason,
                'timestamp': now
            }
        )

    def clear_symbol_flags(self, symbol: str):
        """
        Clear all active flags for a symbol.

        Args:
            symbol: Stock symbol
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()

        cursor.execute("""
            UPDATE symbol_risk_flags
            SET cleared_at = ?
            WHERE symbol = ? AND cleared_at IS NULL
        """, (now, symbol))

        conn.commit()
        conn.close()


# ============================================================================
# INPUT ADAPTERS (Placeholders - to be integrated with real data sources)
# ============================================================================

class InputAdapters:
    """
    Input adapters for reading real-time data.

    NOTE: These are placeholder implementations. In production, these should
    be replaced with actual integrations to IBKR API, micro scanner, etc.
    """

    @staticmethod
    def get_realtime_data() -> Dict[str, Any]:
        """
        Get real-time market data from IBKR.

        Returns:
            Dict with real-time price, bid, ask, volume for active positions
        """
        # PLACEHOLDER: Return mock data
        # TODO: Integrate with IBKR TWS API
        return {
            'symbols': {},  # Empty for now
            'timestamp': datetime.utcnow().isoformat()
        }

    @staticmethod
    def get_micro_scanner_data() -> Dict[str, Any]:
        """
        Get micro scanner predictions and confidence scores.

        Returns:
            Dict with signal strength, confidence, model agreement
        """
        # PLACEHOLDER: Return mock data
        # TODO: Integrate with overnight_scanner.py or real-time scanner
        return {
            'signals': {},  # Empty for now
            'timestamp': datetime.utcnow().isoformat()
        }

    @staticmethod
    def get_news_engine_data() -> List[Dict[str, Any]]:
        """
        Get recent news events with sentiment scores.

        Returns:
            List of news items with sentiment, relevance, symbols
        """
        # PLACEHOLDER: Return mock data
        # TODO: Integrate with news_feed_client.py
        return []

    @staticmethod
    def get_portfolio_state() -> Dict[str, Any]:
        """
        Get current portfolio state (equity, cash, positions, P&L).

        Returns:
            Dict with portfolio metrics
        """
        # PLACEHOLDER: Return mock data
        # TODO: Integrate with IBKR account data or position_manager.py
        return {
            'total_equity': 100000.0,
            'cash_balance': 50000.0,
            'buying_power': 50000.0,
            'positions_count': 0,
            'total_unrealized_pnl': 0.0,
            'total_realized_pnl': 0.0,
            'day_pnl': 0.0,
            'timestamp': datetime.utcnow().isoformat()
        }


# ============================================================================
# DERIVED METRICS ENGINE
# ============================================================================

class DerivedMetricsEngine:
    """
    Computes derived risk metrics from raw inputs.

    Implements formulas exactly as specified in JSON v3.0.0.
    """

    @staticmethod
    def compute_volatility_ratio(realtime_data: Dict[str, Any]) -> float:
        """
        Compute volatility_ratio = realized_vol_1min / baseline_vol_20d.

        Args:
            realtime_data: Real-time market data

        Returns:
            Volatility ratio (float)
        """
        # PLACEHOLDER: Return 1.0 (normal volatility)
        # TODO: Implement actual volatility calculation
        return 1.0

    @staticmethod
    def compute_confidence_collapse_pct(micro_scanner_data: Dict[str, Any]) -> float:
        """
        Compute percentage of positions with model confidence < 0.55.

        Args:
            micro_scanner_data: Micro scanner predictions

        Returns:
            Confidence collapse percentage (0.0 to 1.0)
        """
        # PLACEHOLDER: Return 0.0 (no collapse)
        # TODO: Implement actual confidence analysis
        return 0.0

    @staticmethod
    def compute_liquidity_stress(realtime_data: Dict[str, Any]) -> float:
        """
        Compute maximum bid-ask spread across all positions.

        Args:
            realtime_data: Real-time market data

        Returns:
            Max spread as fraction of mid-price (float)
        """
        # PLACEHOLDER: Return 0.005 (0.5% spread)
        # TODO: Implement actual spread calculation
        return 0.005

    @staticmethod
    def compute_portfolio_drawdown(portfolio_state: Dict[str, Any]) -> float:
        """
        Compute portfolio drawdown as (peak_equity - current_equity) / peak_equity.

        Args:
            portfolio_state: Portfolio state data

        Returns:
            Drawdown percentage (0.0 to 1.0)
        """
        # PLACEHOLDER: Return 0.0 (no drawdown)
        # TODO: Track peak equity and compute actual drawdown
        return 0.0

    @staticmethod
    def compute_all_metrics(realtime_data: Dict[str, Any],
                           micro_scanner_data: Dict[str, Any],
                           news_data: List[Dict[str, Any]],
                           portfolio_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute all derived metrics.

        Args:
            realtime_data: Real-time market data
            micro_scanner_data: Micro scanner data
            news_data: News events
            portfolio_state: Portfolio state

        Returns:
            Dict with all computed metrics
        """
        return {
            'volatility_ratio': DerivedMetricsEngine.compute_volatility_ratio(realtime_data),
            'confidence_collapse_pct': DerivedMetricsEngine.compute_confidence_collapse_pct(micro_scanner_data),
            'liquidity_stress': DerivedMetricsEngine.compute_liquidity_stress(realtime_data),
            'portfolio_drawdown': DerivedMetricsEngine.compute_portfolio_drawdown(portfolio_state)
        }


# ============================================================================
# STATE MACHINE EVALUATOR
# ============================================================================

class MetricHistory:
    """
    Tracks metric values over time for time-window evaluation.

    Per v3.0.0: Transition logic requires "at least N of M metrics"
    within specific time windows.
    """

    def __init__(self, max_window_sec: int = 3600):
        """
        Args:
            max_window_sec: Maximum time window to track (default 3600s = 60 min)
        """
        self.max_window_sec = max_window_sec
        self.history = []  # List of (timestamp, metrics_dict)

    def add_sample(self, metrics: Dict[str, float]):
        """Add a new metrics sample with current timestamp."""
        now = datetime.utcnow()
        self.history.append((now, metrics.copy()))
        self._prune_old_samples()

    def _prune_old_samples(self):
        """Remove samples older than max_window_sec."""
        if not self.history:
            return

        cutoff = datetime.utcnow() - timedelta(seconds=self.max_window_sec)
        self.history = [(ts, m) for ts, m in self.history if ts >= cutoff]

    def count_metrics_above_threshold(self, thresholds: Dict[str, float],
                                     window_sec: int) -> int:
        """
        Count how many metrics are above their thresholds within the window.

        Per v3.0.0: "at_least_2_of_4_within_time_window"

        Args:
            thresholds: Dict of metric_name -> threshold_value
            window_sec: Time window in seconds

        Returns:
            Count of metrics currently above threshold (0-4)
        """
        if not self.history:
            return 0

        cutoff = datetime.utcnow() - timedelta(seconds=window_sec)

        # Get samples within window
        recent_samples = [(ts, m) for ts, m in self.history if ts >= cutoff]

        if not recent_samples:
            return 0

        # Use most recent sample's metrics
        _, latest_metrics = recent_samples[-1]

        # Count how many metrics exceed thresholds
        count = 0
        for metric_name, threshold in thresholds.items():
            if metric_name in latest_metrics:
                if latest_metrics[metric_name] >= threshold:
                    count += 1

        return count

    def all_metrics_below_threshold(self, thresholds: Dict[str, float],
                                   duration_sec: int) -> bool:
        """
        Check if ALL metrics have been below thresholds for entire duration.

        Per v3.0.0: Recovery logic requires "all metrics below" for duration.

        Args:
            thresholds: Dict of metric_name -> threshold_value
            duration_sec: Required duration in seconds

        Returns:
            True if all metrics below thresholds for entire duration
        """
        if not self.history:
            return False

        cutoff = datetime.utcnow() - timedelta(seconds=duration_sec)

        # Get all samples within duration
        duration_samples = [(ts, m) for ts, m in self.history if ts >= cutoff]

        # Need samples spanning the entire duration
        if not duration_samples:
            return False

        earliest_sample_time = duration_samples[0][0]
        time_span = (datetime.utcnow() - earliest_sample_time).total_seconds()

        # If we don't have samples spanning the full duration, return False
        if time_span < duration_sec:
            return False

        # Check that all metrics were below thresholds in ALL samples
        for ts, metrics in duration_samples:
            for metric_name, threshold in thresholds.items():
                if metric_name in metrics:
                    if metrics[metric_name] >= threshold:
                        return False  # Found a metric above threshold

        return True


class StateMachineEvaluator:
    """
    Evaluates state transitions based on derived metrics.

    Implements transition logic exactly as specified in JSON v3.0.0.
    """

    def __init__(self, db: RiskGovernorDB):
        self.db = db
        self.metric_history = MetricHistory(max_window_sec=3600)  # Track last 60 min

    def evaluate_transition(self, current_state: str, metrics: Dict[str, float],
                           entered_at: Optional[str]) -> Dict[str, Any]:
        """
        Evaluate whether a state transition should occur.

        Args:
            current_state: Current state (NORMAL/CAUTION/CRITICAL)
            metrics: Computed derived metrics
            entered_at: ISO timestamp when current state was entered

        Returns:
            Dict with:
                - new_state: New state (same as current if no transition)
                - reason: Reason for state (or transition)
                - freeze_new_entries: Block new entries
                - tighten_exits: Tighten stops
                - force_deleveraging: Force position reduction
                - confidence_penalty_active: Enable confidence penalty
                - confidence_penalty_factor: Confidence penalty multiplier
        """

        # Add current metrics to history for time-window evaluation
        self.metric_history.add_sample(metrics)

        if current_state == RiskGovernorConfig.STATE_NORMAL:
            return self._evaluate_normal(metrics)

        elif current_state == RiskGovernorConfig.STATE_CAUTION:
            return self._evaluate_caution(metrics, entered_at)

        elif current_state == RiskGovernorConfig.STATE_CRITICAL:
            return self._evaluate_critical(metrics, entered_at)

        else:
            # Unknown state - default to NORMAL
            return {
                'new_state': RiskGovernorConfig.STATE_NORMAL,
                'reason': 'Unknown state - reset to NORMAL',
                'freeze_new_entries': False,
                'tighten_exits': False,
                'force_deleveraging': False,
                'confidence_penalty_active': False,
                'confidence_penalty_factor': None
            }

    def _evaluate_normal(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluate transitions from NORMAL state.

        Per v3.0.0: NORMAL -> CAUTION requires "at least 2 of 4 metrics"
        at or above CAUTION thresholds within last 10 minutes (600 seconds).
        """

        # Define CAUTION thresholds per v3.0.0
        caution_thresholds = {
            'volatility_ratio': RiskGovernorConfig.CAUTION_VOLATILITY_RATIO_MIN,
            'confidence_collapse_pct': RiskGovernorConfig.CAUTION_CONFIDENCE_COLLAPSE_MIN,
            'liquidity_stress': RiskGovernorConfig.CAUTION_LIQUIDITY_STRESS_MIN,
            'portfolio_drawdown': RiskGovernorConfig.CAUTION_PORTFOLIO_DRAWDOWN_MIN
        }

        # Count how many metrics are above threshold within time window
        metrics_above_threshold = self.metric_history.count_metrics_above_threshold(
            thresholds=caution_thresholds,
            window_sec=RiskGovernorConfig.NORMAL_TO_CAUTION_TIME_WINDOW_SEC  # 600s = 10 min
        )

        # v3.0.0: Requires at least 2 of 4 metrics >= threshold
        if metrics_above_threshold >= 2:
            # Build reason string
            reasons = []
            for metric_name, threshold in caution_thresholds.items():
                if metric_name in metrics and metrics[metric_name] >= threshold:
                    reasons.append(f"{metric_name}={metrics[metric_name]:.3f}")

            return {
                'new_state': RiskGovernorConfig.STATE_CAUTION,
                'reason': f"NORMAL -> CAUTION: {metrics_above_threshold} of 4 metrics elevated ({', '.join(reasons)})",
                'freeze_new_entries': False,  # v3.0.0: CAUTION allows new entries
                'tighten_exits': True,
                'atr_multiplier': RiskGovernorConfig.CAUTION_ATR_MULTIPLIER,  # v3.0.0: 0.8
                'force_deleveraging': False,
                'confidence_penalty_active': False,
                'confidence_penalty_factor': None
            }

        # Stay in NORMAL
        return {
            'new_state': RiskGovernorConfig.STATE_NORMAL,
            'reason': f'Normal market conditions ({metrics_above_threshold} of 4 metrics elevated)',
            'freeze_new_entries': False,
            'tighten_exits': False,
            'force_deleveraging': False,
            'confidence_penalty_active': False,
            'confidence_penalty_factor': None
        }

    def _evaluate_caution(self, metrics: Dict[str, float], entered_at: Optional[str]) -> Dict[str, Any]:
        """
        Evaluate transitions from CAUTION state.

        Per v3.0.0:
        - CAUTION -> CRITICAL requires "at least 2 of 4 metrics" >= CRITICAL thresholds within last 5 minutes
        - CAUTION -> NORMAL requires "all metrics below CAUTION thresholds" for 30 minutes
        """

        # Define CRITICAL thresholds per v3.0.0
        critical_thresholds = {
            'volatility_ratio': RiskGovernorConfig.CRITICAL_VOLATILITY_RATIO_MIN,
            'confidence_collapse_pct': RiskGovernorConfig.CRITICAL_CONFIDENCE_COLLAPSE_MIN,
            'liquidity_stress': RiskGovernorConfig.CRITICAL_LIQUIDITY_STRESS_MIN,
            'portfolio_drawdown': RiskGovernorConfig.CRITICAL_PORTFOLIO_DRAWDOWN_MIN
        }

        # Check CAUTION -> CRITICAL: at least 2 of 4 metrics >= CRITICAL thresholds within 5 min
        metrics_above_critical = self.metric_history.count_metrics_above_threshold(
            thresholds=critical_thresholds,
            window_sec=RiskGovernorConfig.CAUTION_TO_CRITICAL_TIME_WINDOW_SEC  # 300s = 5 min
        )

        if metrics_above_critical >= 2:
            # Build reason string
            reasons = []
            for metric_name, threshold in critical_thresholds.items():
                if metric_name in metrics and metrics[metric_name] >= threshold:
                    reasons.append(f"{metric_name}={metrics[metric_name]:.3f}")

            return {
                'new_state': RiskGovernorConfig.STATE_CRITICAL,
                'reason': f"CAUTION -> CRITICAL: {metrics_above_critical} of 4 metrics critical ({', '.join(reasons)})",
                'freeze_new_entries': True,
                'tighten_exits': True,
                'atr_multiplier': RiskGovernorConfig.CRITICAL_ATR_MULTIPLIER,  # v3.0.0: 0.5
                'force_deleveraging': True,
                'confidence_penalty_active': True,
                'confidence_penalty_factor': RiskGovernorConfig.CONFIDENCE_PENALTY_FACTOR  # v3.0.0: 0.60
            }

        # Define recovery thresholds per v3.0.0
        recovery_thresholds = {
            'volatility_ratio': RiskGovernorConfig.RECOVERY_VOLATILITY_RATIO_MAX,
            'confidence_collapse_pct': RiskGovernorConfig.RECOVERY_CONFIDENCE_COLLAPSE_MAX,
            'liquidity_stress': RiskGovernorConfig.RECOVERY_LIQUIDITY_STRESS_MAX,
            'portfolio_drawdown': RiskGovernorConfig.RECOVERY_PORTFOLIO_DRAWDOWN_MAX
        }

        # Check CAUTION -> NORMAL: all metrics below recovery thresholds for 30 min
        if self.metric_history.all_metrics_below_threshold(
            thresholds=recovery_thresholds,
            duration_sec=RiskGovernorConfig.CAUTION_TO_NORMAL_STABLE_DURATION_SEC  # 1800s = 30 min
        ):
            return {
                'new_state': RiskGovernorConfig.STATE_NORMAL,
                'reason': "CAUTION -> NORMAL: All metrics stable for 30 min",
                'freeze_new_entries': False,
                'tighten_exits': False,
                'force_deleveraging': False,
                'confidence_penalty_active': False,
                'confidence_penalty_factor': None
            }

        # Stay in CAUTION
        return {
            'new_state': RiskGovernorConfig.STATE_CAUTION,
            'reason': f'Elevated risk - monitoring ({metrics_above_critical} of 4 at CRITICAL level)',
            'freeze_new_entries': False,  # v3.0.0: CAUTION allows new entries
            'tighten_exits': True,
            'atr_multiplier': RiskGovernorConfig.CAUTION_ATR_MULTIPLIER,  # v3.0.0: 0.8
            'force_deleveraging': False,
            'confidence_penalty_active': False,
            'confidence_penalty_factor': None
        }

    def _evaluate_critical(self, metrics: Dict[str, float], entered_at: Optional[str]) -> Dict[str, Any]:
        """
        Evaluate transitions from CRITICAL state.

        Per v3.0.0: CRITICAL -> CAUTION requires "all metrics below CRITICAL thresholds" for 60 minutes
        """

        # Define CRITICAL thresholds per v3.0.0
        critical_thresholds = {
            'volatility_ratio': RiskGovernorConfig.CRITICAL_VOLATILITY_RATIO_MIN,
            'confidence_collapse_pct': RiskGovernorConfig.CRITICAL_CONFIDENCE_COLLAPSE_MIN,
            'liquidity_stress': RiskGovernorConfig.CRITICAL_LIQUIDITY_STRESS_MIN,
            'portfolio_drawdown': RiskGovernorConfig.CRITICAL_PORTFOLIO_DRAWDOWN_MIN
        }

        # Check CRITICAL -> CAUTION: all metrics below CRITICAL thresholds for 60 min
        if self.metric_history.all_metrics_below_threshold(
            thresholds=critical_thresholds,
            duration_sec=RiskGovernorConfig.CRITICAL_TO_CAUTION_STABLE_DURATION_SEC  # 3600s = 60 min
        ):
            return {
                'new_state': RiskGovernorConfig.STATE_CAUTION,
                'reason': "CRITICAL -> CAUTION: All metrics below CRITICAL for 60 min",
                'freeze_new_entries': False,  # v3.0.0: CAUTION allows new entries
                'tighten_exits': True,
                'atr_multiplier': RiskGovernorConfig.CAUTION_ATR_MULTIPLIER,  # v3.0.0: 0.8
                'force_deleveraging': False,
                'confidence_penalty_active': False,
                'confidence_penalty_factor': None
            }

        # Stay in CRITICAL
        return {
            'new_state': RiskGovernorConfig.STATE_CRITICAL,
            'reason': 'Critical risk level - emergency mode',
            'freeze_new_entries': True,
            'tighten_exits': True,
            'atr_multiplier': RiskGovernorConfig.CRITICAL_ATR_MULTIPLIER,  # v3.0.0: 0.5
            'force_deleveraging': True,
            'confidence_penalty_active': True,
            'confidence_penalty_factor': RiskGovernorConfig.CONFIDENCE_PENALTY_FACTOR  # v3.0.0: 0.60
        }


# ============================================================================
# WEBSOCKET EVENT EMITTER (Placeholder)
# ============================================================================

class WebSocketEmitter:
    """
    WebSocket event emitter for real-time risk updates.

    NOTE: This is a placeholder. In production, integrate with actual
    WebSocket server (e.g., api_server.py).
    """

    @staticmethod
    def emit_state_change(old_state: str, new_state: str, reason: str, metrics: Dict[str, float]):
        """Emit risk_state_change event."""
        # PLACEHOLDER: Print to console
        print(f"[WS] risk_state_change: {old_state} -> {new_state} ({reason})")
        # TODO: Integrate with WebSocket server

    @staticmethod
    def emit_portfolio_alert(alert_type: str, metric: str, current_value: float, threshold: float):
        """Emit portfolio_alert event."""
        # PLACEHOLDER: Print to console
        print(f"[WS] portfolio_alert: {alert_type} - {metric}={current_value:.4f} (threshold={threshold:.4f})")
        # TODO: Integrate with WebSocket server


# ============================================================================
# MAIN DAEMON LOOP
# ============================================================================

class RiskGovernorDaemon:
    """Main daemon orchestrator."""

    def __init__(self):
        self.db = RiskGovernorDB(RiskGovernorConfig.DB_PATH)
        self.evaluator = StateMachineEvaluator(self.db)
        self.running = False

    def run(self):
        """Start daemon loop."""
        self.running = True

        print("=" * 80)
        print("REAL-TIME RISK GOVERNOR DAEMON v3.0.0")
        print("=" * 80)
        print(f"Database: {RiskGovernorConfig.DB_PATH}")
        print(f"Loop interval: {RiskGovernorConfig.LOOP_INTERVAL_SEC}s")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()

        iteration = 0

        try:
            while self.running:
                iteration += 1
                self._run_iteration(iteration)
                time.sleep(RiskGovernorConfig.LOOP_INTERVAL_SEC)

        except KeyboardInterrupt:
            print()
            print("=" * 80)
            print("DAEMON STOPPED (KeyboardInterrupt)")
            print("=" * 80)

        except Exception as e:
            print()
            print("=" * 80)
            print(f"DAEMON CRASHED: {e}")
            print("=" * 80)
            traceback.print_exc()

    def _run_iteration(self, iteration: int):
        """Execute single iteration of daemon loop."""

        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Iteration {iteration}")

        try:
            # Step 1: Read inputs
            realtime_data = InputAdapters.get_realtime_data()
            micro_scanner_data = InputAdapters.get_micro_scanner_data()
            news_data = InputAdapters.get_news_engine_data()
            portfolio_state = InputAdapters.get_portfolio_state()

            # Step 2: Compute derived metrics
            metrics = DerivedMetricsEngine.compute_all_metrics(
                realtime_data,
                micro_scanner_data,
                news_data,
                portfolio_state
            )

            print(f"  Metrics: vol={metrics['volatility_ratio']:.2f}, "
                  f"conf_collapse={metrics['confidence_collapse_pct']:.1%}, "
                  f"liquidity={metrics['liquidity_stress']:.3f}, "
                  f"drawdown={metrics['portfolio_drawdown']:.2%}")

            # Step 3: Get current state
            current = self.db.get_current_state()
            current_state = current['current_state']
            entered_at = current['entered_at']

            print(f"  Current state: {current_state}")

            # Step 4: Evaluate transition
            evaluation = self.evaluator.evaluate_transition(current_state, metrics, entered_at)
            new_state = evaluation['new_state']

            # Step 5: Update state if changed
            if new_state != current_state:
                print(f"  STATE TRANSITION: {current_state} -> {new_state}")
                print(f"  Reason: {evaluation['reason']}")

                # Update database
                self.db.update_state(
                    new_state=new_state,
                    reason=evaluation['reason'],
                    details=metrics,
                    freeze_new_entries=evaluation['freeze_new_entries'],
                    tighten_exits=evaluation['tighten_exits'],
                    force_deleveraging=evaluation['force_deleveraging'],
                    confidence_penalty_active=evaluation['confidence_penalty_active'],
                    confidence_penalty_factor=evaluation['confidence_penalty_factor']
                )

                # Emit WebSocket event
                WebSocketEmitter.emit_state_change(
                    old_state=current_state,
                    new_state=new_state,
                    reason=evaluation['reason'],
                    metrics=metrics
                )
            else:
                print(f"  State unchanged: {current_state}")

            print()

        except Exception as e:
            print(f"  [ERROR] Iteration failed: {e}")
            print()

    def stop(self):
        """Stop daemon loop."""
        self.running = False


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Entry point for daemon."""
    daemon = RiskGovernorDaemon()
    daemon.run()


if __name__ == '__main__':
    main()
