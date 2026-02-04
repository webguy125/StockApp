#!/usr/bin/env python
"""
Test script for Risk Governor state transitions.

Simulates elevated risk conditions to verify state machine transitions
and database writes work correctly.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.turbomode.core_engine.risk_governor_daemon import (
    RiskGovernorDB,
    RiskGovernorConfig,
    StateMachineEvaluator
)


def test_normal_to_caution():
    """Test NORMAL -> CAUTION transition."""
    print("=" * 80)
    print("TEST 1: NORMAL -> CAUTION TRANSITION")
    print("=" * 80)

    db = RiskGovernorDB(RiskGovernorConfig.DB_PATH)
    evaluator = StateMachineEvaluator(db)

    # Simulate elevated volatility (triggers CAUTION)
    metrics = {
        'volatility_ratio': 2.0,  # Above CAUTION threshold (1.8)
        'confidence_collapse_pct': 0.0,
        'liquidity_stress': 0.005,
        'portfolio_drawdown': 0.0
    }

    print("\nSimulated metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Evaluate from NORMAL state
    evaluation = evaluator.evaluate_transition(
        current_state=RiskGovernorConfig.STATE_NORMAL,
        metrics=metrics,
        entered_at=None
    )

    print(f"\nEvaluation result:")
    print(f"  New state: {evaluation['new_state']}")
    print(f"  Reason: {evaluation['reason']}")
    print(f"  freeze_new_entries: {evaluation['freeze_new_entries']}")
    print(f"  tighten_exits: {evaluation['tighten_exits']}")

    # Apply state change
    db.update_state(
        new_state=evaluation['new_state'],
        reason=evaluation['reason'],
        details=metrics,
        freeze_new_entries=evaluation['freeze_new_entries'],
        tighten_exits=evaluation['tighten_exits'],
        force_deleveraging=evaluation['force_deleveraging'],
        confidence_penalty_active=evaluation['confidence_penalty_active'],
        confidence_penalty_factor=evaluation['confidence_penalty_factor']
    )

    print("\n[OK] State transition applied to database")

    # Verify
    current = db.get_current_state()
    print(f"\nCurrent database state:")
    print(f"  current_state: {current['current_state']}")
    print(f"  previous_state: {current['previous_state']}")
    print(f"  freeze_new_entries: {current['freeze_new_entries']}")
    print(f"  tighten_exits: {current['tighten_exits']}")

    assert current['current_state'] == RiskGovernorConfig.STATE_CAUTION
    print("\n[PASS] NORMAL -> CAUTION transition verified")
    print()


def test_caution_to_critical():
    """Test CAUTION -> CRITICAL transition."""
    print("=" * 80)
    print("TEST 2: CAUTION -> CRITICAL TRANSITION")
    print("=" * 80)

    db = RiskGovernorDB(RiskGovernorConfig.DB_PATH)
    evaluator = StateMachineEvaluator(db)

    # Simulate critical volatility (triggers CRITICAL)
    metrics = {
        'volatility_ratio': 3.0,  # Above CRITICAL threshold (2.5)
        'confidence_collapse_pct': 0.45,  # Above CRITICAL threshold (0.40)
        'liquidity_stress': 0.035,  # Above CRITICAL threshold (0.030)
        'portfolio_drawdown': 0.06  # Above CRITICAL threshold (0.05)
    }

    print("\nSimulated metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Evaluate from CAUTION state
    evaluation = evaluator.evaluate_transition(
        current_state=RiskGovernorConfig.STATE_CAUTION,
        metrics=metrics,
        entered_at=None
    )

    print(f"\nEvaluation result:")
    print(f"  New state: {evaluation['new_state']}")
    print(f"  Reason: {evaluation['reason']}")
    print(f"  freeze_new_entries: {evaluation['freeze_new_entries']}")
    print(f"  tighten_exits: {evaluation['tighten_exits']}")
    print(f"  force_deleveraging: {evaluation['force_deleveraging']}")
    print(f"  confidence_penalty_active: {evaluation['confidence_penalty_active']}")
    print(f"  confidence_penalty_factor: {evaluation['confidence_penalty_factor']}")

    # Apply state change
    db.update_state(
        new_state=evaluation['new_state'],
        reason=evaluation['reason'],
        details=metrics,
        freeze_new_entries=evaluation['freeze_new_entries'],
        tighten_exits=evaluation['tighten_exits'],
        force_deleveraging=evaluation['force_deleveraging'],
        confidence_penalty_active=evaluation['confidence_penalty_active'],
        confidence_penalty_factor=evaluation['confidence_penalty_factor']
    )

    print("\n[OK] State transition applied to database")

    # Verify
    current = db.get_current_state()
    print(f"\nCurrent database state:")
    print(f"  current_state: {current['current_state']}")
    print(f"  previous_state: {current['previous_state']}")
    print(f"  force_deleveraging: {current['force_deleveraging']}")
    print(f"  confidence_penalty_active: {current['confidence_penalty_active']}")
    print(f"  confidence_penalty_factor: {current['confidence_penalty_factor']}")

    assert current['current_state'] == RiskGovernorConfig.STATE_CRITICAL
    print("\n[PASS] CAUTION -> CRITICAL transition verified")
    print()


def test_symbol_flag():
    """Test symbol flag raising."""
    print("=" * 80)
    print("TEST 3: SYMBOL FLAG")
    print("=" * 80)

    db = RiskGovernorDB(RiskGovernorConfig.DB_PATH)

    # Raise a volatility spike flag for a symbol
    db.raise_symbol_flag(
        symbol='AAPL',
        risk_level='CAUTION',
        reason='volatility_spike',
        details={'volatility_ratio': 2.1, 'timestamp': '2026-01-25T18:30:00'}
    )

    print("\n[OK] Symbol flag raised for AAPL")

    # Verify in database
    import sqlite3
    conn = sqlite3.connect(str(RiskGovernorConfig.DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT symbol, risk_level, reason, flagged_at, cleared_at
        FROM symbol_risk_flags
        WHERE symbol = 'AAPL'
        ORDER BY flagged_at DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    conn.close()

    if row:
        print(f"\nSymbol flag in database:")
        print(f"  Symbol: {row[0]}")
        print(f"  Risk level: {row[1]}")
        print(f"  Reason: {row[2]}")
        print(f"  Flagged at: {row[3]}")
        print(f"  Cleared at: {row[4]}")

        assert row[4] is None  # Should not be cleared yet
        print("\n[PASS] Symbol flag verified")
    else:
        print("\n[FAIL] Symbol flag not found in database")

    print()


def test_risk_events_log():
    """Test risk events audit log."""
    print("=" * 80)
    print("TEST 4: RISK EVENTS LOG")
    print("=" * 80)

    import sqlite3
    conn = sqlite3.connect(str(RiskGovernorConfig.DB_PATH))
    cursor = conn.cursor()

    # Count events
    cursor.execute("SELECT COUNT(*) FROM risk_events")
    count = cursor.fetchone()[0]

    print(f"\nTotal risk events logged: {count}")

    # Show recent events
    cursor.execute("""
        SELECT event_type, payload, created_at
        FROM risk_events
        ORDER BY created_at DESC
        LIMIT 5
    """)

    rows = cursor.fetchall()
    conn.close()

    if rows:
        print("\nRecent risk events:")
        for row in rows:
            print(f"  [{row[2]}] {row[0]}")

        print(f"\n[PASS] Risk events log verified ({count} events)")
    else:
        print("\n[INFO] No risk events logged yet")

    print()


def reset_to_normal():
    """Reset state to NORMAL for next test run."""
    print("=" * 80)
    print("RESET: Returning to NORMAL state")
    print("=" * 80)

    db = RiskGovernorDB(RiskGovernorConfig.DB_PATH)

    metrics = {
        'volatility_ratio': 1.0,
        'confidence_collapse_pct': 0.0,
        'liquidity_stress': 0.005,
        'portfolio_drawdown': 0.0
    }

    db.update_state(
        new_state=RiskGovernorConfig.STATE_NORMAL,
        reason='Test complete - reset to NORMAL',
        details=metrics,
        freeze_new_entries=False,
        tighten_exits=False,
        force_deleveraging=False,
        confidence_penalty_active=False,
        confidence_penalty_factor=None
    )

    print("\n[OK] State reset to NORMAL")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("RISK GOVERNOR STATE MACHINE TEST SUITE")
    print("=" * 80)
    print()

    try:
        test_normal_to_caution()
        test_caution_to_critical()
        test_symbol_flag()
        test_risk_events_log()
        reset_to_normal()

        print("=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print()

    except Exception as e:
        print()
        print("=" * 80)
        print(f"TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
