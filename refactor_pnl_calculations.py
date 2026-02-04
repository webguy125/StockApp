"""
Unified P&L Refactoring Script

This script refactors all P&L calculations to use the canonical calculate_pnl_pct() function.
It also fixes existing corrupted P&L data in signal_history where SELL positions have inverted P/L.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import sqlite3
from backend.turbomode.core_engine.pnl_utils import calculate_pnl_pct


def fix_signal_history_pnl(db_path: str, dry_run: bool = False):
    """
    Fix corrupted P/L values in signal_history for SELL positions.

    SELL positions currently have inverted P/L due to bug in signal_closer.py.
    This function recalculates correct P/L using canonical calculate_pnl_pct().
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 80)
    print("FIXING CORRUPTED P/L DATA IN SIGNAL_HISTORY")
    print("=" * 80)
    print()

    # Get all SELL positions
    cursor.execute('''
        SELECT id, symbol, signal_type, entry_price, exit_price, profit_loss_pct
        FROM signal_history
        WHERE signal_type = 'SELL'
    ''')

    sell_positions = cursor.fetchall()

    if len(sell_positions) == 0:
        print("[INFO] No SELL positions found in signal_history")
        conn.close()
        return

    print(f"Found {len(sell_positions)} SELL positions to fix")
    print()

    fixes = []

    for row in sell_positions:
        signal_id, symbol, signal_type, entry_price, exit_price, old_pnl = row

        # Recalculate correct P/L
        correct_pnl = calculate_pnl_pct(entry_price, exit_price, signal_type)

        # Check if it needs fixing (inverted sign)
        if old_pnl != correct_pnl:
            fixes.append({
                'id': signal_id,
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'old_pnl': old_pnl,
                'correct_pnl': correct_pnl
            })

            print(f"{symbol:6s} SELL | Entry ${entry_price:.2f} -> Exit ${exit_price:.2f}")
            print(f"  OLD P/L (WRONG): {old_pnl:+.2f}%")
            print(f"  NEW P/L (CORRECT): {correct_pnl:+.2f}%")
            print()

    if len(fixes) == 0:
        print("[INFO] All SELL positions already have correct P/L")
        conn.close()
        return

    if dry_run:
        print(f"[DRY RUN] Would fix {len(fixes)} SELL positions")
        conn.close()
        return

    # Apply fixes
    print(f"Applying fixes to {len(fixes)} positions...")
    for fix in fixes:
        cursor.execute('''
            UPDATE signal_history
            SET profit_loss_pct = ?
            WHERE id = ?
        ''', (fix['correct_pnl'], fix['id']))

    conn.commit()
    print(f"[OK] Fixed {len(fixes)} SELL positions")
    print()

    # Verify
    cursor.execute('''
        SELECT symbol, signal_type, entry_price, exit_price, profit_loss_pct
        FROM signal_history
        WHERE signal_type = 'SELL'
    ''')

    print("VERIFICATION - All SELL positions after fix:")
    print("-" * 80)
    for row in cursor.fetchall():
        symbol, signal_type, entry, exit, pnl = row
        print(f"{symbol:6s} SELL | Entry ${entry:.2f} -> Exit ${exit:.2f} | P/L: {pnl:+.2f}%")

    conn.close()
    print()
    print("=" * 80)
    print("P/L FIX COMPLETE")
    print("=" * 80)


def audit_all_pnl_calculations():
    """
    Generate audit report of all files that needed P/L refactoring.
    """
    print()
    print("=" * 80)
    print("P/L REFACTORING AUDIT REPORT")
    print("=" * 80)
    print()

    print("FILES REQUIRING P/L CALCULATION REFACTORING:")
    print()

    print("1. backend/turbomode/core_engine/signal_closer.py")
    print("   Location: check_exit_conditions() method, line ~79")
    print("   Issue: Uses ((current_price - entry_price) / entry_price) * 100 for ALL signal types")
    print("   Impact: SELL positions have inverted P/L (winners appear as losers)")
    print("   Fix: Replace with calculate_pnl_pct(entry_price, current_price, signal_type)")
    print()

    print("2. backend/turbomode/core_engine/position_manager.py")
    print("   Location: close_position() method, line ~231 and ~234")
    print("   Issue: Has separate logic for long/short but uses different formula")
    print("   Impact: Potential inconsistency with signal_closer calculations")
    print("   Fix: Replace with calculate_pnl_pct(entry_price, exit_price, position_type)")
    print()

    print("FILES THAT ONLY READ P/L (NO REFACTORING NEEDED):")
    print()
    print("- backend/turbomode/adaptive_stock_ranker.py (reads from signal_history)")
    print("- backend/turbomode/tools/create_hold_rankings_table.py (schema definition)")
    print("- backend/turbomode/core_engine/hold_intelligence.py (reads from signal_history)")
    print("- backend/turbomode/database_schema.py (schema definition)")
    print()

    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Refactor P/L calculations')
    parser.add_argument('--db', default='backend/data/turbomode.db', help='Path to database')
    parser.add_argument('--dry-run', action='store_true', help='Show what would happen without making changes')
    parser.add_argument('--audit-only', action='store_true', help='Only generate audit report')

    args = parser.parse_args()

    db_path = args.db if args.db.startswith('/') or ':' in args.db else str(project_root / args.db)

    if args.audit_only:
        audit_all_pnl_calculations()
    else:
        # Fix signal_history data
        fix_signal_history_pnl(db_path, dry_run=args.dry_run)

        # Generate audit report
        audit_all_pnl_calculations()

        print()
        print("NEXT STEPS:")
        print()
        print("1. Code refactoring required in 2 files:")
        print("   - backend/turbomode/core_engine/signal_closer.py")
        print("   - backend/turbomode/core_engine/position_manager.py")
        print()
        print("2. Run scanner to verify SELL positions calculate correctly")
        print()
