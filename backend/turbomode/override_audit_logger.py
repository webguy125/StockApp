"""
Override Audit Logger
Tracks all directional override decisions for analysis and future retraining
"""

import os
import csv
import threading
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


# Thread lock for safe concurrent writes
_write_lock = threading.Lock()


def log_override_decision(entry: Dict, output_path: str = None) -> bool:
    """
    Log a directional override decision to CSV file.

    Args:
        entry: Dictionary containing:
            - timestamp: ISO string (e.g. '2026-01-11T19:30:00')
            - symbol: Stock ticker (e.g. 'TSLA')
            - prob_buy: float (0.0 - 1.0)
            - prob_hold: float (0.0 - 1.0)
            - prob_sell: float (0.0 - 1.0)
            - override_triggered: bool (True if override logic was applied)
            - final_prediction: str ('buy', 'sell', or 'hold')
            - override_count: int (number of base models that were overridden)
            - actual_outcome: str (optional, e.g. 'profit', 'loss', 'neutral', or empty)
            - entry_price: float (optional, price at prediction time)
            - exit_price: float (optional, price at outcome evaluation)
            - days_held: int (optional, days between entry and exit)

        output_path: Path to CSV file (default: backend/data/override_audit.csv)

    Returns:
        bool: True if logged successfully, False otherwise
    """
    # Default output path
    if output_path is None:
        base_dir = Path(__file__).parent.parent
        output_path = base_dir / 'data' / 'override_audit.csv'
    else:
        output_path = Path(output_path)

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV header (written only if file doesn't exist)
    fieldnames = [
        'timestamp',
        'symbol',
        'prob_buy',
        'prob_hold',
        'prob_sell',
        'override_triggered',
        'final_prediction',
        'override_count',
        'actual_outcome',
        'entry_price',
        'exit_price',
        'days_held',
        'prob_asymmetry',      # abs(prob_buy - prob_sell)
        'max_directional',     # max(prob_buy, prob_sell)
        'neutral_dominance'    # prob_hold - max(prob_buy, prob_sell)
    ]

    # Calculate derived metrics
    prob_buy = entry.get('prob_buy', 0.0)
    prob_sell = entry.get('prob_sell', 0.0)
    prob_hold = entry.get('prob_hold', 0.0)

    prob_asymmetry = abs(prob_buy - prob_sell)
    max_directional = max(prob_buy, prob_sell)
    neutral_dominance = prob_hold - max_directional

    # Build row data
    row = {
        'timestamp': entry.get('timestamp', datetime.now().isoformat()),
        'symbol': entry.get('symbol', ''),
        'prob_buy': f"{prob_buy:.6f}",
        'prob_hold': f"{prob_hold:.6f}",
        'prob_sell': f"{prob_sell:.6f}",
        'override_triggered': str(entry.get('override_triggered', False)),
        'final_prediction': entry.get('final_prediction', ''),
        'override_count': entry.get('override_count', 0),
        'actual_outcome': entry.get('actual_outcome', ''),
        'entry_price': entry.get('entry_price', ''),
        'exit_price': entry.get('exit_price', ''),
        'days_held': entry.get('days_held', ''),
        'prob_asymmetry': f"{prob_asymmetry:.6f}",
        'max_directional': f"{max_directional:.6f}",
        'neutral_dominance': f"{neutral_dominance:.6f}"
    }

    # Thread-safe write
    try:
        with _write_lock:
            # Check if file exists to determine if header is needed
            file_exists = output_path.exists()

            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()

                # Write data row
                writer.writerow(row)

        return True

    except Exception as e:
        print(f"[ERROR] Override audit logger failed: {e}")
        return False


def update_outcome(symbol: str, timestamp: str, actual_outcome: str,
                   exit_price: float = None, days_held: int = None,
                   output_path: str = None) -> bool:
    """
    Update an existing log entry with actual outcome data.

    This function reads the CSV, finds the matching entry by symbol and timestamp,
    and updates the outcome fields.

    Args:
        symbol: Stock ticker
        timestamp: ISO timestamp of the original prediction
        actual_outcome: Actual result ('profit', 'loss', 'neutral')
        exit_price: Price at exit (optional)
        days_held: Number of days position was held (optional)
        output_path: Path to CSV file

    Returns:
        bool: True if updated successfully, False otherwise
    """
    # Default output path
    if output_path is None:
        base_dir = Path(__file__).parent.parent
        output_path = base_dir / 'data' / 'override_audit.csv'
    else:
        output_path = Path(output_path)

    if not output_path.exists():
        print(f"[ERROR] Audit log not found: {output_path}")
        return False

    try:
        with _write_lock:
            # Read all rows
            rows = []
            with open(output_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)

            # Find and update matching row
            updated = False
            for row in rows:
                if row['symbol'] == symbol and row['timestamp'] == timestamp:
                    row['actual_outcome'] = actual_outcome
                    if exit_price is not None:
                        row['exit_price'] = f"{exit_price:.2f}"
                    if days_held is not None:
                        row['days_held'] = str(days_held)
                    updated = True
                    break

            if not updated:
                print(f"[WARNING] No matching entry found for {symbol} at {timestamp}")
                return False

            # Write back all rows
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            return True

    except Exception as e:
        print(f"[ERROR] Failed to update outcome: {e}")
        return False


def get_override_statistics(output_path: str = None) -> Dict:
    """
    Analyze override audit log and return statistics.

    Args:
        output_path: Path to CSV file

    Returns:
        Dictionary with statistics:
            - total_predictions: int
            - override_count: int
            - override_rate: float
            - outcome_accuracy: float (if outcomes available)
            - override_accuracy: float (accuracy when override was triggered)
            - no_override_accuracy: float (accuracy when no override)
    """
    # Default output path
    if output_path is None:
        base_dir = Path(__file__).parent.parent
        output_path = base_dir / 'data' / 'override_audit.csv'
    else:
        output_path = Path(output_path)

    if not output_path.exists():
        return {
            'total_predictions': 0,
            'override_count': 0,
            'override_rate': 0.0,
            'error': 'Audit log not found'
        }

    try:
        with open(output_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        total = len(rows)
        override_count = sum(1 for r in rows if r['override_triggered'] == 'True')
        override_rate = override_count / total if total > 0 else 0.0

        # Calculate outcome accuracy (if available)
        rows_with_outcome = [r for r in rows if r['actual_outcome'] in ['profit', 'loss']]

        if len(rows_with_outcome) > 0:
            # Profit = correct prediction
            correct = sum(1 for r in rows_with_outcome if r['actual_outcome'] == 'profit')
            overall_accuracy = correct / len(rows_with_outcome)

            # Override vs no-override accuracy
            override_rows = [r for r in rows_with_outcome if r['override_triggered'] == 'True']
            no_override_rows = [r for r in rows_with_outcome if r['override_triggered'] == 'False']

            override_correct = sum(1 for r in override_rows if r['actual_outcome'] == 'profit')
            no_override_correct = sum(1 for r in no_override_rows if r['actual_outcome'] == 'profit')

            override_accuracy = override_correct / len(override_rows) if len(override_rows) > 0 else 0.0
            no_override_accuracy = no_override_correct / len(no_override_rows) if len(no_override_rows) > 0 else 0.0

            return {
                'total_predictions': total,
                'override_count': override_count,
                'override_rate': override_rate,
                'predictions_with_outcomes': len(rows_with_outcome),
                'overall_accuracy': overall_accuracy,
                'override_accuracy': override_accuracy,
                'no_override_accuracy': no_override_accuracy
            }
        else:
            return {
                'total_predictions': total,
                'override_count': override_count,
                'override_rate': override_rate,
                'predictions_with_outcomes': 0
            }

    except Exception as e:
        return {
            'total_predictions': 0,
            'override_count': 0,
            'override_rate': 0.0,
            'error': str(e)
        }


if __name__ == '__main__':
    # Test the logger
    print("Testing override audit logger...")

    # Test entry 1: Override triggered
    test_entry_1 = {
        'timestamp': '2026-01-11T21:45:00',
        'symbol': 'TSLA',
        'prob_buy': 0.024,
        'prob_hold': 0.797,
        'prob_sell': 0.182,
        'override_triggered': True,
        'final_prediction': 'sell',
        'override_count': 4,
        'entry_price': 432.11
    }

    success = log_override_decision(test_entry_1)
    print(f"Test 1 (TSLA override): {'[OK]' if success else '[FAIL]'}")

    # Test entry 2: No override
    test_entry_2 = {
        'timestamp': '2026-01-11T21:45:00',
        'symbol': 'AAPL',
        'prob_buy': 0.011,
        'prob_hold': 0.981,
        'prob_sell': 0.008,
        'override_triggered': False,
        'final_prediction': 'hold',
        'override_count': 1,
        'entry_price': 262.72
    }

    success = log_override_decision(test_entry_2)
    print(f"Test 2 (AAPL no override): {'[OK]' if success else '[FAIL]'}")

    # Test outcome update
    success = update_outcome(
        symbol='TSLA',
        timestamp='2026-01-11T21:45:00',
        actual_outcome='profit',
        exit_price=420.50,
        days_held=3
    )
    print(f"Test 3 (outcome update): {'[OK]' if success else '[FAIL]'}")

    # Get statistics
    stats = get_override_statistics()
    print(f"\nAudit Statistics:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Override count: {stats['override_count']}")
    print(f"  Override rate: {stats['override_rate']:.1%}")

    print("\n[OK] Override audit logger tests complete")
    print("Log file: backend/data/override_audit.csv")
