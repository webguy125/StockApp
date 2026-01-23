"""
OHLCV Coverage Validation Script for 1D/2D Horizon Training

Validates that the canonical candles table contains sufficient OHLCV data
for all training symbols over the required date range.

Database: C:\\StockApp\\master_market_data\\market_data.db
Table: candles
Symbols: From backend/turbomode/training_symbols.py (230 stocks)
Date Range: 2016-01-11 to 2026-01-06

Author: Auto-generated for 1D/2D horizon integration
Date: 2026-01-18
"""

import os
import sys
import sqlite3
from typing import Dict, List
from datetime import datetime

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from turbomode.training_symbols import get_training_symbols

# Constants
CANONICAL_DB_PATH = r"C:\StockApp\master_market_data\market_data.db"
TRAINING_START_DATE = "2016-01-11"
TRAINING_END_DATE = "2026-01-06"


def validate_ohlcv_coverage(
    db_path: str,
    symbols: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, Dict]:
    """
    Validate OHLCV coverage for all configured symbols.

    Args:
        db_path: Absolute path to market_data.db
        symbols: List of stock tickers to validate
        start_date: Required start date (YYYY-MM-DD)
        end_date: Required end date (YYYY-MM-DD)

    Returns:
        Dict mapping symbol -> coverage report with keys:
          - row_count: Number of rows in candles table
          - min_date: Earliest date in data (or None if no data)
          - max_date: Latest date in data (or None if no data)
          - has_full_range: True if min_date <= start_date AND max_date >= end_date

    Behavior:
        - Read-only DB access
        - No schema changes
        - No writes
    """
    if not os.path.isabs(db_path):
        raise ValueError(f"db_path must be absolute, got: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    results = {}

    for symbol in symbols:
        query = """
            SELECT
                COUNT(*) as row_count,
                MIN(DATE(timestamp)) as min_date,
                MAX(DATE(timestamp)) as max_date
            FROM candles
            WHERE symbol = ?
              AND timeframe = '1d'
        """

        cursor.execute(query, (symbol,))
        row = cursor.fetchone()

        row_count = row[0]
        min_date = row[1]
        max_date = row[2]

        # Determine if we have full coverage
        has_full_range = False
        if min_date and max_date:
            has_full_range = (min_date <= start_date and max_date >= end_date)

        results[symbol] = {
            'row_count': row_count,
            'min_date': min_date,
            'max_date': max_date,
            'has_full_range': has_full_range
        }

    conn.close()

    return results


def print_coverage_report(coverage: Dict[str, Dict]) -> None:
    """Print human-readable coverage report."""
    total_symbols = len(coverage)
    full_coverage = sum(1 for v in coverage.values() if v['has_full_range'])
    partial_coverage = sum(1 for v in coverage.values() if v['row_count'] > 0 and not v['has_full_range'])
    no_coverage = sum(1 for v in coverage.values() if v['row_count'] == 0)

    print()
    print("=" * 80)
    print("OHLCV COVERAGE VALIDATION REPORT")
    print("=" * 80)
    print()
    print(f"Database: {CANONICAL_DB_PATH}")
    print(f"Required Date Range: {TRAINING_START_DATE} to {TRAINING_END_DATE}")
    print(f"Total Symbols: {total_symbols}")
    print()
    print("SUMMARY:")
    print(f"  Full Coverage:    {full_coverage:3d} symbols ({100*full_coverage/total_symbols:.1f}%)")
    print(f"  Partial Coverage: {partial_coverage:3d} symbols ({100*partial_coverage/total_symbols:.1f}%)")
    print(f"  No Coverage:      {no_coverage:3d} symbols ({100*no_coverage/total_symbols:.1f}%)")
    print()

    # Print symbols with no coverage
    if no_coverage > 0:
        print("SYMBOLS WITH NO COVERAGE (0 rows):")
        for symbol, info in sorted(coverage.items()):
            if info['row_count'] == 0:
                print(f"  {symbol}")
        print()

    # Print symbols with partial coverage
    if partial_coverage > 0:
        print("SYMBOLS WITH PARTIAL COVERAGE:")
        for symbol, info in sorted(coverage.items()):
            if info['row_count'] > 0 and not info['has_full_range']:
                print(f"  {symbol:8s} | {info['row_count']:5d} rows | {info['min_date']} to {info['max_date']}")
        print()

    print("=" * 80)
    print()

    if full_coverage == total_symbols:
        print("[OK] All symbols have full coverage. Ready for 1D/2D training.")
    else:
        print(f"[WARN] {total_symbols - full_coverage} symbols missing full coverage.")
        print("       Training may proceed with reduced symbol set.")

    print()


if __name__ == '__main__':
    print("Loading training symbols from training_symbols.py...")
    symbols = get_training_symbols()
    print(f"Loaded {len(symbols)} training symbols")

    print()
    print("Validating OHLCV coverage in candles table...")

    coverage = validate_ohlcv_coverage(
        CANONICAL_DB_PATH,
        symbols,
        TRAINING_START_DATE,
        TRAINING_END_DATE
    )

    print_coverage_report(coverage)

    # Save detailed report to file
    report_file = r"C:\StockApp\backend\turbomode\ohlcv_coverage_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"OHLCV Coverage Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Database: {CANONICAL_DB_PATH}\n")
        f.write(f"Required Range: {TRAINING_START_DATE} to {TRAINING_END_DATE}\n")
        f.write(f"\n")
        f.write(f"Symbol    | Row Count | Min Date   | Max Date   | Full Range?\n")
        f.write(f"----------+-----------+------------+------------+------------\n")

        for symbol in sorted(coverage.keys()):
            info = coverage[symbol]
            min_d = info['min_date'] or 'N/A'
            max_d = info['max_date'] or 'N/A'
            full = 'YES' if info['has_full_range'] else 'NO'
            f.write(f"{symbol:8s}  | {info['row_count']:9d} | {min_d:10s} | {max_d:10s} | {full}\n")

    print(f"Detailed report saved to: {report_file}")
