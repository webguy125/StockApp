
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Create hold_rankings table for HOLD intelligence tracking.

This table stores nightly analysis results for HOLD signals only.
Completely separate from BUY/SELL performance tracking.
"""

import sqlite3
from datetime import datetime


def create_hold_rankings_table(db_path: str, dry_run: bool = False):
    """
    Create hold_rankings table for HOLD intelligence analysis.

    Args:
        db_path: Path to turbomode.db
        dry_run: If True, show what would happen without making changes
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print('HOLD RANKINGS TABLE CREATION')
    print()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hold_rankings'")
    table_exists = cursor.fetchone() is not None

    if table_exists:
        print('[INFO] hold_rankings table already exists')
        cursor.execute('SELECT COUNT(*) FROM hold_rankings')
        count = cursor.fetchone()[0]
        print(f'[INFO] Current rows: {count:,}')
        conn.close()
        return

    if dry_run:
        print('[DRY RUN] Would create hold_rankings table')
        print()
        conn.close()
        return

    print('Creating hold_rankings table...')

    # Create hold_rankings table
    cursor.execute('''
        CREATE TABLE hold_rankings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            exit_date TEXT NOT NULL,
            hold_days INTEGER,
            entry_price REAL,
            exit_price REAL,
            profit_loss_pct REAL,
            sector TEXT,
            market_cap TEXT,

            -- Stability metrics
            price_drift_pct REAL,
            max_intraday_drift_pct REAL,
            volatility_drift_pct REAL,
            range_compression_score REAL,
            sector_stability_factor REAL,
            stability_score REAL,

            -- Volatility compression metrics
            iv_compression_pct REAL,
            atr_compression_pct REAL,
            bandwidth_compression REAL,
            volatility_floor_test REAL,
            volatility_score REAL,

            -- Top 20 flag
            is_top_20 INTEGER DEFAULT 0,
            top_20_rank INTEGER,

            created_at TEXT NOT NULL
        )
    ''')

    print('[OK] hold_rankings table created')
    print()

    # Create indexes
    print('Creating indexes...')

    cursor.execute('''
        CREATE INDEX idx_hold_rankings_analysis_date
        ON hold_rankings(analysis_date)
    ''')

    cursor.execute('''
        CREATE INDEX idx_hold_rankings_symbol
        ON hold_rankings(symbol)
    ''')

    cursor.execute('''
        CREATE INDEX idx_hold_rankings_stability_score
        ON hold_rankings(stability_score DESC)
    ''')

    cursor.execute('''
        CREATE INDEX idx_hold_rankings_volatility_score
        ON hold_rankings(volatility_score DESC)
    ''')

    cursor.execute('''
        CREATE INDEX idx_hold_rankings_top_20
        ON hold_rankings(is_top_20, top_20_rank)
    ''')

    print('[OK] Indexes created')
    print()

    conn.commit()
    print('[OK] Transaction committed')
    print()
    print('HOLD RANKINGS TABLE CREATION COMPLETE')
    print()
    print('Table structure:')
    print('  - analysis_date: Date of nightly analysis')
    print('  - symbol, entry_date, exit_date, hold_days: Signal details')
    print('  - stability_score: 0-100 composite stability score')
    print('  - volatility_score: 0-100 composite volatility compression score')
    print('  - is_top_20: Flag for Top 20 candidates')
    print('  - All individual metrics preserved for diagnostics')

    conn.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create hold_rankings table')
    parser.add_argument('--db', default='backend/data/turbomode.db', help='Path to database')
    parser.add_argument('--dry-run', action='store_true', help='Show what would happen without making changes')

    args = parser.parse_args()

    db_path = args.db if args.db.startswith('/') or ':' in args.db else str(project_root / args.db)

    create_hold_rankings_table(db_path, dry_run=args.dry_run)
