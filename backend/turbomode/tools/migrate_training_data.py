
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Contamination-Proof Signal Lifecycle Migration

This script:
1. Creates training_samples table for synthetic backtest data
2. Moves all synthetic data from signal_history to training_samples
3. Leaves signal_history empty for real closed trades only

Run once to establish clean separation between training and performance tracking.
"""

import sqlite3
from datetime import datetime

def migrate_training_data(db_path: str, dry_run: bool = False):
    """
    Migrate synthetic training data to separate table.

    Args:
        db_path: Path to turbomode.db
        dry_run: If True, show what would happen without making changes
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print('CONTAMINATION-PROOF SIGNAL LIFECYCLE MIGRATION')
    print()

    # Step 1: Count current state
    cursor.execute('SELECT COUNT(*) FROM signal_history')
    signal_history_count = cursor.fetchone()[0]
    print(f'Current signal_history rows: {signal_history_count:,}')

    # Check if training_samples table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_samples'")
    training_samples_exists = cursor.fetchone() is not None

    if training_samples_exists:
        cursor.execute('SELECT COUNT(*) FROM training_samples')
        training_samples_count = cursor.fetchone()[0]
        print(f'Current training_samples rows: {training_samples_count:,}')
    else:
        print('training_samples table: does not exist')

    if dry_run:
        print('\nDRY RUN MODE - No changes will be made')

    print()

    # Step 2: Create training_samples table if it doesn't exist
    if not training_samples_exists:
        print('Creating training_samples table...')
        if not dry_run:
            cursor.execute('''
                CREATE TABLE training_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    profit_loss_pct REAL,
                    confidence REAL,
                    hold_days INTEGER,
                    market_cap TEXT,
                    sector TEXT,
                    source TEXT DEFAULT 'training_backtest',
                    created_at TEXT,
                    migrated_at TEXT
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_training_samples_symbol
                ON training_samples(symbol)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_training_samples_entry_date
                ON training_samples(entry_date)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_training_samples_signal_type
                ON training_samples(signal_type)
            ''')

            print('[OK] training_samples table created')
        else:
            print('[DRY RUN] Would create training_samples table')
    else:
        print('training_samples table already exists')

    print()

    # Step 3: Move all signal_history rows to training_samples
    print(f'Moving {signal_history_count:,} rows from signal_history to training_samples...')

    if not dry_run:
        # All current signal_history rows are synthetic (from bootstrap)
        # Created date range 2016-01-27 to 2026-01-09 confirms this is backtest data
        cursor.execute('''
            INSERT INTO training_samples (
                symbol, signal_type, entry_date, exit_date, entry_price, exit_price,
                profit_loss_pct, confidence, hold_days, market_cap, sector,
                source, created_at, migrated_at
            )
            SELECT
                symbol, signal_type, entry_date, exit_date, entry_price, exit_price,
                profit_loss_pct, confidence, hold_days, market_cap, sector,
                'training_backtest',
                created_at,
                ?
            FROM signal_history
        ''', (datetime.now().isoformat(),))

        migrated_count = cursor.rowcount
        print(f'[OK] Migrated {migrated_count:,} rows to training_samples')
    else:
        print(f'[DRY RUN] Would migrate {signal_history_count:,} rows to training_samples')

    print()

    # Step 4: Delete all rows from signal_history
    print('Purging signal_history (all rows are synthetic)...')

    if not dry_run:
        cursor.execute('DELETE FROM signal_history')
        deleted_count = cursor.rowcount
        print(f'[OK] Deleted {deleted_count:,} rows from signal_history')

        # Verify clean state
        cursor.execute('SELECT COUNT(*) FROM signal_history')
        remaining = cursor.fetchone()[0]
        if remaining > 0:
            print(f'[ERROR] {remaining} rows remain in signal_history!')
            conn.rollback()
            conn.close()
            raise Exception('signal_history purge failed')
    else:
        print(f'[DRY RUN] Would delete {signal_history_count:,} rows from signal_history')

    print()

    # Step 5: Commit changes
    if not dry_run:
        conn.commit()
        print('[OK] Transaction committed')

        # Final verification
        cursor.execute('SELECT COUNT(*) FROM signal_history')
        final_signal_history = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM training_samples')
        final_training_samples = cursor.fetchone()[0]

        print()
        print('MIGRATION COMPLETE')
        print()
        print(f'signal_history rows: {final_signal_history} (should be 0)')
        print(f'training_samples rows: {final_training_samples:,}')
        print()
        print('signal_history is now clean and ready for real closed trades only.')
        print('training_samples contains synthetic backtest data for ML training only.')
    else:
        print()
        print('DRY RUN COMPLETE - No changes were made')
        print()
        print('Run without --dry-run to execute migration')

    conn.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Migrate synthetic training data to separate table')
    parser.add_argument('--db', default='backend/data/turbomode.db', help='Path to database')
    parser.add_argument('--dry-run', action='store_true', help='Show what would happen without making changes')

    args = parser.parse_args()

    db_path = args.db if args.db.startswith('/') or ':' in args.db else str(project_root / args.db)

    migrate_training_data(db_path, dry_run=args.dry_run)
