"""
Database Migration: Fix Signal Lifecycle Schema
Adds current_price, signal_timestamp, and fixes UNIQUE constraint
"""

import sqlite3
import os
from datetime import datetime

def migrate_database():
    """
    Migration to fix signal lifecycle issues:
    1. Add current_price field (updated each scan)
    2. Add signal_timestamp field (fixed at creation)
    3. Add entry_min, entry_max if missing
    4. Change UNIQUE constraint from (symbol, signal_type) to (symbol)
    5. Backfill signal_timestamp from created_at for existing records
    """

    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'turbomode.db')

    if not os.path.exists(db_path):
        print(f'[ERROR] Database not found: {db_path}')
        return False

    print(f'[MIGRATION] Starting database migration...')
    print(f'[MIGRATION] Database: {db_path}')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if migration is needed
        cursor.execute("PRAGMA table_info(active_signals)")
        columns = {col[1]: col for col in cursor.fetchall()}

        needs_migration = (
            'current_price' not in columns or
            'signal_timestamp' not in columns or
            'entry_min' not in columns or
            'entry_max' not in columns
        )

        if not needs_migration:
            print('[MIGRATION] Database already migrated. Checking UNIQUE constraint...')

            # Check UNIQUE constraint
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='active_signals'")
            table_sql = cursor.fetchone()[0]

            if 'UNIQUE(symbol, signal_type)' in table_sql:
                print('[MIGRATION] UNIQUE constraint needs update...')
                needs_migration = True
            else:
                print('[MIGRATION] No migration needed.')
                return True

        if needs_migration:
            print('[MIGRATION] Creating new table with updated schema...')

            # Create new table with correct schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS active_signals_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,

                    -- Entry data (FIXED - never changes)
                    entry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_min REAL,
                    entry_max REAL,
                    signal_timestamp TEXT NOT NULL,

                    -- Current data (UPDATED each scan)
                    current_price REAL NOT NULL,

                    -- Targets (based on entry_price)
                    target_price REAL NOT NULL,
                    stop_price REAL NOT NULL,

                    -- Classifications
                    market_cap TEXT NOT NULL,
                    sector TEXT NOT NULL,

                    -- Lifecycle (UPDATED each scan)
                    age_days INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'ACTIVE',

                    -- Metadata
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    UNIQUE(symbol)  -- Only one signal per symbol (allows flipping)
                )
            """)

            print('[MIGRATION] Copying data from old table...')

            # Copy existing data
            cursor.execute("""
                INSERT INTO active_signals_new
                (id, symbol, signal_type, confidence, entry_date, entry_price,
                 entry_min, entry_max, signal_timestamp, current_price,
                 target_price, stop_price, market_cap, sector, age_days, status,
                 created_at, updated_at)
                SELECT
                    id, symbol, signal_type, confidence, entry_date, entry_price,
                    COALESCE(entry_min, entry_price * 0.98),
                    COALESCE(entry_max, entry_price * 1.02),
                    created_at as signal_timestamp,
                    entry_price as current_price,
                    target_price, stop_price, market_cap, sector, age_days, status,
                    created_at, updated_at
                FROM active_signals
            """)

            rows_copied = cursor.rowcount
            print(f'[MIGRATION] Copied {rows_copied} signals')

            # Drop old table
            print('[MIGRATION] Dropping old table...')
            cursor.execute("DROP TABLE active_signals")

            # Rename new table
            print('[MIGRATION] Renaming new table...')
            cursor.execute("ALTER TABLE active_signals_new RENAME TO active_signals")

            conn.commit()
            print('[MIGRATION] Migration completed successfully!')

            # Verify
            cursor.execute("SELECT COUNT(*) FROM active_signals")
            count = cursor.fetchone()[0]
            print(f'[MIGRATION] Verified: {count} signals in new table')

            return True

    except Exception as e:
        print(f'[ERROR] Migration failed: {e}')
        conn.rollback()
        return False

    finally:
        conn.close()

if __name__ == '__main__':
    success = migrate_database()
    if success:
        print('[OK] Database migration completed successfully')
    else:
        print('[FAIL] Database migration failed')
