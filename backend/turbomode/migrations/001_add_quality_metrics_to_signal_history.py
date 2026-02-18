"""
Migration: Add quality metric columns to signal_history table

Phase: 1
Title: TurboMode.db Schema Extension for Trade Quality Analyzer
Description: Add new columns to signal_history table to store model probabilities, ATR,
             target/stop prices, reward-to-risk ratio, and directional margin.

Safety: Additive only - no existing columns modified. Existing rows remain with NULL values.
"""

import sqlite3
from pathlib import Path

def run_migration():
    """Execute schema migration for signal_history table"""
    db_path = Path('C:/StockApp/backend/data/turbomode.db')

    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    migrations = [
        ("prob_buy", "ALTER TABLE signal_history ADD COLUMN prob_buy REAL;"),
        ("prob_sell", "ALTER TABLE signal_history ADD COLUMN prob_sell REAL;"),
        ("prob_hold", "ALTER TABLE signal_history ADD COLUMN prob_hold REAL;"),
        ("entry_atr", "ALTER TABLE signal_history ADD COLUMN entry_atr REAL;"),
        ("target_price", "ALTER TABLE signal_history ADD COLUMN target_price REAL;"),
        ("stop_price", "ALTER TABLE signal_history ADD COLUMN stop_price REAL;"),
        ("rr", "ALTER TABLE signal_history ADD COLUMN rr REAL;"),
        ("directional_margin", "ALTER TABLE signal_history ADD COLUMN directional_margin REAL;")
    ]

    for column_name, sql in migrations:
        try:
            print(f"Adding column: {column_name}")
            cursor.execute(sql)
            print(f"  [OK] Successfully added {column_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"  [SKIP] Column {column_name} already exists, skipping")
            else:
                print(f"  [ERROR] Error adding {column_name}: {e}")
                raise

    conn.commit()

    # Verify schema
    print("\nVerifying new schema...")
    cursor.execute("PRAGMA table_info(signal_history);")
    columns = cursor.fetchall()

    new_columns = ['prob_buy', 'prob_sell', 'prob_hold', 'entry_atr',
                   'target_price', 'stop_price', 'rr', 'directional_margin']

    existing_column_names = [col[1] for col in columns]

    print("\nVerification Results:")
    all_present = True
    for col_name in new_columns:
        if col_name in existing_column_names:
            print(f"  [OK] {col_name}")
        else:
            print(f"  [MISSING] {col_name}")
            all_present = False

    conn.close()

    if all_present:
        print("\n[SUCCESS] Migration completed successfully!")
        print("  All new columns are present in signal_history table.")
    else:
        print("\n[FAILURE] Migration incomplete - some columns missing")
        return False

    return True

if __name__ == '__main__':
    run_migration()
