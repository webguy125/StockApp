"""
Migration: Add entry_min and entry_max columns to active_signals table
For realistic morning execution with gap tolerance (±2%)
"""

import sqlite3
import os
import sys

def migrate():
    """Add entry_min and entry_max columns"""

    # Get database path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    db_path = os.path.join(project_root, "backend/data/turbomode.db")

    print(f"[MIGRATION] Connecting to: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(active_signals)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'entry_min' in columns and 'entry_max' in columns:
            print("[OK] Columns entry_min and entry_max already exist")
            return

        # Add entry_min column (2% below entry_price)
        if 'entry_min' not in columns:
            print("[MIGRATION] Adding entry_min column...")
            cursor.execute("""
                ALTER TABLE active_signals
                ADD COLUMN entry_min REAL
            """)

            # Populate existing rows with entry_price * 0.98
            cursor.execute("""
                UPDATE active_signals
                SET entry_min = entry_price * 0.98
                WHERE entry_min IS NULL
            """)
            print("[OK] Added entry_min column")

        # Add entry_max column (2% above entry_price)
        if 'entry_max' not in columns:
            print("[MIGRATION] Adding entry_max column...")
            cursor.execute("""
                ALTER TABLE active_signals
                ADD COLUMN entry_max REAL
            """)

            # Populate existing rows with entry_price * 1.02
            cursor.execute("""
                UPDATE active_signals
                SET entry_max = entry_price * 1.02
                WHERE entry_max IS NULL
            """)
            print("[OK] Added entry_max column")

        conn.commit()
        print("[SUCCESS] Migration complete!")

        # Show sample data
        cursor.execute("""
            SELECT symbol, entry_price, entry_min, entry_max, target_price, stop_price
            FROM active_signals
            LIMIT 3
        """)

        print("\nSample data:")
        print(f"{'Symbol':<8} {'Entry':<10} {'Min (-2%)':<12} {'Max (+2%)':<12} {'Target':<10} {'Stop':<10}")
        print("-" * 70)
        for row in cursor.fetchall():
            symbol, entry, entry_min, entry_max, target, stop = row
            print(f"{symbol:<8} ${entry:<9.2f} ${entry_min:<11.2f} ${entry_max:<11.2f} ${target:<9.2f} ${stop:<9.2f}")

    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        conn.rollback()
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
