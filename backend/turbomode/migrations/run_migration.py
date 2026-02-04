#!/usr/bin/env python
"""
Database Migration Runner for Risk Governor Tables

Usage:
    python backend/turbomode/migrations/run_migration.py

This script applies the 002_create_risk_governor_tables.sql migration
to the TurboMode database.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import sqlite3
from datetime import datetime

def run_migration():
    """Apply Risk Governor database migration."""

    # Get database path
    db_path = project_root / "backend" / "data" / "turbomode.db"

    print("=" * 80)
    print("RISK GOVERNOR DATABASE MIGRATION")
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Migration: 002_create_risk_governor_tables.sql")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Read migration SQL
    migration_path = Path(__file__).parent / "002_create_risk_governor_tables.sql"

    if not migration_path.exists():
        print(f"[ERROR] Migration file not found: {migration_path}")
        return False

    with open(migration_path, 'r') as f:
        migration_sql = f.read()

    # Connect to database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Execute migration (SQLite supports multiple statements via executescript)
        cursor.executescript(migration_sql)
        conn.commit()

        print("[OK] Migration applied successfully")
        print()

        # Verify tables were created
        print("Verifying table creation...")
        print()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
            AND name IN ('global_risk_state', 'symbol_risk_flags', 'risk_events')
            ORDER BY name
        """)

        tables = [row[0] for row in cursor.fetchall()]

        if len(tables) == 3:
            print("[OK] All 3 tables created successfully:")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  - {table}: {count} rows")
            print()

            # Verify global_risk_state initialization
            cursor.execute("SELECT current_state, entered_at FROM global_risk_state WHERE id = 1")
            row = cursor.fetchone()
            if row:
                state, entered_at = row
                print(f"[OK] global_risk_state initialized: state={state}, entered_at={entered_at}")
            else:
                print("[WARNING] global_risk_state not initialized")

            print()
            print("=" * 80)
            print("MIGRATION COMPLETE")
            print("=" * 80)
            return True
        else:
            print(f"[ERROR] Expected 3 tables, found {len(tables)}")
            return False

    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


if __name__ == '__main__':
    success = run_migration()
    sys.exit(0 if success else 1)
