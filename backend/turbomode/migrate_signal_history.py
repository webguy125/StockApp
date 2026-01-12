"""
Migrate signal_history table to support training sample generation
Adds missing columns for outcome tracking
"""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'turbomode.db')

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Migrating signal_history table...")

try:
    # Add missing columns
    cursor.execute("ALTER TABLE signal_history ADD COLUMN signal_id TEXT")
    print("[OK] Added signal_id column")
except:
    print("[SKIP] signal_id column already exists")

try:
    cursor.execute("ALTER TABLE signal_history ADD COLUMN outcome TEXT")
    print("[OK] Added outcome column")
except:
    print("[SKIP] outcome column already exists")

try:
    cursor.execute("ALTER TABLE signal_history ADD COLUMN is_correct INTEGER DEFAULT 0")
    print("[OK] Added is_correct column")
except:
    print("[SKIP] is_correct column already exists")

try:
    cursor.execute("ALTER TABLE signal_history ADD COLUMN return_pct REAL")
    print("[OK] Added return_pct column")
except:
    print("[SKIP] return_pct column already exists")

try:
    cursor.execute("ALTER TABLE signal_history ADD COLUMN processed_for_training INTEGER DEFAULT 0")
    print("[OK] Added processed_for_training column")
except:
    print("[SKIP] processed_for_training column already exists")

try:
    cursor.execute("ALTER TABLE signal_history ADD COLUMN updated_at TEXT")
    print("[OK] Added updated_at column")
except:
    print("[SKIP] updated_at column already exists")

conn.commit()
conn.close()

print("\n[OK] Migration complete!")
