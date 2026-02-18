import sqlite3
import os

db_path = r"C:\StockApp\backend\data\turbomode.db"

if not os.path.exists(db_path):
    print(f"[ERROR] Database not found: {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(active_signals)")
    columns = cursor.fetchall()

    print(f"Checking active_signals table in: {db_path}")
    print("=" * 60)

    prob_columns = [col for col in columns if 'prob' in col[1].lower()]

    if prob_columns:
        print("Columns with 'prob':")
        for col in prob_columns:
            print(f"  - {col[1]:20s} (Type: {col[2]}, Nullable: {not col[3]})")

        # Check specifically for prob_hold
        if any(col[1] == 'prob_hold' for col in prob_columns):
            print("\n✓ prob_hold column EXISTS")
        else:
            print("\n✗ prob_hold column NOT FOUND")
    else:
        print("✗ No probability columns found in active_signals")

    conn.close()
