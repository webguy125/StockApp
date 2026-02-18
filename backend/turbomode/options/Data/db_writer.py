import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'options_intel.db')


def write_enriched_to_db(data):
    """
    Write enriched signal data to options_intel.db.

    Args:
        data: Dictionary containing enriched signal fields

    Returns:
        True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Add timestamp
        data['snapshot_timestamp'] = datetime.utcnow().isoformat()

        # Build dynamic SQL based on available fields
        fields = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])

        sql = f"INSERT OR REPLACE INTO enriched_options_signals ({fields}) VALUES ({placeholders})"
        cursor.execute(sql, list(data.values()))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB_WRITER] Error writing to DB: {e}")
        return False
