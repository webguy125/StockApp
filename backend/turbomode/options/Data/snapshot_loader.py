import os
import sqlite3

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'options_intel.db')


def load_snapshot(symbol):
    """
    Load cached enriched signal intelligence for a single symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary containing enriched signal data, or None if not found
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM signal_intelligence
            WHERE symbol = ?
            ORDER BY last_updated DESC
            LIMIT 1
        ''', (symbol,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    except Exception as e:
        print(f"[SNAPSHOT_LOADER] Error loading snapshot for {symbol}: {e}")
        return None


def load_all_snapshots():
    """
    Load all cached enriched signal intelligence.

    Returns:
        List of dictionaries containing enriched signal data
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM signal_intelligence
            ORDER BY signal_quality_score DESC
        ''')

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    except Exception as e:
        print(f"[SNAPSHOT_LOADER] Error loading all snapshots: {e}")
        return []


def load_snapshots_by_sector(sector):
    """
    Load cached enriched signals for a specific sector.

    Args:
        sector: Sector name

    Returns:
        List of dictionaries containing enriched signal data
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM signal_intelligence
            WHERE sector = ?
            ORDER BY signal_quality_score DESC
        ''', (sector,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    except Exception as e:
        print(f"[SNAPSHOT_LOADER] Error loading snapshots for sector {sector}: {e}")
        return []


def get_previous_snapshot(symbol):
    """
    Get the previous snapshot for transition detection.
    This is used to compare current vs previous regime state.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with previous snapshot data, or None if not found
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get the most recent snapshot for this symbol
        cursor.execute('''
            SELECT regime_state, drift, quality_score, last_updated
            FROM signal_intelligence
            WHERE symbol = ?
            ORDER BY last_updated DESC
            LIMIT 1
        ''', (symbol,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    except Exception as e:
        print(f"[SNAPSHOT_LOADER] Error loading previous snapshot for {symbol}: {e}")
        return None


def get_cache_age_minutes():
    """
    Get the age of the oldest cache entry in minutes.
    Used to determine if a refresh is needed.

    Returns:
        Age in minutes, or None if no cache exists
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT MIN(last_updated) as oldest
            FROM signal_intelligence
        ''')

        row = cursor.fetchone()
        conn.close()

        if row and row[0]:
            from datetime import datetime
            oldest = datetime.fromisoformat(row[0])
            age_seconds = (datetime.now() - oldest).total_seconds()
            return age_seconds / 60.0
        return None

    except Exception as e:
        print(f"[SNAPSHOT_LOADER] Error calculating cache age: {e}")
        return None
