"""
Rare Event Archive Loader
Loads and serves samples from curated historical market stress events
"""

import sqlite3
import json
import os
import random
from typing import List, Dict, Any, Optional
from datetime import datetime


class RareEventArchive:
    """
    Manages rare event archive samples for training

    Provides weighted sampling from historical crash/stress events
    to supplement the rolling 5-year training window
    """

    def __init__(self, archive_path: str = "backend/data/rare_event_archive"):
        """
        Initialize rare event archive

        Args:
            archive_path: Path to archive directory
        """
        self.archive_path = archive_path
        self.db_path = os.path.join(archive_path, "archive.db")
        self.config_path = os.path.join(archive_path, "metadata", "archive_config.json")

        # Load configuration
        self.config = self._load_config()

        # Check if archive exists
        self.archive_exists = os.path.exists(self.db_path)

        if not self.archive_exists:
            print(f"[INFO] Rare event archive not found at {self.db_path}")
            print(f"[INFO] Run generate_rare_event_archive.py to create it")

    def _load_config(self) -> Dict[str, Any]:
        """Load archive configuration"""
        if not os.path.exists(self.config_path):
            # Return default config
            return {
                'archive_mix_ratio': 0.07,
                'event_weights': {},
                'active_events': []
            }

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def load_archive_samples(self, total_samples: int, stratified: bool = True) -> List[Dict[str, Any]]:
        """
        Load weighted samples from archive

        Args:
            total_samples: Total number of archive samples to load
            stratified: Whether to respect event weights

        Returns:
            List of sample dictionaries
        """
        if not self.archive_exists:
            print("[WARNING] Archive not available, returning empty list")
            return []

        if total_samples <= 0:
            return []

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        samples = []

        if stratified:
            # Load samples respecting event weights
            event_weights = self.config.get('event_weights', {})
            active_events = self.config.get('active_events', [])

            for event_name in active_events:
                weight = event_weights.get(event_name, 0)
                if weight == 0:
                    continue

                # Calculate samples for this event
                event_sample_count = int(total_samples * weight)

                # Fetch samples from this event
                cursor.execute("""
                    SELECT * FROM archive_samples
                    WHERE event_name = ?
                    ORDER BY RANDOM()
                    LIMIT ?
                """, (event_name, event_sample_count))

                event_samples = cursor.fetchall()

                # Convert to dictionaries
                for row in event_samples:
                    sample = self._row_to_dict(row)
                    samples.append(sample)

        else:
            # Load random samples without weighting
            cursor.execute("""
                SELECT * FROM archive_samples
                ORDER BY RANDOM()
                LIMIT ?
            """, (total_samples,))

            rows = cursor.fetchall()
            for row in rows:
                sample = self._row_to_dict(row)
                samples.append(sample)

        conn.close()

        print(f"[ARCHIVE] Loaded {len(samples)} samples from rare event archive")
        return samples

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to sample dictionary"""
        # Parse features JSON
        features = json.loads(row['features']) if row['features'] else {}

        # Create sample in same format as backtest samples
        sample = {
            'symbol': row['symbol'],
            'date': row['date'],
            'entry_price': row['entry_price'],
            'return_pct': row['return_pct'],
            'label': row['label'],
            'exit_reason': row['exit_reason'],
            'features': features,
            'event_name': row['event_name']  # Additional metadata
        }

        return sample

    def get_event_stats(self, event_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific event

        Args:
            event_name: Event identifier

        Returns:
            Dictionary with event statistics
        """
        if not self.archive_exists:
            return {}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get sample count
        cursor.execute("""
            SELECT COUNT(*) as count FROM archive_samples
            WHERE event_name = ?
        """, (event_name,))
        total = cursor.fetchone()[0]

        # Get label distribution
        cursor.execute("""
            SELECT label, COUNT(*) as count
            FROM archive_samples
            WHERE event_name = ?
            GROUP BY label
        """, (event_name,))

        label_dist = {}
        for row in cursor.fetchall():
            label_map = {0: 'buy', 1: 'hold', 2: 'sell'}
            label_dist[label_map[row[0]]] = row[1]

        conn.close()

        return {
            'event_name': event_name,
            'total_samples': total,
            'label_distribution': label_dist
        }

    def get_archive_stats(self) -> Dict[str, Any]:
        """
        Get overall archive statistics

        Returns:
            Dictionary with archive-wide statistics
        """
        if not self.archive_exists:
            return {'error': 'Archive not found'}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total samples
        cursor.execute("SELECT COUNT(*) FROM archive_samples")
        total_samples = cursor.fetchone()[0]

        # Per-event counts
        cursor.execute("""
            SELECT event_name, COUNT(*) as count
            FROM archive_samples
            GROUP BY event_name
        """)

        event_counts = {}
        for row in cursor.fetchall():
            event_counts[row[0]] = row[1]

        # Overall label distribution
        cursor.execute("""
            SELECT label, COUNT(*) as count
            FROM archive_samples
            GROUP BY label
        """)

        label_dist = {}
        for row in cursor.fetchall():
            label_map = {0: 'buy', 1: 'hold', 2: 'sell'}
            label_dist[label_map[row[0]]] = row[1]

        conn.close()

        return {
            'total_samples': total_samples,
            'event_counts': event_counts,
            'label_distribution': label_dist,
            'config': self.config
        }

    def validate_archive(self) -> bool:
        """
        Validate archive integrity

        Returns:
            True if archive is valid
        """
        if not self.archive_exists:
            print("[ERROR] Archive database not found")
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='archive_samples'
            """)
            if not cursor.fetchone():
                print("[ERROR] archive_samples table not found")
                return False

            # Check for samples
            cursor.execute("SELECT COUNT(*) FROM archive_samples")
            count = cursor.fetchone()[0]

            if count == 0:
                print("[WARNING] Archive is empty")
                return False

            # Check all events present
            active_events = self.config.get('active_events', [])
            cursor.execute("""
                SELECT DISTINCT event_name FROM archive_samples
            """)
            found_events = [row[0] for row in cursor.fetchall()]

            missing = set(active_events) - set(found_events)
            if missing:
                print(f"[WARNING] Missing events in archive: {missing}")

            conn.close()

            print(f"[OK] Archive validated: {count} samples, {len(found_events)} events")
            return True

        except Exception as e:
            print(f"[ERROR] Archive validation failed: {e}")
            return False


def create_archive_database(db_path: str):
    """
    Create archive database with schema

    Args:
        db_path: Path to database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create archive_samples table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS archive_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            entry_price REAL,
            return_pct REAL,
            label INTEGER,
            exit_reason TEXT,
            features TEXT,
            created_at TEXT,
            UNIQUE(event_name, symbol, date)
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event ON archive_samples(event_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON archive_samples(symbol)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON archive_samples(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_label ON archive_samples(label)")

    conn.commit()
    conn.close()

    print(f"[OK] Archive database created at {db_path}")


if __name__ == '__main__':
    # Test archive functionality
    print("Testing Rare Event Archive...")

    archive = RareEventArchive()

    if archive.archive_exists:
        # Validate
        valid = archive.validate_archive()

        if valid:
            # Get stats
            stats = archive.get_archive_stats()
            print("\nArchive Statistics:")
            print(f"  Total Samples: {stats['total_samples']}")
            print(f"\n  Per-Event Counts:")
            for event, count in stats['event_counts'].items():
                print(f"    {event}: {count}")

            print(f"\n  Label Distribution:")
            for label, count in stats['label_distribution'].items():
                print(f"    {label}: {count}")

            # Test loading
            print("\nTesting sample loading...")
            samples = archive.load_archive_samples(total_samples=100)
            print(f"  Loaded {len(samples)} samples")

            if len(samples) > 0:
                print(f"\n  Sample example:")
                print(f"    Symbol: {samples[0]['symbol']}")
                print(f"    Date: {samples[0]['date']}")
                print(f"    Event: {samples[0]['event_name']}")
                print(f"    Label: {samples[0]['label']}")
                print(f"    Features: {len(samples[0]['features'])} features")
    else:
        print("\n[INFO] Archive not yet generated")
        print("[INFO] Run: python generate_rare_event_archive.py")
