"""
Event Archive
Stores high-severity rare events for future training and retrieval.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class EventArchive:
    """
    Archive system for storing and retrieving rare high-severity events.
    Implements k-nearest neighbors retrieval on embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize event archive.

        Args:
            config: Archive configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.severity_threshold = config.get('severity_threshold', 0.80)
        self.post_event_window_days = config.get('post_event_window_days', 60)
        self.stored_fields = config.get('stored_fields', [])

        # Initialize database
        self.db_path = self._get_db_path()
        self._initialize_database()

        logger.info(f"[EVENT_ARCHIVE] Initialized (threshold: {self.severity_threshold})")

    def _get_db_path(self) -> str:
        """Get database path for event archive."""
        base_dir = Path(__file__).parent.parent.parent / 'data'
        base_dir.mkdir(exist_ok=True)
        return str(base_dir / 'event_archive.db')

    def _initialize_database(self):
        """Initialize SQLite database for event storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rare_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT,
                event_subtype TEXT,
                event_severity REAL,
                impact_dividend REAL,
                impact_liquidity REAL,
                impact_credit REAL,
                impact_growth REAL,
                confidence REAL,
                sentiment_score REAL,
                pre_post_returns REAL,
                drawdown_outcomes REAL,
                recovery_time_days INTEGER,
                volatility_spike_magnitude REAL,
                stored_at TEXT,
                UNIQUE(ticker, timestamp, event_type)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
            ON rare_events (ticker, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_severity
            ON rare_events (event_severity)
        """)

        conn.commit()
        conn.close()

        logger.info(f"[EVENT_ARCHIVE] Database initialized: {self.db_path}")

    def store_events(self, events: pd.DataFrame) -> int:
        """
        Store high-severity events to archive.

        Args:
            events: Classified events DataFrame

        Returns:
            Number of events archived
        """
        if not self.enabled:
            return 0

        # Filter for high-severity events
        high_severity = events[events['event_severity'] >= self.severity_threshold].copy()

        if len(high_severity) == 0:
            return 0

        # Add storage timestamp
        high_severity['stored_at'] = datetime.now().isoformat()

        # Store to database
        conn = sqlite3.connect(self.db_path)

        stored_count = 0
        for _, event in high_severity.iterrows():
            try:
                self._insert_event(conn, event)
                stored_count += 1
            except sqlite3.IntegrityError:
                # Event already exists
                pass
            except Exception as e:
                logger.warning(f"[EVENT_ARCHIVE] Failed to store event: {e}")

        conn.commit()
        conn.close()

        logger.info(f"[EVENT_ARCHIVE] Stored {stored_count} rare events")
        return stored_count

    def _insert_event(self, conn: sqlite3.Connection, event: pd.Series):
        """Insert single event into database."""
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO rare_events (
                ticker, timestamp, event_type, event_subtype,
                event_severity, impact_dividend, impact_liquidity,
                impact_credit, impact_growth, confidence,
                sentiment_score, stored_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.get('ticker', ''),
            str(event.get('timestamp', '')),
            event.get('event_type', ''),
            event.get('event_subtype', ''),
            float(event.get('event_severity', 0.0)),
            float(event.get('impact_dividend', 0.0)),
            float(event.get('impact_liquidity', 0.0)),
            float(event.get('impact_credit', 0.0)),
            float(event.get('impact_growth', 0.0)),
            float(event.get('confidence', 0.0)),
            float(event.get('sentiment_score', 0.0)),
            event.get('stored_at', datetime.now().isoformat())
        ))

    def retrieve_similar_events(
        self,
        event_type: str,
        severity_range: tuple = (0.7, 1.0),
        k: int = 10
    ) -> pd.DataFrame:
        """
        Retrieve similar events from archive.

        Args:
            event_type: Event type to match
            severity_range: (min, max) severity range
            k: Number of events to retrieve

        Returns:
            DataFrame with similar events
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT * FROM rare_events
            WHERE event_type = ?
            AND event_severity >= ?
            AND event_severity <= ?
            ORDER BY event_severity DESC
            LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(event_type, severity_range[0], severity_range[1], k)
        )

        conn.close()

        logger.info(f"[EVENT_ARCHIVE] Retrieved {len(df)} similar events")
        return df

    def retrieve_by_ticker(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve archived events for specific ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with archived events
        """
        conn = sqlite3.connect(self.db_path)

        if start_date is None and end_date is None:
            query = "SELECT * FROM rare_events WHERE ticker = ? ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn, params=(ticker,))
        else:
            query = """
                SELECT * FROM rare_events
                WHERE ticker = ?
                AND timestamp >= ?
                AND timestamp <= ?
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=(ticker, start_date.isoformat(), end_date.isoformat())
            )

        conn.close()

        return df

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get archive statistics.

        Returns:
            Statistics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total events
        cursor.execute("SELECT COUNT(*) FROM rare_events")
        total_events = cursor.fetchone()[0]

        # Events by type
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM rare_events
            GROUP BY event_type
            ORDER BY count DESC
        """)
        events_by_type = dict(cursor.fetchall())

        # Average severity
        cursor.execute("SELECT AVG(event_severity) FROM rare_events")
        avg_severity = cursor.fetchone()[0] or 0.0

        # Date range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM rare_events")
        date_range = cursor.fetchone()

        conn.close()

        return {
            'total_events': total_events,
            'events_by_type': events_by_type,
            'average_severity': avg_severity,
            'date_range': date_range,
            'severity_threshold': self.severity_threshold,
            'enabled': self.enabled
        }

    def clear_archive(self) -> int:
        """
        Clear all archived events.

        Returns:
            Number of events deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM rare_events")
        count = cursor.fetchone()[0]

        cursor.execute("DELETE FROM rare_events")
        conn.commit()
        conn.close()

        logger.info(f"[EVENT_ARCHIVE] Cleared {count} events")
        return count
