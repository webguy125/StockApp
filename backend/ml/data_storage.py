"""
Data Storage Module for ML Infrastructure
Manages SQLite database for OHLCV data and indicator outputs
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Database file location
DB_PATH = Path(__file__).parent / 'local_ohlcv.db'


class OHLCVDatabase:
    """
    SQLite database manager for OHLCV data and indicator outputs
    """

    def __init__(self, db_path=None):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file (default: local_ohlcv.db)
        """
        self.db_path = db_path or DB_PATH
        self.conn = None
        self._connect()
        self._initialize_schema()

    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"📊 Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    def _initialize_schema(self):
        """Create tables and indexes if they don't exist"""
        cursor = self.conn.cursor()

        # Main OHLCV table with indicator columns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp DATETIME NOT NULL,
                asset TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                -- Triad Trend Pulse indicator outputs
                triad_weighted_trend REAL,
                triad_oscillator REAL,
                triad_short_trend REAL,
                triad_pivot_high INTEGER,
                triad_pivot_low INTEGER,
                triad_pivot_score REAL,
                PRIMARY KEY (timestamp, asset)
            )
        """)

        # Indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_asset_timestamp
            ON ohlcv (asset, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON ohlcv (timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_asset
            ON ohlcv (asset)
        """)

        self.conn.commit()
        logger.info("✅ Database schema initialized")

    def insert_ohlcv(self, df, asset, replace=False):
        """
        Insert OHLCV data into database

        Args:
            df: pandas DataFrame with columns [timestamp, open, high, low, close, volume]
            asset: Asset identifier (e.g., 'BTC/USD', 'AAPL')
            replace: If True, replace existing data; if False, skip duplicates

        Returns:
            int: Number of rows inserted
        """
        try:
            # Ensure timestamp column
            if 'timestamp' not in df.columns:
                df = df.reset_index()
                df.rename(columns={'index': 'timestamp', 'Date': 'timestamp',
                                   'Datetime': 'timestamp'}, inplace=True)

            # Add asset column
            df['asset'] = asset

            # Convert to float32 for memory efficiency
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(np.float32)

            # Select columns
            cols = ['timestamp', 'asset', 'open', 'high', 'low', 'close', 'volume']
            df_insert = df[cols].copy()

            # Convert timestamps to string format
            df_insert['timestamp'] = pd.to_datetime(df_insert['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Insert with conflict handling
            if replace:
                df_insert.to_sql('ohlcv', self.conn, if_exists='append', index=False,
                                 method='multi', chunksize=1000)
            else:
                # Insert or ignore duplicates
                cursor = self.conn.cursor()
                inserted = 0
                for _, row in df_insert.iterrows():
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO ohlcv
                            (timestamp, asset, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, tuple(row))
                        if cursor.rowcount > 0:
                            inserted += 1
                    except Exception as e:
                        logger.warning(f"Row insert failed: {e}")

                self.conn.commit()
                logger.info(f"✅ Inserted {inserted} rows for {asset}")
                return inserted

            self.conn.commit()
            logger.info(f"✅ Inserted {len(df_insert)} rows for {asset}")
            return len(df_insert)

        except Exception as e:
            logger.error(f"❌ Insert failed: {e}")
            self.conn.rollback()
            raise

    def update_indicator_outputs(self, df, asset):
        """
        Update indicator output columns for existing OHLCV data

        Args:
            df: DataFrame with timestamp and indicator columns
            asset: Asset identifier

        Returns:
            int: Number of rows updated
        """
        try:
            cursor = self.conn.cursor()
            updated = 0

            for _, row in df.iterrows():
                timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

                cursor.execute("""
                    UPDATE ohlcv SET
                        triad_weighted_trend = ?,
                        triad_oscillator = ?,
                        triad_short_trend = ?,
                        triad_pivot_high = ?,
                        triad_pivot_low = ?,
                        triad_pivot_score = ?
                    WHERE timestamp = ? AND asset = ?
                """, (
                    row.get('weighted_trend'),
                    row.get('oscillator'),
                    row.get('short_trend'),
                    int(row.get('pivot_high', 0)),
                    int(row.get('pivot_low', 0)),
                    row.get('pivot_score'),
                    timestamp,
                    asset
                ))

                if cursor.rowcount > 0:
                    updated += 1

            self.conn.commit()
            logger.info(f"✅ Updated {updated} indicator outputs for {asset}")
            return updated

        except Exception as e:
            logger.error(f"❌ Update failed: {e}")
            self.conn.rollback()
            raise

    def get_ohlcv(self, asset, start_date=None, end_date=None, limit=None):
        """
        Retrieve OHLCV data from database

        Args:
            asset: Asset identifier
            start_date: Start date (datetime or string)
            end_date: End date (datetime or string)
            limit: Maximum number of rows to return

        Returns:
            pandas DataFrame with OHLCV data
        """
        try:
            query = "SELECT * FROM ohlcv WHERE asset = ?"
            params = [asset]

            if start_date:
                query += " AND timestamp >= ?"
                params.append(pd.to_datetime(start_date).strftime('%Y-%m-%d %H:%M:%S'))

            if end_date:
                query += " AND timestamp <= ?"
                params.append(pd.to_datetime(end_date).strftime('%Y-%m-%d %H:%M:%S'))

            query += " ORDER BY timestamp ASC"

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql_query(query, self.conn, params=params)

            # Convert timestamp to datetime
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            logger.info(f"📊 Retrieved {len(df)} rows for {asset}")
            return df

        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise

    def get_latest(self, asset, n=100):
        """
        Get the most recent N bars for an asset

        Args:
            asset: Asset identifier
            n: Number of bars to retrieve

        Returns:
            pandas DataFrame
        """
        return self.get_ohlcv(asset, limit=n)

    def cleanup_old_data(self, days_to_keep=30):
        """
        Delete old 1-minute data beyond retention period

        Args:
            days_to_keep: Number of days to retain (default: 30)

        Returns:
            int: Number of rows deleted
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d %H:%M:%S')

            cursor = self.conn.cursor()
            cursor.execute("""
                DELETE FROM ohlcv
                WHERE timestamp < ?
                AND asset LIKE '%1min%'
            """, (cutoff_date,))

            deleted = cursor.rowcount
            self.conn.commit()

            logger.info(f"🗑️ Cleaned up {deleted} old rows (older than {days_to_keep} days)")
            return deleted

        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            self.conn.rollback()
            raise

    def get_assets(self):
        """
        Get list of all assets in database

        Returns:
            list: Asset identifiers
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT asset FROM ohlcv ORDER BY asset")
            assets = [row[0] for row in cursor.fetchall()]
            return assets
        except Exception as e:
            logger.error(f"❌ Asset query failed: {e}")
            raise

    def get_date_range(self, asset):
        """
        Get the date range of data for an asset

        Args:
            asset: Asset identifier

        Returns:
            tuple: (earliest_date, latest_date)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM ohlcv
                WHERE asset = ?
            """, (asset,))
            result = cursor.fetchone()

            if result and result[0]:
                return (
                    pd.to_datetime(result[0]),
                    pd.to_datetime(result[1])
                )
            return (None, None)

        except Exception as e:
            logger.error(f"❌ Date range query failed: {e}")
            raise

    def vacuum(self):
        """
        Optimize database by running VACUUM command
        """
        try:
            self.conn.execute("VACUUM")
            logger.info("✅ Database optimized (VACUUM completed)")
        except Exception as e:
            logger.error(f"❌ VACUUM failed: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("📊 Database connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Singleton instance
_db_instance = None

def get_database():
    """
    Get singleton database instance

    Returns:
        OHLCVDatabase instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = OHLCVDatabase()
    return _db_instance
